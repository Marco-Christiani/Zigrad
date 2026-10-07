//! TVM runtime module and tensor wrappers.
//!
//! Covers `tvm/runtime/module.h` (RuntimeModule) and `tvm/runtime/ndarray.h`
//! (Tensor via DLPack bridge).
const std = @import("std");
const ffi = @import("ffi.zig");
const c = @import("c.zig");
const dlpack = @import("../dlpack.zig");
const Value = ffi.Value;
const TvmError = ffi.TvmError;

const log = std.log.scoped(.@"zg/tvm_runtime");

/// Callable function loaded from a TVM runtime module.
pub const Function = struct {
    function: ffi.PackedFunction,

    fn take(value: *ffi.OwnedValue) !Function {
        return .{ .function = try .take(value) };
    }

    /// Invoke the function with tensor arguments.
    pub fn call(
        self: *const Function,
        allocator: std.mem.Allocator,
        tensors: []const Tensor,
    ) !void {
        const values = try allocator.alloc(Value, tensors.len);
        defer allocator.free(values);
        for (tensors, values) |tensor, *value| value.* = tensor.as_value();
        var result = try self.function.call(allocator, values);
        defer result.deinit();
        if (!result.borrow().is_none()) return error.TvmTypeMismatch;
    }

    /// Release the TVM function.
    pub fn deinit(self: *Function) void {
        self.function.deinit();
        self.* = undefined;
    }
};

/// Device-aware TVM time evaluator.
pub const TimeEvaluator = struct {
    function: ffi.PackedFunction,
    repeats: usize,

    /// Measure the function and return one average per repeat.
    pub fn measure(
        self: *const TimeEvaluator,
        allocator: std.mem.Allocator,
        tensors: []const Tensor,
    ) ![]f64 {
        const values = try allocator.alloc(Value, tensors.len);
        defer allocator.free(values);
        for (tensors, values) |tensor, *value| value.* = tensor.as_value();

        var result = try self.function.call(allocator, values);
        defer result.deinit();
        const encoded = try result.borrow().dupe_bytes(allocator);
        defer allocator.free(encoded);
        if (encoded.len != self.repeats * @sizeOf(f64))
            return error.InvalidTimingResult;

        const times = try allocator.alloc(f64, self.repeats);
        errdefer allocator.free(times);
        for (times, 0..) |*time, index| {
            time.* = std.mem.bytesToValue(
                f64,
                encoded[index * @sizeOf(f64) ..][0..@sizeOf(f64)],
            );
            if (!std.math.isFinite(time.*) or time.* <= 0.0)
                return error.InvalidTimingResult;
        }
        return times;
    }

    /// Release the TVM evaluator.
    pub fn deinit(self: *TimeEvaluator) void {
        self.function.deinit();
        self.* = undefined;
    }
};

/// TVM runtime module.
pub const RuntimeModule = struct {
    object: ffi.Object,

    /// Take a packed result as a runtime module.
    pub fn take(value: *ffi.OwnedValue) TvmError!RuntimeModule {
        try value.borrow().require_type_index(c.kTVMFFIModule);
        return .{ .object = try .take(value) };
    }

    /// Load a compiled module from disk.
    pub fn load_from_file(
        /// Allocator used by the packed call.
        allocator: std.mem.Allocator,
        /// Module path interpreted by TVM from its suffix.
        path: [:0]const u8,
    ) !RuntimeModule {
        return try ffi.call_global_take(RuntimeModule, allocator, "ffi.ModuleLoadFromFile", &.{
            Value.str(path),
        });
    }

    /// Get a packed function from this module by name.
    pub fn get_function(
        self: RuntimeModule,
        /// Allocator used by the packed call.
        allocator: std.mem.Allocator,
        /// Registered function name.
        name: [:0]const u8,
        /// Search imported modules when the function is absent locally.
        query_imports: bool,
    ) !Function {
        var result = try ffi.call_global(allocator, "ffi.ModuleGetFunction", &.{
            self.as_value(), Value.str(name), Value.boolean(query_imports),
        });
        errdefer result.deinit();
        if (result.borrow().is_none()) {
            log.err("TVM module function not found: {s}", .{name});
            return error.TvmFunctionNotFound;
        }
        return try .take(&result);
    }

    /// Create TVM's device-aware evaluator for one module function.
    pub fn time_evaluator(
        self: RuntimeModule,
        /// Allocator used by the packed call.
        allocator: std.mem.Allocator,
        /// Module function to measure.
        name: [:0]const u8,
        /// Device type used by TVM's timer implementation.
        device_type: dlpack.DeviceType,
        /// Device index within `device_type`.
        device_ordinal: i32,
        /// Repetition and cache policy.
        options: TimeEvaluatorOptions,
    ) !TimeEvaluator {
        try options.validate();
        var result = try ffi.call_global(allocator, "runtime.RPCTimeEvaluator", &.{
            self.as_value(),
            Value.str(name),
            Value.int(@backingInt(device_type)),
            Value.int(device_ordinal),
            Value.int(options.number),
            Value.int(options.repeats),
            Value.int(options.min_repeat_ms),
            Value.int(100), // limit_zero_time_iterations
            Value.int(0), // cooldown_interval_ms
            Value.int(1), // repeats_to_cooldown
            Value.int(options.cache_flush_bytes),
            Value.str(""), // f_preproc
        });
        errdefer result.deinit();
        return .{
            .function = try .take(&result),
            .repeats = @intCast(options.repeats),
        };
    }

    /// Write the module using a TVM-supported format.
    pub fn write_to_file(
        self: RuntimeModule,
        /// Allocator used by the packed call.
        allocator: std.mem.Allocator,
        /// Destination file path.
        path: [:0]const u8,
        /// TVM format name such as `o`, `so`, or `ptx`.
        format: [:0]const u8,
    ) !void {
        try ffi.call_global_void(allocator, "ffi.ModuleWriteToFile", &.{
            self.as_value(),
            Value.str(path),
            Value.str(format),
        });
        log.debug("wrote module to {s} (format={s})", .{ path, format });
    }

    /// Pack device module imports into an LLVM blob (for CUDA .so linking).
    pub fn pack_imports_to_llvm(self: RuntimeModule, allocator: std.mem.Allocator) !RuntimeModule {
        return try ffi.call_global_take(RuntimeModule, allocator, "runtime.ModulePackImportsToLLVM", &.{
            self.as_value(),
            Value.boolean(false), // system_lib
            Value.str("llvm"),
            Value.str(""),
        });
    }

    pub fn as_value(self: *const RuntimeModule) Value {
        return self.object.as_value();
    }

    /// Release the TVM runtime module.
    pub fn deinit(self: *RuntimeModule) void {
        self.object.deinit();
        self.* = undefined;
    }
};

/// Policy for TVM's device-aware time evaluator.
pub const TimeEvaluatorOptions = struct {
    /// Initial invocations averaged into each repeat.
    number: i32 = 3,
    /// Repeat averages returned to the caller.
    repeats: i32 = 15,
    /// Minimum duration of each repeat before averaging.
    min_repeat_ms: i32 = 10,
    /// Device bytes copied before each repeat to perturb caches.
    cache_flush_bytes: i32 = 0,

    /// Validate timing counts and byte limits.
    pub fn validate(self: TimeEvaluatorOptions) error{InvalidTimeEvaluatorOptions}!void {
        if (self.number <= 0 or self.repeats <= 0 or
            self.min_repeat_ms < 0 or self.cache_flush_bytes < 0)
        {
            return error.InvalidTimeEvaluatorOptions;
        }
    }
};

test TimeEvaluatorOptions {
    try std.testing.expectError(
        error.InvalidTimeEvaluatorOptions,
        (TimeEvaluatorOptions{ .number = 0 }).validate(),
    );
    try std.testing.expectError(
        error.InvalidTimeEvaluatorOptions,
        (TimeEvaluatorOptions{ .repeats = 0 }).validate(),
    );
    try std.testing.expectError(
        error.InvalidTimeEvaluatorOptions,
        (TimeEvaluatorOptions{ .min_repeat_ms = -1 }).validate(),
    );
    try std.testing.expectError(
        error.InvalidTimeEvaluatorOptions,
        (TimeEvaluatorOptions{ .cache_flush_bytes = -1 }).validate(),
    );
}

/// TVM tensor.
pub const Tensor = struct {
    object: ffi.Object,
    /// Number of logical tensor bytes accepted by copy operations.
    byte_count: usize,

    /// Create a TVM tensor that assumes ownership of a DLPack tensor.
    pub fn from_dlpack(
        /// Tensor transferred to TVM on success. The caller retains ownership
        ///  on error.
        managed: *dlpack.ManagedTensor,
    ) TvmError!Tensor {
        if (managed.dl_tensor.ndim < 0) return error.InvalidTensorLayout;
        const rank: usize = @intCast(managed.dl_tensor.ndim);
        const bytes = dlpack.byte_count(
            managed.dl_tensor.shape[0..rank],
            managed.dl_tensor.dtype,
        ) catch return error.InvalidTensorLayout;
        var out: c.TVMFFIObjectHandle = null;
        const status = c.TVMFFITensorFromDLPack(@ptrCast(managed), 0, 0, &out);
        if (status != 0)
            return ffi.call_error("TVMFFITensorFromDLPack", status);
        if (out == null) return error.TvmValueAbsent;
        var value = ffi.OwnedValue{
            .value = Value.from_object(out, c.kTVMFFITensor),
        };
        errdefer value.deinit();
        return .{ .object = try .take(&value), .byte_count = bytes };
    }

    /// Allocate a TVM tensor on the given device and copy data from host.
    pub fn allocate(
        /// Allocator used to construct the runtime call.
        allocator: std.mem.Allocator,
        /// Host bytes copied into the tensor. The byte count must match
        ///  `shape` and `dtype`.
        data: []const u8,
        /// Tensor dimensions copied by the TVM runtime.
        shape: []const i64,
        /// Element representation of `data`.
        dtype: dlpack.DataType,
        /// DLPack device type used for allocation.
        device_type: dlpack.DeviceType,
        /// Device index within `device_type`.
        device_ordinal: i32,
    ) TvmError!Tensor {
        const bytes = dlpack.byte_count(shape, dtype) catch
            return error.InvalidTensorLayout;
        if (data.len != bytes) return error.InvalidTensorDataSize;
        const data_len = std.math.cast(i64, data.len) orelse
            return error.InvalidTensorDataSize;

        const shape_vals = try allocator.alloc(Value, shape.len);
        defer allocator.free(shape_vals);
        for (shape, 0..) |dim, i| {
            shape_vals[i] = Value.int(dim);
        }
        var shape_obj = try ffi.call_global(allocator, "ffi.Shape", shape_vals);
        defer shape_obj.deinit();
        try shape_obj.borrow().require_type_index(c.kTVMFFIShape);

        var tensor_value = try ffi.call_global(allocator, "runtime.TVMTensorAllocWithScope", &.{
            shape_obj.borrow(),
            dtype_value(dtype),
            device_value(device_type, device_ordinal),
            Value.none(),
        });
        errdefer tensor_value.deinit();
        try tensor_value.borrow().require_type_index(c.kTVMFFITensor);

        try ffi.call_global_void(allocator, "runtime.TVMTensorCopyFromBytes", &.{
            tensor_value.borrow(),
            ptr_value(@ptrCast(@constCast(data.ptr))),
            Value.int(data_len),
        });

        return .{ .object = try .take(&tensor_value), .byte_count = bytes };
    }

    /// Copy tensor data back to host memory.
    pub fn copy_to_host(
        self: Tensor,
        /// Allocator used to construct the runtime call.
        allocator: std.mem.Allocator,
        /// Destination whose byte count matches the tensor allocation.
        dest: []u8,
    ) TvmError!void {
        if (dest.len != self.byte_count) return error.InvalidTensorDataSize;
        const dest_len = std.math.cast(i64, dest.len) orelse
            return error.InvalidTensorDataSize;
        try ffi.call_global_void(allocator, "runtime.TVMTensorCopyToBytes", &.{
            self.as_value(),
            ptr_value(@ptrCast(dest.ptr)),
            Value.int(dest_len),
        });
    }

    pub fn as_value(self: *const Tensor) Value {
        return self.object.as_value();
    }

    /// Release the TVM tensor.
    pub fn deinit(self: *Tensor) void {
        self.object.deinit();
        self.* = undefined;
    }
};

// Private helpers for constructing special Value types needed by Tensor methods.

fn device_value(device_type: dlpack.DeviceType, device_id: i32) Value {
    var v = std.mem.zeroes(c.TVMFFIAny);
    v.type_index = c.kTVMFFIDevice;
    v.unnamed_1.v_device = .{ .device_type = @intCast(@backingInt(device_type)), .device_id = device_id };
    return .{ .raw = v };
}

fn dtype_value(dtype: dlpack.DataType) Value {
    var v = std.mem.zeroes(c.TVMFFIAny);
    v.type_index = c.kTVMFFIDataType;
    v.unnamed_1.v_dtype = .{
        .code = @intCast(@backingInt(dtype.code)),
        .bits = dtype.bits,
        .lanes = dtype.lanes,
    };
    return .{ .raw = v };
}

fn ptr_value(ptr: *anyopaque) Value {
    var v = std.mem.zeroes(c.TVMFFIAny);
    v.type_index = c.kTVMFFIOpaquePtr;
    v.unnamed_1.v_int64 = @bitCast(@intFromPtr(ptr));
    return .{ .raw = v };
}

/// Select the stream used by subsequent TVM calls on one device.
pub fn set_stream(
    allocator: std.mem.Allocator,
    device_type: dlpack.DeviceType,
    device_ordinal: i32,
    stream: *anyopaque,
) TvmError!void {
    try ffi.call_global_void(allocator, "runtime.TVMSetStream", &.{
        Value.int(@backingInt(device_type)),
        Value.int(device_ordinal),
        ptr_value(stream),
    });
}
