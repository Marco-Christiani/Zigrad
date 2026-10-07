//! TVM IR, TIR, and Target wrappers.
//!
//! Covers the `tvm/ir/`, `tvm/tir/`, `tvm/target/`, and `tvm/te/` operations
//!  required by TIR lowering.
const std = @import("std");
const ffi = @import("ffi.zig");
const container = @import("container.zig");
const c = @import("c.zig");
const dlpack = @import("../dlpack.zig");
const Value = ffi.Value;
const TvmError = ffi.TvmError;
const config = @import("../../tvm/config.zig");
const TargetKind = config.TargetKind;
const Device = @import("device").Device;

const log = std.log.scoped(.@"zg/tvm_tir");

/// TVM TensorIR function.
pub const PrimFunc = struct {
    object: ffi.Object,

    /// Take a packed result as a TIR function.
    pub fn take(value: *ffi.OwnedValue) TvmError!PrimFunc {
        try value.borrow().require_instance("tir.PrimFunc");
        return .{ .object = try .take(value) };
    }

    fn as_value(self: *const PrimFunc) Value {
        return self.object.as_value();
    }

    /// Copy the function with an explicit alignment for its external buffers.
    ///  Body buffers retain their data variables. Internal allocations retain their alignment.
    pub fn with_buffer_alignment(self: *const PrimFunc, allocator: std.mem.Allocator, alignment: u32) !PrimFunc {
        std.debug.assert(std.math.isPowerOfTwo(alignment));
        const field_names = [_][]const u8{ "params", "body", "ret_type", "buffer_map", "attrs", "span" };
        var fields: [field_names.len]ffi.OwnedValue = undefined;
        var initialized: usize = 0;
        defer for (fields[0..initialized]) |*field| field.deinit();
        for (field_names, &fields) |name, *field| {
            field.* = try ffi.get_field(self.as_value(), name);
            initialized += 1;
        }
        var params = container.Array{ .object = try ffi.Object.retain(fields[0].borrow()) };
        defer params.deinit();
        var buffers = try container.Map.retain(fields[3].borrow());
        defer buffers.deinit();
        var pairs: std.ArrayList(Value) = .empty;
        defer pairs.deinit(allocator);
        var owned: std.ArrayList(ffi.OwnedValue) = .empty;
        defer {
            for (owned.items) |*value| value.deinit();
            owned.deinit(allocator);
        }
        for (0..try params.len(allocator)) |i| {
            var param = try params.get(allocator, i);
            defer param.deinit();
            if (!try buffers.contains(allocator, param.borrow())) continue;
            var buffer = try buffers.get(allocator, param.borrow());
            defer buffer.deinit();
            const names = [_][]const u8{ "data", "dtype", "shape", "strides", "elem_offset", "name", "offset_factor", "buffer_type", "axis_separators", "span" };
            var parts: [names.len]ffi.OwnedValue = undefined;
            var count: usize = 0;
            defer for (parts[0..count]) |*part| part.deinit();
            for (names, &parts) |name, *part| {
                part.* = try ffi.get_field(buffer.borrow(), name);
                count += 1;
            }
            const replacement = blk: {
                var value = try ffi.call_global(allocator, "tir.Buffer", &.{
                    parts[0].borrow(),                                                                        parts[1].borrow(), parts[2].borrow(),    parts[3].borrow(),
                    parts[4].borrow(),                                                                        parts[5].borrow(), Value.int(alignment), parts[6].borrow(),
                    Value.str(if (try parts[7].borrow().require_int() == 2) "auto_broadcast" else "default"), parts[8].borrow(), parts[9].borrow(),
                });
                errdefer value.deinit();
                try owned.append(allocator, value);
                break :blk value.borrow();
            };
            try pairs.appendSlice(allocator, &.{ param.borrow(), replacement });
        }
        var buffer_map = try container.Map.from_pairs(allocator, pairs.items);
        defer buffer_map.deinit();
        return try ffi.call_global_take(PrimFunc, allocator, "tir.PrimFunc", &.{
            fields[0].borrow(), fields[1].borrow(), fields[2].borrow(), buffer_map.as_value(), fields[4].borrow(), fields[5].borrow(),
        });
    }

    /// Release the TVM TIR function.
    pub fn deinit(self: *PrimFunc) void {
        self.object.deinit();
        self.* = undefined;
    }
};

/// TVM IR module.
pub const IRModule = struct {
    object: ffi.Object,

    /// Take a packed result as an IR module.
    pub fn take(value: *ffi.OwnedValue) TvmError!IRModule {
        try value.borrow().require_instance("ir.IRModule");
        return .{ .object = try .take(value) };
    }

    /// Copy an IR module reference.
    pub fn retain(value: Value) TvmError!IRModule {
        try value.require_instance("ir.IRModule");
        return .{ .object = try ffi.Object.retain(value) };
    }

    /// Create a module containing one named entry function.
    pub fn from_entry(
        /// Allocator used by TVM packed calls.
        allocator: std.mem.Allocator,
        /// Name assigned to the function and its global symbol.
        name: [:0]const u8,
        /// Function referenced by the new module.
        function: *const PrimFunc,
    ) !IRModule {
        var entry = try ffi.call_global(allocator, "ir.BaseFuncWithAttr", &.{
            function.as_value(), Value.str("global_symbol"), Value.str(name),
        });
        defer entry.deinit();
        var global = try ffi.call_global(
            allocator,
            "ir.GlobalVar",
            &.{Value.str(name)},
        );
        defer global.deinit();
        var functions = try container.Map.from_pairs(allocator, &.{
            global.borrow(),
            entry.borrow(),
        });
        defer functions.deinit();
        var global_info = try container.Map.from_pairs(allocator, &.{});
        defer global_info.deinit();
        return try ffi.call_global_take(IRModule, allocator, "ir.IRModule", &.{
            functions.as_value(), Value.none(), global_info.as_value(),
        });
    }

    /// Apply a single TIR transform pass to this module (in-place replacement).
    pub fn apply_pass(self: *IRModule, allocator: std.mem.Allocator, pass: TirPass) !void {
        var pass_value = try pass.create(allocator);
        defer pass_value.deinit();

        const old_ptr = self.object.ptr();

        var result = try ffi.call_global(allocator, "transform.RunPass", &.{
            pass_value.borrow(),
            self.as_value(),
        });
        errdefer result.deinit();

        try result.borrow().require_instance("ir.IRModule");
        const new_object = try ffi.Object.take(&result);
        const new_ptr = new_object.ptr();
        // Keep the reference returned by RunPass and release the prior one.
        self.object.deinit();
        self.object = new_object;
        log.debug("applied {s} (object {s}, type {d})", .{
            pass.name(),
            if (old_ptr == new_ptr) "retained" else "replaced",
            self.as_value().raw.type_index,
        });
    }

    pub fn as_value(self: *const IRModule) Value {
        return self.object.as_value();
    }

    /// Copy this module's serialized TIR. The caller frees the returned bytes.
    pub fn dupe_json(self: *const IRModule, allocator: std.mem.Allocator) ![]u8 {
        var serialized = try ffi.call_global(allocator, "node.SaveJSON", &.{self.as_value()});
        defer serialized.deinit();
        return serialized.borrow().dupe_string(allocator);
    }

    /// Release the TVM IR module.
    pub fn deinit(self: *IRModule) void {
        self.object.deinit();
        self.* = undefined;
    }
};

/// TVM compilation target.
pub const Target = struct {
    object: ffi.Object,

    /// Take a packed result as a target.
    pub fn take(value: *ffi.OwnedValue) !Target {
        try value.borrow().require_instance("target.Target");
        return .{ .object = try .take(value) };
    }

    /// Create a Target from a TargetKind.
    ///
    /// For CPU targets, includes `-num-cores` (required by MetaSchedule).
    pub fn create(
        /// Allocator used by target detection and TVM packed calls.
        allocator: std.mem.Allocator,
        /// Target family to detect.
        kind: TargetKind,
        /// Device index used for target detection.
        device_ordinal: i32,
    ) !Target {
        var resolution = try describe(
            allocator,
            kind,
            device_ordinal,
        );
        defer resolution.deinit();
        return try create_from_description(allocator, resolution.description);
    }

    /// Parse a TVM target description without probing a device.
    pub fn from_description(
        /// Allocator used by the target constructor.
        allocator: std.mem.Allocator,
        /// Null-terminated TVM target description.
        description: [:0]const u8,
    ) !Target {
        return try create_from_description(allocator, description);
    }

    /// Resolve a compilation payload after validating target/device agreement.
    pub fn resolve(allocator: std.mem.Allocator, inputs: config.TargetInputs, device: Device) !ResolvedTarget {
        const kind = inputs.kind();
        if (!kind.accepts(device)) {
            log.err("TVM target {s} disagrees with the selected device", .{@tagName(kind)});
            return error.TargetDeviceMismatch;
        }
        var resolution = try describe(allocator, kind, device.ordinal);
        errdefer resolution.deinit();
        const target = try create_from_description(allocator, resolution.description);
        return .{
            .allocator = allocator,
            .device = device,
            .payload = switch (inputs) {
                .cpu => .{ .cpu = .{ .description = resolution.description, .target = target } },
                .cuda => |cuda| .{ .cuda = .{
                    .description = resolution.description,
                    .target = target,
                    .inputs = cuda,
                    .gpu_arch = resolution.inputs.cuda.gpu_arch,
                } },
            },
        };
    }

    /// Detect the target properties used for compilation and artifact caching.
    pub fn describe(
        /// Allocator used until the returned description is deinitialized.
        allocator: std.mem.Allocator,
        /// Target family to detect.
        kind: TargetKind,
        /// Device index used for target detection.
        device_ordinal: i32,
    ) !TargetDescription {
        return switch (kind) {
            .cpu => blk: {
                const cpu_count = std.Thread.getCpuCount() catch 1;
                break :blk .{
                    .allocator = allocator,
                    .description = try std.fmt.allocPrintSentinel(
                        allocator,
                        "llvm -num-cores={d}",
                        .{cpu_count},
                        0,
                    ),
                    .inputs = .cpu,
                };
            },
            .cuda => blk: {
                const properties = try detect_cuda_properties(
                    allocator,
                    device_ordinal,
                );
                defer allocator.free(properties.compute_version);
                const gpu_arch = try cuda_architecture(
                    allocator,
                    properties.compute_version,
                );
                errdefer allocator.free(gpu_arch);
                break :blk .{
                    .allocator = allocator,
                    .description = try cuda_target_description(
                        allocator,
                        properties,
                        gpu_arch,
                    ),
                    .inputs = .{ .cuda = .{ .gpu_arch = gpu_arch } },
                };
            },
        };
    }

    /// Create a composite target with a host target attached.
    pub fn with_host(
        self: Target,
        /// Allocator used by the packed call.
        allocator: std.mem.Allocator,
        /// Host target attached to the returned target.
        host: Target,
    ) !Target {
        return try ffi.call_global_take(Target, allocator, "target.WithHost", &.{
            self.as_value(),
            host.as_value(),
        });
    }

    pub fn as_value(self: *const Target) Value {
        return self.object.as_value();
    }

    /// Release the TVM target.
    pub fn deinit(self: *Target) void {
        self.object.deinit();
        self.* = undefined;
    }
};

const DetectedInputs = union(TargetKind) { cpu, cuda: struct { gpu_arch: []u8 } };

/// Resolved target owns its description and architecture until deinit.
///  CUDA configuration is borrowed for the same lifetime; callback registration copies it.
pub const ResolvedTarget = struct {
    allocator: std.mem.Allocator,
    device: Device,
    payload: union(TargetKind) {
        cpu: struct { description: [:0]u8, target: Target },
        cuda: struct { description: [:0]u8, target: Target, inputs: config.CudaInputs, gpu_arch: []u8 },
    },

    pub fn target(self: *const ResolvedTarget) Target {
        return switch (self.payload) {
            inline else => |p| p.target,
        };
    }

    pub fn description(self: *const ResolvedTarget) [:0]const u8 {
        return switch (self.payload) {
            inline else => |p| p.description,
        };
    }

    pub fn kind(self: *const ResolvedTarget) TargetKind {
        return std.meta.activeTag(self.payload);
    }

    pub fn deinit(self: *ResolvedTarget) void {
        switch (self.payload) {
            .cpu => |*cpu| {
                cpu.target.deinit();
                self.allocator.free(cpu.description);
            },
            .cuda => |*cuda| {
                cuda.target.deinit();
                self.allocator.free(cuda.description);
                self.allocator.free(cuda.gpu_arch);
            },
        }
        self.* = undefined;
    }
};

const CudaProperties = struct {
    compute_version: []const u8,
    max_threads_per_block: i64,
    thread_warp_size: i64,
    max_shared_memory_per_block: i64,
    registers_per_block: i64,
    l2_cache_size_bytes: i64,
};

fn create_from_description(
    allocator: std.mem.Allocator,
    description: [:0]const u8,
) !Target {
    return try ffi.call_global_take(
        Target,
        allocator,
        "target.Target",
        &.{Value.str(description)},
    );
}

/// Detected target properties used before constructing a TVM target.
pub const TargetDescription = struct {
    /// Allocator used to release `description` and `gpu_arch`.
    allocator: std.mem.Allocator,
    /// TVM target description.
    description: [:0]u8,
    /// CUDA owns its NVRTC architecture spelling.
    inputs: DetectedInputs,

    /// Release the copied target description.
    pub fn deinit(self: *TargetDescription) void {
        self.allocator.free(self.description);
        switch (self.inputs) {
            .cpu => {},
            .cuda => |cuda| self.allocator.free(cuda.gpu_arch),
        }
        self.* = undefined;
    }
};

fn detect_cuda_properties(
    allocator: std.mem.Allocator,
    device_ordinal: i32,
) !CudaProperties {
    if (device_ordinal < 0) return error.InvalidDeviceOrdinal;
    if (try device_int_attribute(device_ordinal, .exist) == 0) {
        log.err("TVM cannot access CUDA device {d}", .{device_ordinal});
        return error.DeviceUnavailable;
    }

    const compute_version = try device_string_attribute(
        allocator,
        device_ordinal,
        .compute_version,
    );
    errdefer allocator.free(compute_version);

    return .{
        .compute_version = compute_version,
        .max_threads_per_block = try device_int_attribute(
            device_ordinal,
            .max_threads_per_block,
        ),
        .thread_warp_size = try device_int_attribute(device_ordinal, .warp_size),
        .max_shared_memory_per_block = try device_int_attribute(
            device_ordinal,
            .max_shared_memory_per_block,
        ),
        .registers_per_block = try device_int_attribute(
            device_ordinal,
            .max_registers_per_block,
        ),
        .l2_cache_size_bytes = try device_int_attribute(
            device_ordinal,
            .l2_cache_size_bytes,
        ),
    };
}

const DeviceAttribute = enum(i64) {
    exist = 0,
    max_threads_per_block = 1,
    warp_size = 2,
    max_shared_memory_per_block = 3,
    compute_version = 4,
    max_registers_per_block = 9,
    l2_cache_size_bytes = 13,
};

fn device_int_attribute(
    device_ordinal: i32,
    attribute: DeviceAttribute,
) !i64 {
    var result = try ffi.call_global(
        std.heap.c_allocator,
        "runtime.GetDeviceAttr",
        &.{
            Value.int(@backingInt(dlpack.DeviceType.cuda)),
            Value.int(device_ordinal),
            Value.int(@backingInt(attribute)),
        },
    );
    defer result.deinit();
    return try result.borrow().to_int();
}

fn device_string_attribute(
    allocator: std.mem.Allocator,
    device_ordinal: i32,
    attribute: DeviceAttribute,
) ![]u8 {
    var result = try ffi.call_global(
        allocator,
        "runtime.GetDeviceAttr",
        &.{
            Value.int(@backingInt(dlpack.DeviceType.cuda)),
            Value.int(device_ordinal),
            Value.int(@backingInt(attribute)),
        },
    );
    defer result.deinit();
    return try result.borrow().dupe_string(allocator);
}

fn cuda_target_description(
    allocator: std.mem.Allocator,
    properties: CudaProperties,
    gpu_arch: []const u8,
) ![:0]u8 {
    if (gpu_arch.len == 0) return error.InvalidGpuArchitecture;

    return try std.fmt.allocPrintSentinel(
        allocator,
        "cuda -arch={s} -max_shared_memory_per_block={d} " ++
            "-max_threads_per_block={d} -thread_warp_size={d} " ++
            "-registers_per_block={d} -l2_cache_size_bytes={d}",
        .{
            gpu_arch,
            properties.max_shared_memory_per_block,
            properties.max_threads_per_block,
            properties.thread_warp_size,
            properties.registers_per_block,
            properties.l2_cache_size_bytes,
        },
        0,
    );
}

fn cuda_architecture(
    allocator: std.mem.Allocator,
    compute_version: []const u8,
) ![]u8 {
    var architecture: [16]u8 = undefined;
    var architecture_len: usize = 0;
    for (compute_version) |byte| switch (byte) {
        '0'...'9' => {
            if (architecture_len == architecture.len) return error.InvalidComputeCapability;
            architecture[architecture_len] = byte;
            architecture_len += 1;
        },
        '.' => {},
        else => return error.InvalidComputeCapability,
    };
    if (architecture_len == 0) return error.InvalidComputeCapability;
    return try allocator.print("sm_{s}", .{architecture[0..architecture_len]});
}

test cuda_target_description {
    const description = try cuda_target_description(
        std.testing.allocator,
        .{
            .compute_version = "8.9",
            .max_threads_per_block = 1024,
            .thread_warp_size = 32,
            .max_shared_memory_per_block = 49152,
            .registers_per_block = 65536,
            .l2_cache_size_bytes = 75497472,
        },
        "sm_89",
    );
    defer std.testing.allocator.free(description);

    try std.testing.expectEqualStrings(
        "cuda -arch=sm_89 -max_shared_memory_per_block=49152 " ++
            "-max_threads_per_block=1024 -thread_warp_size=32 " ++
            "-registers_per_block=65536 -l2_cache_size_bytes=75497472",
        description,
    );
}

test cuda_architecture {
    const architecture = try cuda_architecture(std.testing.allocator, "8.9");
    defer std.testing.allocator.free(architecture);
    try std.testing.expectEqualStrings("sm_89", architecture);
    try std.testing.expectError(
        error.InvalidComputeCapability,
        cuda_architecture(std.testing.allocator, ""),
    );
    try std.testing.expectError(
        error.InvalidComputeCapability,
        cuda_architecture(std.testing.allocator, "8.x"),
    );
}

/// TIR transform passes as a tagged union. Compile-time checked names
///  prevent string typos. Passes with arguments carry their args inline.
pub const TirPass = union(enum) {
    // No-arg passes
    lower_cross_thread_reduction,
    lower_init_block,
    plan_and_update_buffer_allocation,
    convert_blocks_to_opaque,
    lift_thread_binding,
    manifest_shared_memory_local_stage,
    lower_auto_copy,
    unify_thread_binding,
    lower_match_buffer,
    inject_permuted_layout,
    annotate_irregular_loop,
    inject_software_pipeline,
    transform_mma_buffer_layout,
    lower_opaque_block,
    flatten_buffer,
    bf16_compute_legalize,
    loop_partition,
    inject_virtual_thread,
    inject_double_buffer,
    storage_rewrite,
    hoist_if_then_else,
    simplify,
    remove_no_op,
    rewrite_unsafe_select,
    verify_vtcm_limit,
    lower_vtcm_alloc,
    verify_memory,
    annotate_entry_func,
    infer_fragment,
    lower_thread_allreduce,
    annotate_device_regions,
    split_host_device,
    merge_shared_memory_allocations,
    make_packed_api,
    fp8_storage_legalize,
    bf16_storage_legalize,
    lower_device_kernel_launch,
    lower_tvm_builtin,
    lower_custom_datatypes,
    lower_intrin,
    lower_device_storage_access_info,
    combine_context_call,
    lower_warp_memory,
    unroll_loop,
    renormalize_split_pattern,

    // Passes with arguments
    filter: struct { predicate: Value },
    bind_target: struct { target: Target },
    thread_sync: struct { scope: [:0]const u8 },
    compact_buffer_alloc: struct { is_strict: bool },
    narrow_data_type: struct { target_bits: i64 },
    vectorize_loop: struct { enable: bool },
    common_subexpr_elim: struct { enable_cse: bool, enable_equiv: bool },
    fp8_compute_legalize: struct { promote_dtype: [:0]const u8 },

    /// Returns the TVM global function name for this pass.
    ///
    /// Most names follow `"tir.transform." ++ PascalCase(@tagName)`. Variants
    ///  where TVM's name diverges from that convention have explicit overrides.
    pub fn name(self: TirPass) []const u8 {
        return switch (self) {
            // Overrides where TVM name diverges from PascalCase(@tagName)
            .filter => "tir.transform.Filter",
            .plan_and_update_buffer_allocation => "tir.transform.PlanAndUpdateBufferAllocationLocation",
            .compact_buffer_alloc => "tir.transform.CompactBufferAllocation",
            .common_subexpr_elim => "tir.transform.CommonSubexprElimTIR",
            .make_packed_api => "tir.transform.MakePackedAPI",
            .lower_tvm_builtin => "tir.transform.LowerTVMBuiltin",
            .bf16_compute_legalize => "tir.transform.BF16ComputeLegalize",
            .fp8_compute_legalize => "tir.transform.FP8ComputeLegalize",
            .verify_vtcm_limit => "tir.transform.VerifyVTCMLimit",
            .fp8_storage_legalize => "tir.transform.FP8StorageLegalize",
            .bf16_storage_legalize => "tir.transform.BF16StorageLegalize",
            inline else => |_, tag| comptime tag_to_pass_name(@tagName(tag)),
        };
    }

    /// Create the TVM pass object by calling the global function with args.
    pub fn create(self: TirPass, allocator: std.mem.Allocator) TvmError!ffi.OwnedValue {
        return switch (self) {
            .filter => |args| try ffi.call_global(allocator, self.name(), &.{args.predicate}),
            .bind_target => |args| try ffi.call_global(allocator, self.name(), &.{args.target.as_value()}),
            .thread_sync => |args| try ffi.call_global(allocator, self.name(), &.{Value.str(args.scope)}),
            .compact_buffer_alloc => |args| try ffi.call_global(allocator, self.name(), &.{Value.boolean(args.is_strict)}),
            .narrow_data_type => |args| try ffi.call_global(allocator, self.name(), &.{Value.int(args.target_bits)}),
            .vectorize_loop => |args| try ffi.call_global(allocator, self.name(), &.{Value.boolean(args.enable)}),
            .common_subexpr_elim => |args| try ffi.call_global(allocator, self.name(), &.{
                Value.boolean(args.enable_cse),
                Value.boolean(args.enable_equiv),
            }),
            .fp8_compute_legalize => |args| try ffi.call_global(
                allocator,
                self.name(),
                &.{Value.str(args.promote_dtype)},
            ),
            .verify_vtcm_limit => try ffi.call_global(allocator, self.name(), &.{Value.none()}),
            else => try ffi.call_global(allocator, self.name(), &.{}),
        };
    }

    /// Comptime: "tir.transform." ++ snake_to_pascal(tag_name).
    fn tag_to_pass_name(comptime tag_name: [:0]const u8) [:0]const u8 {
        @setEvalBranchQuota(5000);
        const result = comptime blk: {
            const prefix = "tir.transform.";
            var underscores: usize = 0;
            for (tag_name) |ch| {
                if (ch == '_') underscores += 1;
            }
            const total_len = prefix.len + tag_name.len - underscores;
            var buf: [total_len:0]u8 = undefined;
            for (prefix, 0..) |ch, i| buf[i] = ch;
            var ri: usize = prefix.len;
            var cap_next = true;
            for (tag_name) |ch| {
                if (ch == '_') {
                    cap_next = true;
                } else {
                    buf[ri] = if (cap_next) (ch - 32) else ch;
                    ri += 1;
                    cap_next = false;
                }
            }
            break :blk buf;
        };
        return &result;
    }
};

// Function attribute access

/// Get the attribute Map from an IR function (PrimFunc, etc.).
///
/// Calls `ir.BaseFunc_Attrs` then `ir.DictAttrsGetDict` and wraps the
///  result as a Map. Returns null if the function has no attributes.
pub fn get_func_attrs(allocator: std.mem.Allocator, func: Value) TvmError!?container.Map {
    var dict_attrs = try ffi.call_global(allocator, "ir.BaseFunc_Attrs", &.{func});
    defer dict_attrs.deinit();
    if (dict_attrs.borrow().is_none()) return null;

    var map_value = try ffi.call_global(
        allocator,
        "ir.DictAttrsGetDict",
        &.{dict_attrs.borrow()},
    );
    defer map_value.deinit();
    return try container.Map.retain(map_value.borrow());
}

/// Register a tensor intrinsic by name.
pub fn register_tensor_intrin(
    /// Allocator used by the registry call.
    allocator: std.mem.Allocator,
    /// Null-terminated registry name.
    name: [:0]const u8,
    /// TensorIntrin object to register. The registry retains it.
    intrinsic: Value,
) TvmError!void {
    try ffi.call_global_void(allocator, "tir.TensorIntrinRegister", &.{
        Value.str(name), intrinsic, Value.boolean(true),
    });
}

/// Load and register each name and tensor intrinsic in a serialized bundle.
pub fn register_tensor_intrin_bundle(
    /// Allocator used to deserialize and register the bundle.
    allocator: std.mem.Allocator,
    /// Null-terminated JSON array of alternating names and intrinsics.
    json: [:0]const u8,
) !usize {
    var root = try ffi.load_json(allocator, json);
    defer root.deinit();
    var entries = try container.Array.retain(root.borrow());
    defer entries.deinit();

    const entry_count = try entries.len(allocator);
    if (entry_count == 0 or entry_count % 2 != 0)
        return error.InvalidTensorIntrinsicBundle;

    var index: usize = 0;
    while (index < entry_count) : (index += 2) {
        var name_value = try entries.get(allocator, index);
        defer name_value.deinit();
        const name = try name_value.borrow().dupe_string(allocator);
        defer allocator.free(name);
        const name_z = try allocator.dupeSentinel(u8, name, 0);
        defer allocator.free(name_z);

        var intrinsic = try entries.get(allocator, index + 1);
        defer intrinsic.deinit();
        try register_tensor_intrin(allocator, name_z, intrinsic.borrow());
    }
    return entry_count / 2;
}

/// Return whether a tensor intrinsic is registered under `name`.
pub fn tensor_intrin_registered(
    allocator: std.mem.Allocator,
    name: [:0]const u8,
) !bool {
    var intrinsic = try ffi.call_global(
        allocator,
        "tir.TensorIntrinGet",
        &.{ Value.str(name), Value.boolean(true) },
    );
    defer intrinsic.deinit();
    return !intrinsic.borrow().is_none();
}
