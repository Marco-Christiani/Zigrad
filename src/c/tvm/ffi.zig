//! TVM FFI bindings.
//!
//! Provides packed values, object references, and call helpers.
//!
//! Runtime-library lifecycle lives in the top-level TVM integration. Object
//!  wrappers in this directory build on this raw call layer.
const std = @import("std");
const c = @import("c.zig");

const log = std.log.scoped(.@"zg/tvm_api");

/// TVM `AnyView` representation.
pub const Value = struct {
    raw: c.TVMFFIAny,

    /// Construct TVM's absence value.
    pub fn none() Value {
        var v = std.mem.zeroes(c.TVMFFIAny);
        v.type_index = c.kTVMFFINone;
        return .{ .raw = v };
    }

    /// Construct an immediate integer value.
    pub fn int(val: i64) Value {
        var v = std.mem.zeroes(c.TVMFFIAny);
        v.type_index = c.kTVMFFIInt;
        v.unnamed_1.v_int64 = val;
        return .{ .raw = v };
    }

    /// Construct an immediate Boolean value.
    pub fn boolean(val: bool) Value {
        var v = std.mem.zeroes(c.TVMFFIAny);
        v.type_index = c.kTVMFFIBool;
        v.unnamed_1.v_int64 = if (val) 1 else 0;
        return .{ .raw = v };
    }

    /// Construct an immediate floating-point value.
    pub fn float(val: f64) Value {
        var v = std.mem.zeroes(c.TVMFFIAny);
        v.type_index = c.kTVMFFIFloat;
        v.unnamed_1.v_float64 = val;
        return .{ .raw = v };
    }

    /// Borrow a sentinel-terminated string for one packed call.
    pub fn str(s: [*:0]const u8) Value {
        var v = std.mem.zeroes(c.TVMFFIAny);
        v.type_index = c.kTVMFFIRawStr;
        v.unnamed_1.v_c_str = s;
        return .{ .raw = v };
    }

    /// Borrow an object handle with its runtime type index.
    pub fn from_object(handle: c.TVMFFIObjectHandle, type_index: i32) Value {
        var v = std.mem.zeroes(c.TVMFFIAny);
        v.type_index = type_index;
        v.unnamed_1.v_obj = @ptrCast(@alignCast(handle));
        return .{ .raw = v };
    }

    /// Return the object handle, or null for an immediate value.
    pub fn as_object(self: Value) c.TVMFFIObjectHandle {
        if (self.raw.type_index < c.kTVMFFIStaticObjectBegin) return null;
        const obj = self.raw.unnamed_1.v_obj;
        return @ptrCast(@alignCast(obj));
    }

    /// Return the immediate integer, or null for another value type.
    pub fn as_int(self: Value) ?i64 {
        if (self.raw.type_index != c.kTVMFFIInt) return null;
        return self.raw.unnamed_1.v_int64;
    }

    /// Return the immediate float, or null for another value type.
    pub fn as_float(self: Value) ?f64 {
        if (self.raw.type_index != c.kTVMFFIFloat) return null;
        return self.raw.unnamed_1.v_float64;
    }

    /// Return an immediate integer.
    pub fn require_int(self: Value) TvmError!i64 {
        try self.require_type_index(c.kTVMFFIInt);
        return self.raw.unnamed_1.v_int64;
    }

    /// Return an immediate float.
    pub fn require_float(self: Value) TvmError!f64 {
        try self.require_type_index(c.kTVMFFIFloat);
        return self.raw.unnamed_1.v_float64;
    }

    /// Copy a TVM string value. The caller frees the returned bytes.
    pub fn dupe_string(self: Value, allocator: std.mem.Allocator) ![]u8 {
        return try self.dupe_byte_payload(
            allocator,
            c.kTVMFFISmallStr,
            c.kTVMFFIStr,
        );
    }

    /// Copy a TVM byte string. The caller frees the returned bytes.
    pub fn dupe_bytes(
        self: Value,
        /// Allocator used for the returned bytes.
        allocator: std.mem.Allocator,
    ) ![]u8 {
        return try self.dupe_byte_payload(
            allocator,
            c.kTVMFFISmallBytes,
            c.kTVMFFIBytes,
        );
    }

    fn dupe_byte_payload(
        self: Value,
        allocator: std.mem.Allocator,
        small_type: i32,
        object_type: i32,
    ) ![]u8 {
        if (self.is_none()) return error.TvmValueAbsent;
        if (self.raw.type_index == small_type) {
            const n: usize = @intCast(self.raw.unnamed_0.small_str_len);
            return try allocator.dupe(u8, self.raw.unnamed_1.v_bytes[0..n]);
        }
        if (self.raw.type_index == object_type) {
            const obj: c.TVMFFIObjectHandle = @ptrCast(self.raw.unnamed_1.v_obj);
            // Matches TVMFFIBytesGetByteArrayPtr in tvm/ffi/c_api.h.
            const hdr_size = @sizeOf(c.TVMFFIObject);
            const ba_ptr: *const c.TVMFFIByteArray = @ptrCast(@alignCast(@as([*]const u8, @ptrCast(obj)) + hdr_size));
            if (ba_ptr.data == null or ba_ptr.size == 0) return try allocator.dupe(u8, "");
            return try allocator.dupe(u8, ba_ptr.data[0..ba_ptr.size]);
        }
        return error.TvmTypeMismatch;
    }

    /// Extract an integer, handling both raw kTVMFFIInt and IntImm objects.
    ///
    /// TVM sometimes returns integers as raw `kTVMFFIInt` values and sometimes
    /// as `IntImm` objects with a `value` field. This method tries both.
    pub fn to_int(self: Value) TvmError!i64 {
        if (self.as_int()) |v| return v;
        var field = try get_field(self, "value");
        defer field.deinit();
        return try field.borrow().require_int();
    }

    /// Extract a float, accepting either an immediate or a `FloatImm` object.
    pub fn to_float(self: Value) TvmError!f64 {
        if (self.as_float()) |value| return value;
        var field = try get_field(self, "value");
        defer field.deinit();
        return try field.borrow().require_float();
    }

    /// Return whether this is TVM's absence value.
    pub fn is_none(self: Value) bool {
        return self.raw.type_index == c.kTVMFFINone;
    }

    fn require_object(self: Value) TvmError!void {
        if (self.is_none()) return error.TvmValueAbsent;
        if (self.as_object() == null) return error.TvmTypeMismatch;
    }

    /// Require the exact FFI type index.
    pub fn require_type_index(self: Value, type_index: i32) TvmError!void {
        if (self.is_none()) {
            log.err("TVM value absent: expected type index {d}", .{type_index});
            return error.TvmValueAbsent;
        }
        if (self.raw.type_index != type_index) {
            log_type_mismatch_index(type_index, self.raw.type_index);
            return error.TvmTypeMismatch;
        }
    }

    /// Require `type_key` or one of its derived object types.
    pub fn require_instance(self: Value, type_key: []const u8) TvmError!void {
        if (self.is_none()) {
            log.err("TVM value absent: expected {s}", .{type_key});
            return error.TvmValueAbsent;
        }
        if (self.as_object() == null) {
            log_type_mismatch_key(type_key, self.raw.type_index);
            return error.TvmTypeMismatch;
        }
        const info = c.TVMFFIGetTypeInfo(self.raw.type_index) orelse {
            log.err("TVM type info missing for index {d}", .{self.raw.type_index});
            return error.TvmTypeInfoMissing;
        };
        if (type_key_eql(info, type_key)) return;

        const depth = std.math.cast(usize, info.type_depth) orelse
            return error.TvmTypeInfoMissing;
        if (depth > 0) {
            const ancestors = info.type_ancestors orelse
                return error.TvmTypeInfoMissing;
            for (ancestors[0..depth]) |ancestor| {
                if (type_key_eql(ancestor, type_key)) return;
            }
        }
        log_type_mismatch_key(type_key, self.raw.type_index);
        return error.TvmTypeMismatch;
    }
};

/// TVM `Any` representation.
pub const OwnedValue = struct {
    value: Value,

    fn from_raw(raw: c.TVMFFIAny) OwnedValue {
        return .{ .value = .{ .raw = raw } };
    }

    /// Copy a Zig string into a TVM string.
    pub fn from_string(string: []const u8) TvmError!OwnedValue {
        var bytes: c.TVMFFIByteArray = .{ .data = string.ptr, .size = string.len };
        var out: c.TVMFFIAny = std.mem.zeroes(c.TVMFFIAny);
        const status = c.TVMFFIStringFromByteArray(&bytes, &out);
        if (status != 0)
            return call_error("TVMFFIStringFromByteArray", status);
        return .from_raw(out);
    }

    /// Retain `value`.
    pub fn retain(value: Value) TvmError!OwnedValue {
        var out = Value.none().raw;
        const status = c.TVMFFIAnyViewToOwnedAny(&value.raw, &out);
        if (status != 0)
            return call_error("TVMFFIAnyViewToOwnedAny", status);
        return .from_raw(out);
    }

    pub fn borrow(self: *const OwnedValue) Value {
        return self.value;
    }

    /// Transfer the packed value to the caller.
    pub fn take(self: *OwnedValue) Value {
        const value = self.value;
        self.* = undefined;
        return value;
    }

    pub fn deinit(self: *OwnedValue) void {
        if (self.value.as_object()) |object| object_dec_ref(object);
        self.* = undefined;
    }
};

/// Reference to any TVM object.
pub const Object = struct {
    value: OwnedValue,

    /// Take an object from `value`.
    pub fn take(value: *OwnedValue) TvmError!Object {
        try value.borrow().require_object();
        const owned = value.*;
        value.* = undefined;
        return .{ .value = owned };
    }

    /// Transfer the object as a packed value.
    pub fn take_value(self: *Object) OwnedValue {
        const value = self.value;
        self.* = undefined;
        return value;
    }

    /// Copy an object reference.
    pub fn retain(value: Value) TvmError!Object {
        try value.require_object();
        return .{ .value = try OwnedValue.retain(value) };
    }

    pub fn as_value(self: *const Object) Value {
        return self.value.borrow();
    }

    /// Return the raw object pointer for the TVM C API.
    pub fn ptr(self: *const Object) c.TVMFFIObjectHandle {
        return self.as_value().as_object();
    }

    pub fn deinit(self: *Object) void {
        self.value.deinit();
        self.* = undefined;
    }
};

/// TVM packed function.
pub const PackedFunction = struct {
    value: OwnedValue,

    /// Create a packed function backed by a Zig callback.
    ///
    /// On success, `destructor` releases `context` with the function. The
    ///  caller retains `context` when creation fails.
    pub fn create(
        context: ?*anyopaque,
        callback: Callback,
        destructor: ?CallbackDestructor,
    ) TvmError!PackedFunction {
        const state = try std.heap.c_allocator.create(CallbackState);
        errdefer std.heap.c_allocator.destroy(state);
        state.* = .{
            .context = context,
            .callback = callback,
            .destructor = destructor,
        };
        return .{
            .value = try create_packed_func(
                state,
                CallbackState.invoke,
                CallbackState.destroy,
            ),
        };
    }

    /// Copy a packed-function reference.
    pub fn retain(value: Value) TvmError!PackedFunction {
        try value.require_type_index(c.kTVMFFIFunction);
        return .{ .value = try OwnedValue.retain(value) };
    }

    /// Take a packed function from `value`.
    pub fn take(value: *OwnedValue) TvmError!PackedFunction {
        try value.borrow().require_type_index(c.kTVMFFIFunction);
        const owned = value.*;
        value.* = undefined;
        return .{ .value = owned };
    }

    pub fn as_value(self: *const PackedFunction) Value {
        return self.value.borrow();
    }

    /// Invoke the packed function.
    pub fn call(
        self: *const PackedFunction,
        allocator: std.mem.Allocator,
        args: []const Value,
    ) TvmError!OwnedValue {
        const handle = self.as_value().as_object() orelse
            return error.TvmTypeMismatch;
        return try call_handle(allocator, handle, args);
    }

    pub fn deinit(self: *PackedFunction) void {
        self.value.deinit();
        self.* = undefined;
    }
};

const CallbackArgsState = struct {
    values: [*c]const c.TVMFFIAny,
    len: usize,
};

/// Arguments supplied to a Zig packed-function callback.
pub const CallbackArgs = opaque {
    fn state(self: *const CallbackArgs) *const CallbackArgsState {
        return @ptrCast(@alignCast(self));
    }

    /// Return the callback argument count.
    pub fn len(self: *const CallbackArgs) usize {
        return self.state().len;
    }

    /// Return one borrowed callback argument.
    pub fn get(self: *const CallbackArgs, index: usize) Value {
        const args = self.state();
        std.debug.assert(index < args.len);
        return .{ .raw = args.values[index] };
    }
};

/// Value returned by a Zig packed-function callback.
pub const CallbackOutput = union(enum) {
    none,
    integer: i64,
    boolean: bool,
    float: f64,
    /// Object or copied string transferred to TVM.
    owned: OwnedValue,

    fn write(self: *CallbackOutput, output: *c.TVMFFIAny) void {
        output.* = switch (self.*) {
            .none => Value.none().raw,
            .integer => |value| Value.int(value).raw,
            .boolean => |value| Value.boolean(value).raw,
            .float => |value| Value.float(value).raw,
            .owned => |*value| value.take().raw,
        };
        self.* = undefined;
    }
};

/// Zig callback invoked through TVM's packed-function ABI.
pub const Callback = *const fn (
    context: ?*anyopaque,
    args: *const CallbackArgs,
) anyerror!CallbackOutput;

/// Releases callback context after TVM releases the packed function.
pub const CallbackDestructor = *const fn (context: ?*anyopaque) void;

const CallbackState = struct {
    context: ?*anyopaque,
    callback: Callback,
    destructor: ?CallbackDestructor,

    fn invoke(
        self_ptr: ?*anyopaque,
        raw_args: [*c]const c.TVMFFIAny,
        num_args: i32,
        raw_result: [*c]c.TVMFFIAny,
    ) callconv(.c) c_int {
        const self: *CallbackState = @ptrCast(@alignCast(self_ptr orelse
            return fail_callback_contract("callback state is absent")));
        if (num_args < 0)
            return fail_callback_contract("callback argument count is negative");
        if (num_args > 0 and raw_args == null)
            return fail_callback_contract("callback arguments are absent");
        if (raw_result == null)
            return fail_callback_contract("callback result is absent");
        assert_callback_result_empty(raw_result);

        const args_state = CallbackArgsState{
            .values = raw_args,
            .len = @intCast(num_args),
        };
        var output = self.callback(
            self.context,
            @ptrCast(&args_state),
        ) catch |err| return fail_callback(err);
        output.write(raw_result);
        return 0;
    }

    fn destroy(self_ptr: ?*anyopaque) callconv(.c) void {
        const self: *CallbackState = @ptrCast(@alignCast(self_ptr orelse return));
        if (self.destructor) |destructor| destructor(self.context);
        std.heap.c_allocator.destroy(self);
    }
};

fn object_dec_ref(object: c.TVMFFIObjectHandle) void {
    const status = c.TVMFFIObjectDecRef(object);
    if (status != 0) {
        const failure = call_error("TVMFFIObjectDecRef", status);
        log.err("failed to release TVM object: {s}", .{@errorName(failure)});
    }
}

fn type_key_eql(info: *const c.TVMFFITypeInfo, expected: []const u8) bool {
    const key = byte_array_slice(info.type_key);
    return std.mem.eql(u8, key, expected);
}

fn byte_array_slice(bytes: c.TVMFFIByteArray) []const u8 {
    if (bytes.data == null or bytes.size == 0) return "";
    return bytes.data[0..bytes.size];
}

fn log_type_mismatch_index(expected: i32, actual: i32) void {
    log.err("TVM type mismatch: expected index {d}, got {d}", .{ expected, actual });
}

fn log_type_mismatch_key(expected: []const u8, actual: i32) void {
    const info = c.TVMFFIGetTypeInfo(actual) orelse {
        log.err("TVM type mismatch: expected {s}, got index {d}", .{ expected, actual });
        return;
    };
    log.err("TVM type mismatch: expected {s}, got {s}", .{
        expected,
        byte_array_slice(info.type_key),
    });
}

pub const TvmError = error{
    /// Host byte count disagrees with the tensor shape and dtype.
    InvalidTensorDataSize,
    /// Tensor rank, dimensions, or element representation is invalid.
    InvalidTensorLayout,
    /// An integer cannot be represented by the target ABI type.
    IntegerOutOfRange,
    /// A packed call or reflected field accessor failed.
    TvmCallFailed,
    /// A packed call reports an error held by its embedding frontend.
    TvmFrontendCallFailed,
    /// A packed call returned an invalid status or omitted its raised error.
    TvmInvalidErrorState,
    /// A runtime module or shared library could not be loaded.
    TvmLoadFailed,
    /// The packed-function registry contains no entry for the requested name.
    TvmFunctionNotFound,
    /// A required packed value is `None`.
    TvmValueAbsent,
    /// A packed value does not satisfy the function contract.
    TvmTypeMismatch,
    /// Runtime type information is unavailable.
    TvmTypeInfoMissing,
    /// A reflected field does not exist.
    TvmFieldNotFound,
    /// A reflected field cannot be read.
    TvmFieldUnreadable,
    /// Memory allocation failed.
    OutOfMemory,
};

fn log_raised_error(operation: []const u8) bool {
    var err_obj: c.TVMFFIObjectHandle = null;
    c.TVMFFIErrorMoveFromRaised(&err_obj);
    if (err_obj == null) {
        log.err("{s} failed without a TVM error", .{operation});
        return false;
    }
    defer object_dec_ref(err_obj);

    // Matches TVMFFIErrorGetCellPtr in tvm/ffi/c_api.h.
    const hdr_size = @sizeOf(c.TVMFFIObject);
    const cell_ptr: *const c.TVMFFIErrorCell = @ptrCast(@alignCast(@as([*]const u8, @ptrCast(err_obj)) + hdr_size));
    const kind = byte_array_slice(cell_ptr.kind);
    const message = byte_array_slice(cell_ptr.message);
    const backtrace = byte_array_slice(cell_ptr.backtrace);
    if (backtrace.len == 0) {
        log.err("{s} failed [{s}]: {s}", .{ operation, kind, message });
    } else {
        log.err("{s} failed [{s}]: {s}\n{s}", .{ operation, kind, message, backtrace });
    }
    return true;
}

/// Classify a failed TVM C call and consume any raised error.
pub fn call_error(operation: []const u8, status: c_int) TvmError {
    std.debug.assert(status != 0);
    if (status == -2) {
        log.err("{s} failed in the embedding frontend", .{operation});
        return error.TvmFrontendCallFailed;
    }
    if (status != -1) {
        log.err("{s} returned invalid TVM status {d}", .{ operation, status });
        return error.TvmInvalidErrorState;
    }
    if (!log_raised_error(operation)) return error.TvmInvalidErrorState;
    return error.TvmCallFailed;
}

/// Set TVM's raised error from a Zig callback error.
fn fail_callback(err: anyerror) c_int {
    var message_buffer: [256]u8 = undefined;
    const message = std.fmt.bufPrintZ(&message_buffer, "Zig callback failed: {s}", .{
        @errorName(err),
    }) catch "Zig callback failed";
    c.TVMFFIErrorSetRaisedFromCStr("RuntimeError", message.ptr);
    return -1;
}

/// Set TVM's raised error from a callback contract violation.
fn fail_callback_contract(message: [*:0]const u8) c_int {
    c.TVMFFIErrorSetRaisedFromCStr("ValueError", message);
    return -1;
}

/// Assert the result-slot precondition for a TVM packed callback.
fn assert_callback_result_empty(result: [*c]const c.TVMFFIAny) void {
    std.debug.assert(result != null);
    std.debug.assert(result.*.type_index == c.kTVMFFINone);
}

// Call helpers.

/// Look up a TVM global function by name.
pub fn get_global(name: []const u8) TvmError!PackedFunction {
    var name_arr: c.TVMFFIByteArray = .{ .data = name.ptr, .size = name.len };
    var out: c.TVMFFIObjectHandle = null;
    const status = c.TVMFFIFunctionGetGlobal(&name_arr, &out);
    if (status != 0)
        return call_error("TVMFFIFunctionGetGlobal", status);
    if (out == null) {
        log.err("TVM global function not found: {s}", .{name});
        return error.TvmFunctionNotFound;
    }
    var value = OwnedValue{
        .value = Value.from_object(out, c.kTVMFFIFunction),
    };
    errdefer value.deinit();
    return try .take(&value);
}

/// Call a TVM function handle with args, writing the result to `out`.
///
/// `out` is initialized to `None` because `SafeCallImpl` inspects its type
///  index before invoking the function.
fn call(func: c.TVMFFIObjectHandle, args: []const c.TVMFFIAny, out: *c.TVMFFIAny) TvmError!void {
    out.* = Value.none().raw;
    const arg_count = std.math.cast(i32, args.len) orelse
        return error.IntegerOutOfRange;
    const arg_ptr = if (args.len == 0) null else @constCast(args.ptr);
    const status = c.TVMFFIFunctionCall(func, arg_ptr, arg_count, out);
    if (status != 0)
        return call_error("TVMFFIFunctionCall", status);
}

/// Call a TVM global function and return ownership of its result.
pub fn call_global(allocator: std.mem.Allocator, func_name: []const u8, args: []const Value) TvmError!OwnedValue {
    var function = try get_global(func_name);
    defer function.deinit();
    return try function.call(allocator, args);
}

/// Call a TVM global function and transfer its result into `Result`.
pub fn call_global_take(
    comptime Result: type,
    /// Allocator used by the packed call.
    allocator: std.mem.Allocator,
    /// Registered TVM global function name.
    func_name: []const u8,
    /// Packed-call arguments.
    args: []const Value,
) TvmError!Result {
    var result = try call_global(allocator, func_name, args);
    errdefer result.deinit();
    return try Result.take(&result);
}

/// Call a TVM global function whose registered return type is void.
pub fn call_global_void(
    /// Allocator used by the packed call.
    allocator: std.mem.Allocator,
    /// Registered TVM global function name.
    func_name: []const u8,
    /// Packed-call arguments.
    args: []const Value,
) TvmError!void {
    var result = try call_global(allocator, func_name, args);
    defer result.deinit();
    if (!result.borrow().is_none()) return error.TvmTypeMismatch;
}

/// Call a TVM function handle and return ownership of its result.
fn call_handle(allocator: std.mem.Allocator, func: c.TVMFFIObjectHandle, args: []const Value) TvmError!OwnedValue {
    return try call_with_values(allocator, func, args);
}

/// Shared implementation: convert Value args to raw TVMFFIAny and call.
/// Uses a stack buffer for <=16 args, heap for larger lists.
fn call_with_values(allocator: std.mem.Allocator, func: c.TVMFFIObjectHandle, args: []const Value) TvmError!OwnedValue {
    if (args.len <= 16) {
        var raw_args: [16]c.TVMFFIAny = undefined;
        for (args, 0..) |a, i| raw_args[i] = a.raw;
        var out: c.TVMFFIAny = undefined;
        try call(func, raw_args[0..args.len], &out);
        return .from_raw(out);
    } else {
        const raw_args = try allocator.alloc(c.TVMFFIAny, args.len);
        defer allocator.free(raw_args);
        for (args, 0..) |a, i| raw_args[i] = a.raw;
        var out: c.TVMFFIAny = undefined;
        try call(func, raw_args, &out);
        return .from_raw(out);
    }
}

/// Register a TVM global function by name.
pub fn set_global(name: []const u8, function: *const PackedFunction, override: bool) TvmError!void {
    const func_handle = function.as_value().as_object() orelse
        return error.TvmTypeMismatch;
    var name_arr: c.TVMFFIByteArray = .{ .data = name.ptr, .size = name.len };
    const status = c.TVMFFIFunctionSetGlobal(&name_arr, func_handle, if (override) 1 else 0);
    if (status != 0)
        return call_error("TVMFFIFunctionSetGlobal", status);
}

const PackedFuncCallback = *const fn (
    self_ptr: ?*anyopaque,
    args: [*c]const c.TVMFFIAny,
    num_args: i32,
    result: [*c]c.TVMFFIAny,
) callconv(.c) c_int;

const PackedFuncDestructor = *const fn (self_ptr: ?*anyopaque) callconv(.c) void;

/// Create a TVM packed function from a Zig callback with optional userdata.
fn create_packed_func(
    self_ptr: ?*anyopaque,
    callback: PackedFuncCallback,
    destructor: ?PackedFuncDestructor,
) TvmError!OwnedValue {
    var func_handle: c.TVMFFIObjectHandle = null;
    const status = c.TVMFFIFunctionCreate(
        self_ptr,
        callback,
        destructor,
        &func_handle,
    );
    if (status != 0) return call_error("TVMFFIFunctionCreate", status);
    return .{ .value = Value.from_object(func_handle, c.kTVMFFIFunction) };
}

// Object reflection

/// Read a reflected field from a TVM object.
pub fn get_field(obj: Value, field_name: []const u8) TvmError!OwnedValue {
    try obj.require_object();
    const obj_ptr = obj.raw.unnamed_1.v_obj orelse {
        log.warn("get_field: null object", .{});
        return error.TvmTypeMismatch;
    };

    const type_info: ?*const c.TVMFFITypeInfo = c.TVMFFIGetTypeInfo(obj.raw.type_index);
    if (type_info == null) {
        log.warn("get_field: TVMFFIGetTypeInfo({d}) returned null", .{obj.raw.type_index});
        return error.TvmTypeInfoMissing;
    }

    const field = (try find_field(type_info.?, field_name)) orelse {
        log.debug("get_field: '{s}' not found (type_index={d})", .{
            field_name, obj.raw.type_index,
        });
        return error.TvmFieldNotFound;
    };

    const getter = field.getter orelse {
        log.warn("get_field: '{s}' has no getter", .{field_name});
        return error.TvmFieldUnreadable;
    };

    // Compute field address: obj_ptr + offset
    const offset = std.math.cast(usize, field.offset) orelse {
        log.warn("get_field: invalid offset for '{s}'", .{field_name});
        return error.TvmTypeInfoMissing;
    };
    // Matches tvm::ffi::reflection::FieldGetter in reflection/accessor.h.
    const field_ptr: *anyopaque = @ptrCast(@as([*]u8, @ptrCast(obj_ptr)) + offset);

    var result: c.TVMFFIAny = Value.none().raw;
    const status = getter(field_ptr, &result);
    if (status != 0)
        return call_error("TVMFFIFieldGetter", status);
    return .from_raw(result);
}

fn find_field(type_info: *const c.TVMFFITypeInfo, field_name: []const u8) TvmError!?*const c.TVMFFIFieldInfo {
    const depth = std.math.cast(usize, type_info.type_depth) orelse
        return error.TvmTypeInfoMissing;
    if (depth > 1) {
        const ancestors = type_info.type_ancestors orelse
            return error.TvmTypeInfoMissing;
        for (ancestors[1..depth]) |ancestor| {
            if (try find_direct_field(ancestor, field_name)) |field| return field;
        }
    }
    return try find_direct_field(type_info, field_name);
}

fn find_direct_field(type_info: *const c.TVMFFITypeInfo, field_name: []const u8) TvmError!?*const c.TVMFFIFieldInfo {
    const count = std.math.cast(usize, type_info.num_fields) orelse
        return error.TvmTypeInfoMissing;
    if (count == 0) return null;
    const fields = type_info.fields orelse return error.TvmTypeInfoMissing;
    for (fields[0..count]) |*field| {
        if (std.mem.eql(u8, byte_array_slice(field.name), field_name)) return field;
    }
    return null;
}

/// List all TVM global function names, sorted alphabetically.
///
/// Uses `ffi.FunctionListGlobalNamesFunctor` to enumerate all registered
/// TVM functions. The caller frees the slice and each name.
pub fn list_global_names(allocator: std.mem.Allocator) ![][]const u8 {
    var functor = try call_global(allocator, "ffi.FunctionListGlobalNamesFunctor", &.{});
    defer functor.deinit();

    var function = try PackedFunction.take(&functor);
    defer function.deinit();

    // functor(-1) returns the count
    var count_value = try function.call(allocator, &.{Value.int(-1)});
    defer count_value.deinit();
    const count_value_int = try count_value.borrow().require_int();
    const count = std.math.cast(usize, count_value_int) orelse
        return error.IntegerOutOfRange;

    var names = try std.ArrayList([]const u8).initCapacity(allocator, count);
    errdefer {
        for (names.items) |n| allocator.free(n);
        names.deinit(allocator);
    }

    for (0..count) |i| {
        const index = std.math.cast(i64, i) orelse return error.IntegerOutOfRange;
        var name_value = try function.call(allocator, &.{Value.int(index)});
        defer name_value.deinit();
        const s = try name_value.borrow().dupe_string(allocator);
        names.append(allocator, s) catch |err| {
            allocator.free(s);
            return err;
        };
    }

    const items = try names.toOwnedSlice(allocator);
    std.mem.sort([]const u8, items, {}, struct {
        fn lessThan(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.order(u8, a, b) == .lt;
        }
    }.lessThan);
    return items;
}

/// Deserialize a TVM object from its JSON representation.
pub fn load_json(allocator: std.mem.Allocator, json: [:0]const u8) TvmError!OwnedValue {
    return try call_global(allocator, "node.LoadJSON", &.{Value.str(json)});
}
