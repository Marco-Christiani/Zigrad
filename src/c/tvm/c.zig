//! TVM C API FFI bindings.
//!
//! This module always provides TVM C types/constants at compile time (headers
//!  must be present), while function symbols are resolved at runtime.

const std = @import("std");
const dylib = @import("../dylib.zig");

const log = std.log.scoped(.@"zg/tvm_cffi");

const c = @import("c-tvm");

pub const TVMFFIAny = c.TVMFFIAny;
pub const TVMFFIObjectHandle = c.TVMFFIObjectHandle;
pub const TVMFFIByteArray = c.TVMFFIByteArray;
pub const TVMFFIErrorCell = c.TVMFFIErrorCell;
pub const TVMFFIFieldInfo = c.TVMFFIFieldInfo;
pub const TVMFFIObject = c.TVMFFIObject;
pub const TVMFFITypeInfo = c.TVMFFITypeInfo;
pub const DLManagedTensor = c.struct_DLManagedTensor;

pub const kTVMFFINone = c.kTVMFFINone;
pub const kTVMFFIInt = c.kTVMFFIInt;
pub const kTVMFFIFloat = c.kTVMFFIFloat;
pub const kTVMFFIBool = c.kTVMFFIBool;
pub const kTVMFFIRawStr = c.kTVMFFIRawStr;
pub const kTVMFFITensor = c.kTVMFFITensor;
pub const kTVMFFIOpaquePtr = c.kTVMFFIOpaquePtr;
pub const kTVMFFIModule = c.kTVMFFIModule;
pub const kTVMFFIFunction = c.kTVMFFIFunction;
pub const kTVMFFIShape = c.kTVMFFIShape;
pub const kTVMFFIArray = c.kTVMFFIArray;
pub const kTVMFFIMap = c.kTVMFFIMap;
pub const kTVMFFISmallStr = c.kTVMFFISmallStr;
pub const kTVMFFIStr = c.kTVMFFIStr;
pub const kTVMFFISmallBytes = c.kTVMFFISmallBytes;
pub const kTVMFFIBytes = c.kTVMFFIBytes;
pub const kTVMFFIDataType = c.kTVMFFIDataType;
pub const kTVMFFIDevice = c.kTVMFFIDevice;
pub const kTVMFFIStaticObjectBegin = c.kTVMFFIStaticObjectBegin;

const PackedCFunc = *const fn (?*anyopaque, [*c]const TVMFFIAny, i32, [*c]TVMFFIAny) callconv(.c) c_int;
const PackedCFuncFinalizer = *const fn (?*anyopaque) callconv(.c) void;

const FnObjectDecRef = *const fn (TVMFFIObjectHandle) callconv(.c) c_int;
const FnAnyViewToOwnedAny = *const fn (*const TVMFFIAny, *TVMFFIAny) callconv(.c) c_int;
const FnFunctionCreate = *const fn (?*anyopaque, ?PackedCFunc, ?PackedCFuncFinalizer, *TVMFFIObjectHandle) callconv(.c) c_int;
const FnFunctionCall = *const fn (TVMFFIObjectHandle, ?[*]TVMFFIAny, i32, *TVMFFIAny) callconv(.c) c_int;
const FnFunctionGetGlobal = *const fn (*TVMFFIByteArray, *TVMFFIObjectHandle) callconv(.c) c_int;
const FnFunctionSetGlobal = *const fn (*TVMFFIByteArray, TVMFFIObjectHandle, i32) callconv(.c) c_int;
const FnErrorMoveFromRaised = *const fn (*TVMFFIObjectHandle) callconv(.c) void;
const FnErrorSetRaisedFromCStr = *const fn ([*:0]const u8, [*:0]const u8) callconv(.c) void;
const FnGetTypeInfo = *const fn (i32) callconv(.c) ?*const TVMFFITypeInfo;
const FnStringFromByteArray = *const fn (*TVMFFIByteArray, *TVMFFIAny) callconv(.c) c_int;
const FnTensorFromDLPack = *const fn ([*c]DLManagedTensor, i32, i32, *TVMFFIObjectHandle) callconv(.c) c_int;

const Symbols = struct {
    object_dec_ref: FnObjectDecRef,
    any_view_to_owned_any: FnAnyViewToOwnedAny,
    function_create: FnFunctionCreate,
    function_call: FnFunctionCall,
    function_get_global: FnFunctionGetGlobal,
    function_set_global: FnFunctionSetGlobal,
    error_move_from_raised: FnErrorMoveFromRaised,
    error_set_raised_from_c_str: FnErrorSetRaisedFromCStr,
    get_type_info: FnGetTypeInfo,
    string_from_byte_array: FnStringFromByteArray,
    tensor_from_dlpack: FnTensorFromDLPack,

    fn load(library: dylib.Library) LoadError!Symbols {
        return .{
            .object_dec_ref = try load_symbol(FnObjectDecRef, library, "TVMFFIObjectDecRef"),
            .any_view_to_owned_any = try load_symbol(FnAnyViewToOwnedAny, library, "TVMFFIAnyViewToOwnedAny"),
            .function_create = try load_symbol(FnFunctionCreate, library, "TVMFFIFunctionCreate"),
            .function_call = try load_symbol(FnFunctionCall, library, "TVMFFIFunctionCall"),
            .function_get_global = try load_symbol(FnFunctionGetGlobal, library, "TVMFFIFunctionGetGlobal"),
            .function_set_global = try load_symbol(FnFunctionSetGlobal, library, "TVMFFIFunctionSetGlobal"),
            .error_move_from_raised = try load_symbol(FnErrorMoveFromRaised, library, "TVMFFIErrorMoveFromRaised"),
            .error_set_raised_from_c_str = try load_symbol(FnErrorSetRaisedFromCStr, library, "TVMFFIErrorSetRaisedFromCStr"),
            .get_type_info = try load_symbol(FnGetTypeInfo, library, "TVMFFIGetTypeInfo"),
            .string_from_byte_array = try load_symbol(FnStringFromByteArray, library, "TVMFFIStringFromByteArray"),
            .tensor_from_dlpack = try load_symbol(FnTensorFromDLPack, library, "TVMFFITensorFromDLPack"),
        };
    }
};

var symbols: ?Symbols = null;

pub const LoadError = error{
    TvmSymbolMissing,
};

/// Install the complete TVM FFI symbol table.
///
/// The caller serializes process initialization and keeps `library` open.
pub fn install_symbols(library: dylib.Library) LoadError!void {
    if (symbols != null) return;
    symbols = try Symbols.load(library);
}

fn get_symbol(comptime name: []const u8) @TypeOf(@field(@as(Symbols, undefined), name)) {
    std.debug.assert(symbols != null);
    return @field(symbols.?, name);
}

pub fn TVMFFIObjectDecRef(handle: TVMFFIObjectHandle) c_int {
    const f = get_symbol("object_dec_ref");
    return f(handle);
}

pub fn TVMFFIAnyViewToOwnedAny(value: *const TVMFFIAny, out: *TVMFFIAny) c_int {
    const f = get_symbol("any_view_to_owned_any");
    return f(value, out);
}

pub fn TVMFFIFunctionCreate(
    self_ptr: ?*anyopaque,
    callback: ?PackedCFunc,
    destructor: ?PackedCFuncFinalizer,
    out: *TVMFFIObjectHandle,
) c_int {
    const f = get_symbol("function_create");
    return f(self_ptr, callback, destructor, out);
}

pub fn TVMFFIFunctionCall(
    func: TVMFFIObjectHandle,
    args: ?[*]TVMFFIAny,
    num_args: i32,
    out: *TVMFFIAny,
) c_int {
    const f = get_symbol("function_call");
    return f(func, args, num_args, out);
}

pub fn TVMFFIFunctionGetGlobal(name: *TVMFFIByteArray, out: *TVMFFIObjectHandle) c_int {
    const f = get_symbol("function_get_global");
    return f(name, out);
}

pub fn TVMFFIFunctionSetGlobal(name: *TVMFFIByteArray, func: TVMFFIObjectHandle, override: i32) c_int {
    const f = get_symbol("function_set_global");
    return f(name, func, override);
}

pub fn TVMFFIErrorMoveFromRaised(out: *TVMFFIObjectHandle) void {
    const f = get_symbol("error_move_from_raised");
    f(out);
}

pub fn TVMFFIErrorSetRaisedFromCStr(kind: [*:0]const u8, message: [*:0]const u8) void {
    const f = get_symbol("error_set_raised_from_c_str");
    f(kind, message);
}

pub fn TVMFFIGetTypeInfo(type_index: i32) ?*const TVMFFITypeInfo {
    const f = get_symbol("get_type_info");
    return f(type_index);
}

pub fn TVMFFIStringFromByteArray(bytes: *TVMFFIByteArray, out: *TVMFFIAny) c_int {
    const f = get_symbol("string_from_byte_array");
    return f(bytes, out);
}

pub fn TVMFFITensorFromDLPack(
    managed_tensor: [*c]DLManagedTensor,
    require_alignment: i32,
    require_contiguous: i32,
    out: *TVMFFIObjectHandle,
) c_int {
    const f = get_symbol("tensor_from_dlpack");
    return f(managed_tensor, require_alignment, require_contiguous, out);
}

fn load_symbol(comptime T: type, library: dylib.Library, comptime symbol: [:0]const u8) LoadError!T {
    return library.lookup(T, symbol) orelse {
        log.err("missing symbol {s}: {s}", .{ symbol, dylib.error_message() });
        return error.TvmSymbolMissing;
    };
}
