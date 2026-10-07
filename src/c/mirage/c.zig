//! Runtime-loaded declarations for the Zigrad Mirage adapter.
const std = @import("std");
const dylib = @import("../dylib.zig");

const log = std.log.scoped(.@"zg/mirage_cffi");
const C = @import("c-mirage");

pub const abi_version: u32 = C.ZG_MIRAGE_ABI_VERSION;
pub const required_capabilities: u64 =
    C.ZG_MIRAGE_CAP_SYMBOLIC_OPTIMIZE |
    C.ZG_MIRAGE_CAP_CUDA_TRANSPILE |
    C.ZG_MIRAGE_CAP_EXPLICIT_CHECKPOINT |
    C.ZG_MIRAGE_CAP_LAUNCH_METADATA;

pub const MirageStatus = C.zg_mirage_status_t;
pub const status_ok = C.ZG_MIRAGE_STATUS_OK;
pub const status_invalid_argument = C.ZG_MIRAGE_STATUS_INVALID_ARGUMENT;
pub const status_unsupported = C.ZG_MIRAGE_STATUS_UNSUPPORTED;
pub const status_not_found = C.ZG_MIRAGE_STATUS_NOT_FOUND;
pub const status_external_failure = C.ZG_MIRAGE_STATUS_EXTERNAL_FAILURE;
pub const status_out_of_memory = C.ZG_MIRAGE_STATUS_OUT_OF_MEMORY;
pub const status_abi_mismatch = C.ZG_MIRAGE_STATUS_ABI_MISMATCH;

pub const MirageSession = C.zg_mirage_session_t;
pub const MirageGraph = C.zg_mirage_graph_t;
pub const MirageSource = C.zg_mirage_source_t;
pub const MirageTensor = C.zg_mirage_tensor_t;

pub const max_rank = C.ZG_MIRAGE_MAX_RANK;
pub const TensorSpec = C.zg_mirage_tensor_spec_t;
pub const OptimizeOptions = C.zg_mirage_optimize_options_t;
pub const TranspileOptions = C.zg_mirage_transpile_options_t;
pub const RawKernelMeta = C.zg_mirage_kernel_meta_t;
pub const RawKernelArg = C.zg_mirage_kernel_arg_t;

pub const dtype_f16 = C.ZG_MIRAGE_DTYPE_F16;
pub const dtype_bf16 = C.ZG_MIRAGE_DTYPE_BF16;
pub const dtype_f32 = C.ZG_MIRAGE_DTYPE_F32;
pub const dtype_f64 = C.ZG_MIRAGE_DTYPE_F64;

pub const unary_exp = C.ZG_MIRAGE_UNARY_EXP;
pub const unary_sqrt = C.ZG_MIRAGE_UNARY_SQRT;
pub const unary_silu = C.ZG_MIRAGE_UNARY_SILU;
pub const unary_gelu = C.ZG_MIRAGE_UNARY_GELU;
pub const unary_relu = C.ZG_MIRAGE_UNARY_RELU;
pub const unary_log = C.ZG_MIRAGE_UNARY_LOG;

pub const binary_add = C.ZG_MIRAGE_BINARY_ADD;
pub const binary_mul = C.ZG_MIRAGE_BINARY_MUL;
pub const binary_div = C.ZG_MIRAGE_BINARY_DIV;
pub const binary_pow = C.ZG_MIRAGE_BINARY_POW;

pub const search_default = C.ZG_MIRAGE_SEARCH_DEFAULT;
pub const search_attention = C.ZG_MIRAGE_SEARCH_ATTENTION;
pub const search_lora = C.ZG_MIRAGE_SEARCH_LORA;
pub const search_mlp = C.ZG_MIRAGE_SEARCH_MLP;

pub const arg_input = C.ZG_MIRAGE_ARG_INPUT;
pub const arg_output = C.ZG_MIRAGE_ARG_OUTPUT;
pub const arg_workspace = C.ZG_MIRAGE_ARG_WORKSPACE;

const FnAbiVersion = *const fn () callconv(.c) u32;
const FnCapabilities = *const fn () callconv(.c) u64;
const FnStatusString = *const fn (MirageStatus) callconv(.c) [*:0]const u8;
const FnLastError = *const fn () callconv(.c) [*:0]const u8;

const FnSessionCreate = *const fn (i32, *?*MirageSession) callconv(.c) MirageStatus;
const FnSessionDestroy = *const fn (?*MirageSession) callconv(.c) void;

const FnGraphCreate = *const fn (*?*MirageGraph) callconv(.c) MirageStatus;
const FnGraphDestroy = *const fn (?*MirageGraph) callconv(.c) void;
const FnGraphNewInput = *const fn (?*MirageGraph, *const TensorSpec, *MirageTensor) callconv(.c) MirageStatus;
const FnGraphMatmul = *const fn (?*MirageGraph, MirageTensor, MirageTensor, *MirageTensor) callconv(.c) MirageStatus;
const FnGraphUnary = *const fn (?*MirageGraph, c_uint, MirageTensor, *MirageTensor) callconv(.c) MirageStatus;
const FnGraphBinary = *const fn (?*MirageGraph, c_uint, MirageTensor, MirageTensor, *MirageTensor) callconv(.c) MirageStatus;
const FnGraphReduction = *const fn (?*MirageGraph, MirageTensor, i32, i32, *MirageTensor) callconv(.c) MirageStatus;
const FnGraphRmsNorm = *const fn (?*MirageGraph, MirageTensor, i32, *MirageTensor) callconv(.c) MirageStatus;
const FnGraphMarkOutput = *const fn (?*MirageGraph, MirageTensor) callconv(.c) MirageStatus;

const FnOptimize = *const fn (
    ?*MirageSession,
    ?*const MirageGraph,
    ?*const OptimizeOptions,
    *?*MirageGraph,
) callconv(.c) MirageStatus;

const FnTranspile = *const fn (
    ?*const MirageGraph,
    ?*const TranspileOptions,
    *?*MirageSource,
) callconv(.c) MirageStatus;
const FnSourceDestroy = *const fn (?*MirageSource) callconv(.c) void;
const FnSourceCode = *const fn (?*const MirageSource) callconv(.c) [*:0]const u8;
const FnSourceCodeLen = *const fn (?*const MirageSource) callconv(.c) usize;
const FnSourceWorkspaceSize = *const fn (?*const MirageSource) callconv(.c) usize;
const FnSourceNumKernels = *const fn (?*const MirageSource) callconv(.c) usize;
const FnSourceKernelMeta = *const fn (?*const MirageSource, usize, *RawKernelMeta) callconv(.c) MirageStatus;
const FnSourceKernelNumArgs = *const fn (?*const MirageSource, usize) callconv(.c) usize;
const FnSourceKernelArg = *const fn (?*const MirageSource, usize, usize, *RawKernelArg) callconv(.c) MirageStatus;

const Symbols = struct {
    abi_version: FnAbiVersion,
    capabilities: FnCapabilities,
    status_string: FnStatusString,
    last_error: FnLastError,
    session_create: FnSessionCreate,
    session_destroy: FnSessionDestroy,
    graph_create: FnGraphCreate,
    graph_destroy: FnGraphDestroy,
    graph_new_input: FnGraphNewInput,
    graph_matmul: FnGraphMatmul,
    graph_unary: FnGraphUnary,
    graph_binary: FnGraphBinary,
    graph_reduction: FnGraphReduction,
    graph_rms_norm: FnGraphRmsNorm,
    graph_mark_output: FnGraphMarkOutput,
    optimize: FnOptimize,
    transpile: FnTranspile,
    source_destroy: FnSourceDestroy,
    source_code: FnSourceCode,
    source_code_len: FnSourceCodeLen,
    source_workspace_size: FnSourceWorkspaceSize,
    source_num_kernels: FnSourceNumKernels,
    source_kernel_meta: FnSourceKernelMeta,
    source_kernel_num_args: FnSourceKernelNumArgs,
    source_kernel_arg: FnSourceKernelArg,

    fn load(library: dylib.Library) LoadError!Symbols {
        var loaded: Symbols = undefined;
        inline for (@field(@typeInfo(Symbols), @tagName(@typeInfo(Symbols))).field_names, @field(@typeInfo(Symbols), @tagName(@typeInfo(Symbols))).field_types) |field_name, field_type| {
            const name = comptime "zg_mirage_" ++ field_name;
            @field(loaded, field_name) = try load_symbol(field_type, library, name);
        }
        return loaded;
    }
};

var symbols: ?Symbols = null;

pub const LoadError = error{MirageSymbolMissing};

/// Install the complete adapter symbol table.
///
/// The caller serializes process initialization and keeps `library` open.
pub fn install_symbols(library: dylib.Library) LoadError!void {
    if (symbols != null) return;
    symbols = try Symbols.load(library);
}

fn get_symbol(comptime name: []const u8) ?@TypeOf(@field(@as(Symbols, undefined), name)) {
    const loaded = symbols orelse return null;
    return @field(loaded, name);
}

pub fn mirage_abi_version() u32 {
    const f = get_symbol("abi_version") orelse return 0;
    return f();
}

pub fn mirage_capabilities() u64 {
    const f = get_symbol("capabilities") orelse return 0;
    return f();
}

pub fn mirage_status_string(status: MirageStatus) [*:0]const u8 {
    const f = get_symbol("status_string") orelse return "unknown_status";
    return f(status);
}

pub fn mirage_last_error() [*:0]const u8 {
    const f = get_symbol("last_error") orelse return "";
    return f();
}

pub fn mirage_session_create(ordinal: i32, out: *?*MirageSession) MirageStatus {
    const f = get_symbol("session_create") orelse return status_external_failure;
    return f(ordinal, out);
}

pub fn mirage_session_destroy(session: ?*MirageSession) void {
    const f = get_symbol("session_destroy") orelse return;
    f(session);
}

pub fn mirage_graph_create(out: *?*MirageGraph) MirageStatus {
    const f = get_symbol("graph_create") orelse return status_external_failure;
    return f(out);
}

pub fn mirage_graph_destroy(graph: ?*MirageGraph) void {
    const f = get_symbol("graph_destroy") orelse return;
    f(graph);
}

pub fn mirage_graph_new_input(graph: ?*MirageGraph, spec: *const TensorSpec, out: *MirageTensor) MirageStatus {
    const f = get_symbol("graph_new_input") orelse return status_external_failure;
    return f(graph, spec, out);
}

pub fn mirage_graph_matmul(graph: ?*MirageGraph, lhs: MirageTensor, rhs: MirageTensor, out: *MirageTensor) MirageStatus {
    const f = get_symbol("graph_matmul") orelse return status_external_failure;
    return f(graph, lhs, rhs, out);
}

pub fn mirage_graph_unary(graph: ?*MirageGraph, op: c_uint, input: MirageTensor, out: *MirageTensor) MirageStatus {
    const f = get_symbol("graph_unary") orelse return status_external_failure;
    return f(graph, op, input, out);
}

pub fn mirage_graph_binary(graph: ?*MirageGraph, op: c_uint, lhs: MirageTensor, rhs: MirageTensor, out: *MirageTensor) MirageStatus {
    const f = get_symbol("graph_binary") orelse return status_external_failure;
    return f(graph, op, lhs, rhs, out);
}

pub fn mirage_graph_reduction(graph: ?*MirageGraph, input: MirageTensor, dim: i32, factor: i32, out: *MirageTensor) MirageStatus {
    const f = get_symbol("graph_reduction") orelse return status_external_failure;
    return f(graph, input, dim, factor, out);
}

pub fn mirage_graph_rms_norm(graph: ?*MirageGraph, input: MirageTensor, normalized_size: i32, out: *MirageTensor) MirageStatus {
    const f = get_symbol("graph_rms_norm") orelse return status_external_failure;
    return f(graph, input, normalized_size, out);
}

pub fn mirage_graph_mark_output(graph: ?*MirageGraph, tensor: MirageTensor) MirageStatus {
    const f = get_symbol("graph_mark_output") orelse return status_external_failure;
    return f(graph, tensor);
}

pub fn mirage_optimize(
    session: ?*MirageSession,
    graph: ?*const MirageGraph,
    options: ?*const OptimizeOptions,
    out: *?*MirageGraph,
) MirageStatus {
    const f = get_symbol("optimize") orelse return status_external_failure;
    return f(session, graph, options, out);
}

pub fn mirage_transpile(
    graph: ?*const MirageGraph,
    options: ?*const TranspileOptions,
    out: *?*MirageSource,
) MirageStatus {
    const f = get_symbol("transpile") orelse return status_external_failure;
    return f(graph, options, out);
}

pub fn mirage_source_destroy(source: ?*MirageSource) void {
    const f = get_symbol("source_destroy") orelse return;
    f(source);
}

pub fn mirage_source_code(source: ?*const MirageSource) [*:0]const u8 {
    const f = get_symbol("source_code") orelse return "";
    return f(source);
}

pub fn mirage_source_code_len(source: ?*const MirageSource) usize {
    const f = get_symbol("source_code_len") orelse return 0;
    return f(source);
}

pub fn mirage_source_workspace_size(source: ?*const MirageSource) usize {
    const f = get_symbol("source_workspace_size") orelse return 0;
    return f(source);
}

pub fn mirage_source_num_kernels(source: ?*const MirageSource) usize {
    const f = get_symbol("source_num_kernels") orelse return 0;
    return f(source);
}

pub fn mirage_source_kernel_meta(source: ?*const MirageSource, index: usize, out: *RawKernelMeta) MirageStatus {
    const f = get_symbol("source_kernel_meta") orelse return status_external_failure;
    return f(source, index, out);
}

pub fn mirage_source_kernel_num_args(source: ?*const MirageSource, kernel_index: usize) usize {
    const f = get_symbol("source_kernel_num_args") orelse return 0;
    return f(source, kernel_index);
}

pub fn mirage_source_kernel_arg(
    source: ?*const MirageSource,
    kernel_index: usize,
    arg_index: usize,
    out: *RawKernelArg,
) MirageStatus {
    const f = get_symbol("source_kernel_arg") orelse return status_external_failure;
    return f(source, kernel_index, arg_index, out);
}

fn load_symbol(comptime T: type, library: dylib.Library, comptime symbol: [:0]const u8) LoadError!T {
    return library.lookup(T, symbol) orelse {
        log.err("missing symbol {s}: {s}", .{ symbol, dylib.error_message() });
        return error.MirageSymbolMissing;
    };
}
