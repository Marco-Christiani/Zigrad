//! NVRTC ABI adapter for runtime CUDA source compilation.
//!
//! Translated declarations, program handles, result codes, and C function
//!  signatures remain private.

const std = @import("std");
const dylib = @import("../dylib.zig");

const log = std.log.scoped(.@"zg/cuda_nvrtc");

const Impl = @import("c-nvrtc");

const nvrtcProgram = Impl.nvrtcProgram;
const nvrtcResult = Impl.nvrtcResult;
const NVRTC_SUCCESS = Impl.NVRTC_SUCCESS;

const FnCreateProgram = *const fn (*nvrtcProgram, ?[*:0]const u8, ?[*:0]const u8, c_int, ?[*]const [*:0]const u8, ?[*]const [*:0]const u8) callconv(.c) nvrtcResult;
const FnDestroyProgram = *const fn (*nvrtcProgram) callconv(.c) nvrtcResult;
const FnCompileProgram = *const fn (nvrtcProgram, c_int, ?[*]const [*:0]const u8) callconv(.c) nvrtcResult;
const FnGetPTXSize = *const fn (nvrtcProgram, *usize) callconv(.c) nvrtcResult;
const FnGetPTX = *const fn (nvrtcProgram, [*]u8) callconv(.c) nvrtcResult;
const FnGetProgramLogSize = *const fn (nvrtcProgram, *usize) callconv(.c) nvrtcResult;
const FnGetProgramLog = *const fn (nvrtcProgram, [*]u8) callconv(.c) nvrtcResult;

var fn_create_program: ?FnCreateProgram = null;
var fn_destroy_program: ?FnDestroyProgram = null;
var fn_compile_program: ?FnCompileProgram = null;
var fn_get_ptx_size: ?FnGetPTXSize = null;
var fn_get_ptx: ?FnGetPTX = null;
var fn_get_program_log_size: ?FnGetProgramLogSize = null;
var fn_get_program_log: ?FnGetProgramLog = null;

var nvrtc_library: ?dylib.Library = null;
var nvrtc_path: ?[]const u8 = null;

pub const LoadError = std.mem.Allocator.Error || error{
    NvrtcLibraryNotFound,
    NvrtcLibraryConflict,
    NvrtcSymbolMissing,
};

pub const CompileError = LoadError || error{
    NvrtcCreateProgramFailed,
    NvrtcCompileFailed,
    NvrtcGetPtxFailed,
    OutOfMemory,
};

pub const CompileOptions = struct {
    program_name: []const u8 = "kernel.cu",
    compiler_options: []const []const u8 = &.{},
};

pub fn ensure_loaded(path: []const u8) LoadError!void {
    if (nvrtc_path) |loaded_path| {
        std.debug.assert(nvrtc_library != null);
        if (!std.mem.eql(u8, loaded_path, path)) {
            log.err("NVRTC is already loaded from '{s}', cannot load '{s}'", .{ loaded_path, path });
            return error.NvrtcLibraryConflict;
        }
        return;
    }

    var library = dylib.Library.open(std.heap.page_allocator, path, .{
        .visibility = .global,
    }) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        error.OpenFailed => {
            log.err("failed to load NVRTC '{s}': {s}", .{ path, dylib.error_message() });
            return error.NvrtcLibraryNotFound;
        },
    };
    errdefer library.close();

    fn_create_program = try load_symbol(FnCreateProgram, library, "nvrtcCreateProgram");
    fn_destroy_program = try load_symbol(FnDestroyProgram, library, "nvrtcDestroyProgram");
    fn_compile_program = try load_symbol(FnCompileProgram, library, "nvrtcCompileProgram");
    fn_get_ptx_size = try load_symbol(FnGetPTXSize, library, "nvrtcGetPTXSize");
    fn_get_ptx = try load_symbol(FnGetPTX, library, "nvrtcGetPTX");
    fn_get_program_log_size = try load_symbol(FnGetProgramLogSize, library, "nvrtcGetProgramLogSize");
    fn_get_program_log = try load_symbol(FnGetProgramLog, library, "nvrtcGetProgramLog");

    const retained_path = try std.heap.page_allocator.dupe(u8, path);
    nvrtc_library = library;
    nvrtc_path = retained_path;
}

/// Compiles CUDA source into a PTX buffer allocated by `allocator`.
///
/// The returned buffer includes NVRTC's trailing null byte.
pub fn compile(
    allocator: std.mem.Allocator,
    library_path: []const u8,
    source: []const u8,
    options: CompileOptions,
) CompileError![]u8 {
    try ensure_loaded(library_path);

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const temp = arena.allocator();

    const source_z = try temp.dupeSentinel(u8, source, 0);
    const name_z = try temp.dupeSentinel(u8, options.program_name, 0);
    const raw_options = try temp.alloc([*:0]const u8, options.compiler_options.len);
    for (options.compiler_options, raw_options) |option, *raw_option| {
        raw_option.* = (try temp.dupeSentinel(u8, option, 0)).ptr;
    }

    var program: nvrtcProgram = std.mem.zeroes(nvrtcProgram);
    const create_result = create_program(
        &program,
        source_z.ptr,
        name_z.ptr,
        0,
        null,
        null,
    );
    if (create_result != NVRTC_SUCCESS) {
        log.err("nvrtcCreateProgram failed: {d}", .{create_result});
        return error.NvrtcCreateProgramFailed;
    }
    defer _ = destroy_program(&program);

    const compile_result = compile_program(
        program,
        @intCast(raw_options.len),
        if (raw_options.len == 0) null else raw_options.ptr,
    );

    var log_size: usize = 0;
    if (get_program_log_size(program, &log_size) == NVRTC_SUCCESS and log_size > 1) {
        const compile_log = try temp.alloc(u8, log_size);
        if (get_program_log(program, compile_log.ptr) == NVRTC_SUCCESS) {
            const content = std.mem.trimEnd(u8, compile_log[0 .. log_size - 1], "\x00");
            if (content.len != 0) {
                log.debug("NVRTC log ({d} bytes): {s}", .{
                    content.len,
                    content[0..@min(content.len, 500)],
                });
            }
        }
    }

    if (compile_result != NVRTC_SUCCESS) {
        log.err("nvrtcCompileProgram failed: {d}", .{compile_result});
        return error.NvrtcCompileFailed;
    }

    var ptx_size: usize = 0;
    if (get_ptx_size(program, &ptx_size) != NVRTC_SUCCESS or ptx_size == 0) {
        return error.NvrtcGetPtxFailed;
    }

    const ptx = try allocator.alloc(u8, ptx_size);
    errdefer allocator.free(ptx);
    if (get_ptx(program, ptx.ptr) != NVRTC_SUCCESS) {
        return error.NvrtcGetPtxFailed;
    }
    return ptx;
}

fn create_program(
    prog: *nvrtcProgram,
    src: ?[*:0]const u8,
    name: ?[*:0]const u8,
    num_headers: c_int,
    headers: ?[*]const [*:0]const u8,
    include_names: ?[*]const [*:0]const u8,
) nvrtcResult {
    const f = fn_create_program orelse return 1;
    return f(prog, src, name, num_headers, headers, include_names);
}

fn destroy_program(prog: *nvrtcProgram) nvrtcResult {
    const f = fn_destroy_program orelse return 1;
    return f(prog);
}

fn compile_program(prog: nvrtcProgram, num_options: c_int, options: ?[*]const [*:0]const u8) nvrtcResult {
    const f = fn_compile_program orelse return 1;
    return f(prog, num_options, options);
}

fn get_ptx_size(prog: nvrtcProgram, ptx_size_ret: *usize) nvrtcResult {
    const f = fn_get_ptx_size orelse return 1;
    return f(prog, ptx_size_ret);
}

fn get_ptx(prog: nvrtcProgram, ptx: [*]u8) nvrtcResult {
    const f = fn_get_ptx orelse return 1;
    return f(prog, ptx);
}

fn get_program_log_size(prog: nvrtcProgram, log_size_ret: *usize) nvrtcResult {
    const f = fn_get_program_log_size orelse return 1;
    return f(prog, log_size_ret);
}

fn get_program_log(prog: nvrtcProgram, output_log: [*]u8) nvrtcResult {
    const f = fn_get_program_log orelse return 1;
    return f(prog, output_log);
}

fn load_symbol(comptime T: type, library: dylib.Library, comptime symbol: [:0]const u8) LoadError!T {
    return library.lookup(T, symbol) orelse {
        log.err("missing NVRTC symbol {s}: {s}", .{ symbol, dylib.error_message() });
        return error.NvrtcSymbolMissing;
    };
}
