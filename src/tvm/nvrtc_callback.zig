//! NVRTC compilation callback for TVM.
//!
//! Registers `tvm_callback_cuda_compile` and adapts TVM-generated CUDA source
//!  to the shared Zigrad NVRTC compiler.

const std = @import("std");
const builtin = @import("builtin");
const tvm_compile = @import("../c/tvm/compile.zig");
const cuda_nvrtc = @import("../cuda/nvrtc.zig");

const log = std.log.scoped(.@"zg/tvm_nvrtc");

pub const RegisterError = cuda_nvrtc.LoadError || tvm_compile.RegisterCudaCompilerError;

const CallbackState = struct {
    library_path: []u8,
    toolkit_root: []u8,
    glibc_include_dir: ?[]u8,
    gcc_include_dir: ?[]u8,
    gpu_arch: []u8,

    fn create(input: cuda_nvrtc.Config, gpu_arch: []const u8) std.mem.Allocator.Error!*CallbackState {
        const allocator = std.heap.c_allocator;
        const self = try allocator.create(CallbackState);
        errdefer allocator.destroy(self);

        self.library_path = try allocator.dupe(u8, input.library_path);
        errdefer allocator.free(self.library_path);

        self.toolkit_root = try allocator.dupe(u8, input.toolkit_root);
        errdefer allocator.free(self.toolkit_root);

        self.glibc_include_dir = null;
        self.gcc_include_dir = null;
        self.gpu_arch = undefined;

        if (input.glibc_include_dir) |path| {
            self.glibc_include_dir = try allocator.dupe(u8, path);
        }
        errdefer if (self.glibc_include_dir) |path| allocator.free(path);

        if (input.gcc_include_dir) |path| {
            self.gcc_include_dir = try allocator.dupe(u8, path);
        }
        errdefer if (self.gcc_include_dir) |path| allocator.free(path);

        self.gpu_arch = try allocator.dupe(u8, gpu_arch);
        return self;
    }

    const Snapshot = struct {
        config: cuda_nvrtc.Config,
        gpu_arch: []const u8,
    };

    fn snapshot(self: *const CallbackState) Snapshot {
        return .{
            .config = .{
                .library_path = self.library_path,
                .toolkit_root = self.toolkit_root,
                .glibc_include_dir = self.glibc_include_dir,
                .gcc_include_dir = self.gcc_include_dir,
            },
            .gpu_arch = self.gpu_arch,
        };
    }

    fn destroy(self: *CallbackState) void {
        const allocator = std.heap.c_allocator;
        allocator.free(self.library_path);
        allocator.free(self.toolkit_root);
        if (self.glibc_include_dir) |path| allocator.free(path);
        if (self.gcc_include_dir) |path| allocator.free(path);
        allocator.free(self.gpu_arch);
        allocator.destroy(self);
    }
};

/// Register the NVRTC compilation callback with TVM.
///
/// Registration copies configuration and architecture until TVM calls its destructor.
///  Registration must precede TVM CUDA compilation.
pub fn register(
    config: cuda_nvrtc.Config,
    gpu_arch: []const u8,
) RegisterError!void {
    try cuda_nvrtc.ensure_available(config);

    const state = try CallbackState.create(config, gpu_arch);
    try tvm_compile.register_cuda_compiler(
        state,
        compile_cuda,
        destroy_callback_state,
    );
    log.info("registered tvm_callback_cuda_compile callback", .{});
}

fn destroy_callback_state(handle: ?*anyopaque) void {
    const state: *CallbackState = @ptrCast(@alignCast(handle orelse return));
    state.destroy();
}

fn compile_cuda(
    handle: ?*anyopaque,
    allocator: std.mem.Allocator,
    original_code: []const u8,
) ![]const u8 {
    const state: *const CallbackState = @ptrCast(@alignCast(handle orelse {
        log.err("NVRTC callback has no configuration", .{});
        return error.MissingCallbackState;
    }));

    const preview_len = @min(original_code.len, 500);
    log.debug("CUDA code preview ({d} bytes total):\n{s}...", .{ original_code.len, original_code[0..preview_len] });

    // Remove host includes that this NVRTC source path cannot compile.
    //
    // `cuda.h` reaches the host C library, and `cstdint` requires C++ standard
    //  library headers.
    var filtered_code: std.ArrayList(u8) = .empty;
    defer filtered_code.deinit(allocator);

    var lines = std.mem.splitSequence(u8, original_code, "\n");
    while (lines.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t\r");
        if (std.mem.eql(u8, trimmed, "#include <cuda.h>") or
            std.mem.eql(u8, trimmed, "#include <cstdint>"))
        {
            log.debug("Stripped: {s}", .{trimmed});
            continue;
        }
        try filtered_code.appendSlice(allocator, line);
        try filtered_code.append(allocator, '\n');
    }
    const patched_code = filtered_code.items;

    const patched_preview_len = @min(patched_code.len, 300);
    log.debug("Filtered code preview ({d} bytes total):\n{s}...", .{ patched_code.len, patched_code[0..patched_preview_len] });

    return compile_with_nvrtc(
        allocator,
        patched_code,
        state.snapshot(),
    ) catch |err| {
        log.err("NVRTC compilation failed: {s}", .{@errorName(err)});
        return err;
    };
}

/// Compile CUDA source to PTX with the configured include paths.
///
/// The caller frees the returned string.
fn compile_with_nvrtc(
    allocator: std.mem.Allocator,
    code: []const u8,
    snapshot: CallbackState.Snapshot,
) cuda_nvrtc.CompileError![]const u8 {
    const defines: []const []const u8 = switch (builtin.target.cpu.arch) {
        .x86_64 => &.{"__x86_64__"},
        else => &.{},
    };
    const ptx = try cuda_nvrtc.compile(allocator, code, snapshot.config, .{
        .gpu_arch = snapshot.gpu_arch,
        .program_name = "tvm_kernel.cu",
        .defines = defines,
    });
    if (ptx.len == 0) return error.NvrtcGetPtxFailed;
    return ptx[0 .. ptx.len - 1];
}

test "CallbackState owns configuration and architecture copies" {
    var architecture = [_]u8{ 's', 'm', '_', '8', '0' };
    const state = try CallbackState.create(.{ .library_path = "libnvrtc.so", .toolkit_root = "cuda", .glibc_include_dir = "include" }, &architecture);
    defer state.destroy();
    architecture[4] = '9';
    const snapshot = state.snapshot();
    try std.testing.expectEqualStrings("sm_80", snapshot.gpu_arch);
    try std.testing.expectEqualStrings("include", snapshot.config.glibc_include_dir.?);
    try std.testing.expect(snapshot.config.gcc_include_dir == null);
}
