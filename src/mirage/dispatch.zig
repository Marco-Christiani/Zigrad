const std = @import("std");
const device = @import("device");
const kernel = @import("kernel");
const mirage = @import("../c/mirage/api.zig");
const cuda_driver = @import("../cuda/driver.zig");
const cuda_nvrtc = @import("../cuda/nvrtc.zig");
const artifact = @import("artifact.zig");
const Artifact = artifact.Artifact;
const KernelDesc = artifact.KernelDesc;
const CompileConfig = @import("config.zig").CompileConfig;
const TypedPtr = @import("rtti").TypedPtr;

const log = std.log.scoped(.@"zg/mirage_dispatch");

/// A compiled CUDA module with cached function handles.
const CompiledModule = struct {
    module: *cuda_driver.Module,
    /// One entry per kernel in the artifact, same order.
    funcs: []*cuda_driver.Function,

    fn deinit(self: *CompiledModule, allocator: std.mem.Allocator) void {
        self.module.deinit();
        allocator.free(self.funcs);
    }
};

pub const MirageDispatchState = struct {
    allocator: std.mem.Allocator,
    compile_config: CompileConfig,

    /// Cache of compiled modules keyed by a FNV hash of the artifact data.
    cache: std.AutoHashMap(u64, CompiledModule),
    cache_mutex: std.Io.Mutex = .init,

    /// Initialize dispatch with explicit NVRTC compilation inputs.
    ///
    /// Borrowed strings in `compile_config` must outlive the returned state.
    pub fn init(
        allocator: std.mem.Allocator,
        compile_config: CompileConfig,
    ) MirageDispatchState {
        return .{
            .allocator = allocator,
            .compile_config = compile_config,
            .cache = std.AutoHashMap(u64, CompiledModule).init(allocator),
        };
    }

    pub fn deinit(self: *MirageDispatchState) void {
        var it = self.cache.valueIterator();
        while (it.next()) |mod| {
            mod.deinit(self.allocator);
        }
        self.cache.deinit();
    }

    pub fn dispatch(
        provider_ctx: TypedPtr,
        artifact_data: []const u8,
        kernel_key: []const u8,
        ctx: kernel.DispatchContext,
    ) kernel.DispatchError!void {
        const self = provider_ctx.cast(MirageDispatchState);
        self.dispatch_impl(artifact_data, kernel_key, ctx) catch |err| {
            log.err("mirage dispatch failed for '{s}': {s}", .{ kernel_key, @errorName(err) });
            return err;
        };
    }

    fn dispatch_impl(
        self: *MirageDispatchState,
        artifact_data: []const u8,
        kernel_key: []const u8,
        ctx: kernel.DispatchContext,
    ) kernel.DispatchError!void {
        if (!ctx.device.platform.eql(.cuda)) return error.UnsupportedDevice;

        const art = artifact.decode(self.allocator, artifact_data) catch |err| {
            if (err == error.OutOfMemory) return error.OutOfMemory;
            log.err("failed to decode mirage artifact for '{s}'", .{kernel_key});
            return error.DispatchFailed;
        };
        defer {
            for (art.kernels) |k| self.allocator.free(k.args);
            self.allocator.free(art.kernels);
        }

        if (art.kernels.len == 0) {
            log.err("mirage artifact for '{s}' has no kernels", .{kernel_key});
            return error.DispatchFailed;
        }

        var hasher = std.hash.Fnv1a_64.init();
        hasher.update(artifact_data);
        hasher.update(ctx.device.platform.name);
        hasher.update(std.mem.asBytes(&ctx.device.ordinal));
        const hash = hasher.final();
        const compiled = try self.get_or_compile(hash, art, ctx.device);

        for (art.kernels, 0..) |kd, ki| {
            try self.launch_kernel(compiled.funcs[ki], kd, ctx);
        }
    }

    fn get_or_compile(
        self: *MirageDispatchState,
        hash: u64,
        art: Artifact,
        selected_device: device.Device,
    ) kernel.DispatchError!CompiledModule {
        std.Io.Threaded.mutexLock(&self.cache_mutex);
        defer std.Io.Threaded.mutexUnlock(&self.cache_mutex);

        if (self.cache.get(hash)) |mod| return mod;

        const capability = cuda_driver.compute_capability(
            selected_device.ordinal,
        ) catch |err| {
            log.err("CUDA target detection failed: {s}", .{@errorName(err)});
            return error.DispatchFailed;
        };
        if (capability.major <= 0 or capability.minor < 0) {
            return error.DispatchFailed;
        }
        var gpu_arch_buffer: [16]u8 = undefined;
        const gpu_arch = std.fmt.bufPrint(
            &gpu_arch_buffer,
            "sm_{d}{d}",
            .{ capability.major, capability.minor },
        ) catch return error.DispatchFailed;
        log.info(
            "resolved CUDA target for device {d}: {s}",
            .{ selected_device.ordinal, gpu_arch },
        );

        const ptx = compile_to_ptx(
            self.allocator,
            art.source,
            self.compile_config,
            gpu_arch,
        ) catch |err| {
            if (err == error.OutOfMemory) return error.OutOfMemory;
            log.err("Compilation to PTX failed: {s}", .{@errorName(err)});
            return error.DispatchFailed;
        };
        defer self.allocator.free(ptx);

        const module = cuda_driver.Module.load(ptx) catch |err| {
            log.err("CUDA module load failed: {s}", .{@errorName(err)});
            return error.DispatchFailed;
        };
        errdefer module.deinit();

        const funcs = try self.allocator.alloc(*cuda_driver.Function, art.kernels.len);
        errdefer self.allocator.free(funcs);

        for (art.kernels, 0..) |kd, ki| {
            const name_z = try self.allocator.dupeZ(u8, kd.func_name);
            defer self.allocator.free(name_z);

            funcs[ki] = module.function(name_z) catch |err| {
                log.err("CUDA function lookup failed for '{s}': {s}", .{
                    kd.func_name,
                    @errorName(err),
                });
                return error.DispatchFailed;
            };

            if (kd.smem_bytes > 48 * 1024) {
                funcs[ki].set_max_dynamic_shared_memory(kd.smem_bytes) catch |err| {
                    log.err("CUDA shared-memory configuration failed for '{s}': {s}", .{
                        kd.func_name,
                        @errorName(err),
                    });
                    return error.DispatchFailed;
                };
            }
        }

        const compiled = CompiledModule{ .module = module, .funcs = funcs };
        self.cache.put(hash, compiled) catch |e| {
            // Compiled kernels remain usable when the lookup cache cannot grow.
            log.warn("OOM on cache put: {s}", .{@errorName(e)});
        };
        return compiled;
    }

    fn launch_kernel(
        self: *MirageDispatchState,
        func: *cuda_driver.Function,
        kd: KernelDesc,
        ctx: kernel.DispatchContext,
    ) kernel.DispatchError!void {
        // Each CUDA argument points to storage containing its device pointer.
        const arg_ptrs = try self.allocator.alloc(?*anyopaque, kd.args.len);
        defer self.allocator.free(arg_ptrs);

        const ptr_values = try self.allocator.alloc(*anyopaque, kd.args.len);
        defer self.allocator.free(ptr_values);

        for (kd.args, 0..) |arg, i| {
            switch (arg.source) {
                .input => {
                    if (arg.index_or_offset >= ctx.inputs.len) {
                        log.err("kernel arg input index {d} out of range (have {d} inputs)", .{ arg.index_or_offset, ctx.inputs.len });
                        return error.DispatchFailed;
                    }
                    ptr_values[i] = ctx.inputs[arg.index_or_offset].data;
                },
                .output => {
                    if (arg.index_or_offset >= ctx.outputs.len) {
                        log.err("kernel arg output index {d} out of range (have {d} outputs)", .{ arg.index_or_offset, ctx.outputs.len });
                        return error.DispatchFailed;
                    }
                    ptr_values[i] = ctx.outputs[arg.index_or_offset].data;
                },
                .buf => {
                    if (ctx.workspace == null) {
                        log.err("kernel arg requires workspace buffer but none provided", .{});
                        return error.WorkspaceUnavailable;
                    }
                    const base: [*]u8 = @ptrCast(ctx.workspace.?);
                    ptr_values[i] = @ptrCast(base + arg.index_or_offset);
                },
            }
            arg_ptrs[i] = @ptrCast(&ptr_values[i]);
        }

        func.launch(.{
            .grid_dim = kd.grid_dim,
            .block_dim = kd.block_dim,
            .shared_memory_bytes = kd.smem_bytes,
            .stream = ctx.stream,
            .params = arg_ptrs,
        }) catch |err| {
            log.err("CUDA kernel launch failed: {s}", .{@errorName(err)});
            return error.DispatchFailed;
        };
    }
};

/// Compile Mirage CUDA source to PTX using shared NVRTC policy.
pub fn compile_to_ptx(
    allocator: std.mem.Allocator,
    source: []const u8,
    config: CompileConfig,
    gpu_arch: []const u8,
) cuda_nvrtc.CompileError![]const u8 {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const tmp = arena.allocator();

    var include_dirs: std.ArrayList([]const u8) = .empty;

    try include_dirs.append(
        tmp,
        try std.fmt.allocPrint(tmp, "{s}/include", .{config.sdk_root}),
    );
    try include_dirs.append(
        tmp,
        try std.fmt.allocPrint(
            tmp,
            "{s}/include/mirage/transpiler/runtime",
            .{config.sdk_root},
        ),
    );

    if (config.mirage_include_dir) |value| {
        try include_dirs.append(tmp, value);
        try include_dirs.append(
            tmp,
            try std.fmt.allocPrint(
                tmp,
                "{s}/mirage/transpiler/runtime",
                .{value},
            ),
        );
    }

    return try cuda_nvrtc.compile(allocator, source, config.cuda, .{
        .gpu_arch = gpu_arch,
        .program_name = "mirage_kernel.cu",
        .include_dirs = include_dirs.items,
        .defines = &.{"MIRAGE_BACKEND_USE_CUDA"},
    });
}

/// Filter Mirage transpiler output for NVRTC compilation.
///
/// Strips host-only code and replaces `runtime.h` with individual
///  device-compatible includes.
pub fn filter_source_for_nvrtc(
    allocator: std.mem.Allocator,
    source: []const u8,
) std.mem.Allocator.Error![]const u8 {
    var out: std.ArrayList(u8) = .empty;

    try out.appendSlice(allocator, "#define USE_NVSHMEM 0\n#define NUM_GPUS 1\n\n");

    var in_host_function = false;
    var brace_depth: i32 = 0;

    var lines = std.mem.splitScalar(u8, source, '\n');
    while (lines.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t\r");

        if (std.mem.startsWith(u8, trimmed, "#include <vector>") or
            std.mem.startsWith(u8, trimmed, "#include <cuda_runtime") or
            std.mem.startsWith(u8, trimmed, "#include <cublas"))
        {
            continue;
        }

        if (std.mem.eql(u8, trimmed, "#include \"runtime.h\"")) {
            try out.appendSlice(allocator,
                \\#include <cute/layout.hpp>
                \\#include <cute/tensor.hpp>
                \\#include <cutlass/cutlass.h>
                \\#include "config.h"
                \\#include "kernel/element_unary.h"
                \\#include "kernel/element_binary.h"
                \\#include "kernel/reduction.h"
                \\#include "threadblock/threadblock.h"
                \\#include "utils.h"
                \\
            );
            continue;
        }

        if (!in_host_function) {
            if (std.mem.startsWith(u8, trimmed, "static void _init()") or
                std.mem.startsWith(u8, trimmed, "static void _execute_mugraph(") or
                std.mem.startsWith(u8, trimmed, "extern \"C\" void execute_mugraph"))
            {
                in_host_function = true;
                brace_depth = 0;
                for (line) |ch| {
                    if (ch == '{') brace_depth += 1;
                    if (ch == '}') brace_depth -= 1;
                }
                if (brace_depth <= 0 and std.mem.indexOfScalar(u8, line, '{') != null) {
                    in_host_function = false;
                }
                continue;
            }
        }

        if (in_host_function) {
            for (line) |ch| {
                if (ch == '{') brace_depth += 1;
                if (ch == '}') brace_depth -= 1;
            }
            if (brace_depth <= 0) {
                in_host_function = false;
            }
            continue;
        }

        try out.appendSlice(allocator, line);
        try out.append(allocator, '\n');
    }

    return out.items;
}
