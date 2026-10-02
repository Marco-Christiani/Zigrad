//! TVM compilation orchestration.
//!
//! Combines TIR lowering, target-specific compilation, host/device splitting,
//!  and linking into shared libraries. Zigrad composes the `tvm/tir/`,
//!  `tvm/target/`, and `tvm/runtime/` subsystems here.
const std = @import("std");
const ffi = @import("ffi.zig");
const container = @import("container.zig");
const tir = @import("tir.zig");
const runtime = @import("runtime.zig");
const Value = ffi.Value;
const IRModule = tir.IRModule;
const Target = tir.Target;
const TirPass = tir.TirPass;
const TargetKind = @import("../../tvm/config.zig").TargetKind;
const RuntimeModule = runtime.RuntimeModule;

const log = std.log.scoped(.@"zg/tvm_compile");

/// CUDA source compiler used by TVM target code generation.
pub const CudaCompiler = *const fn (
    context: ?*anyopaque,
    allocator: std.mem.Allocator,
    source: []const u8,
) anyerror![]const u8;

/// Releases CUDA compiler context after TVM releases the callback.
pub const CudaCompilerDestructor = *const fn (context: ?*anyopaque) void;
/// Errors raised while registering TVM's CUDA source compiler.
pub const RegisterCudaCompilerError = std.mem.Allocator.Error || ffi.TvmError;

/// Register TVM's CUDA source compiler.
pub fn register_cuda_compiler(
    /// Context transferred to TVM, including when registration fails.
    context: ?*anyopaque,
    compiler: CudaCompiler,
    destructor: ?CudaCompilerDestructor,
) RegisterCudaCompilerError!void {
    const state = std.heap.c_allocator.create(CudaCompilerState) catch |err| {
        if (destructor) |deinit| deinit(context);
        return err;
    };
    state.* = .{
        .context = context,
        .compiler = compiler,
        .destructor = destructor,
    };

    var function = ffi.PackedFunction.create(
        state,
        CudaCompilerState.invoke,
        CudaCompilerState.destroy,
    ) catch |err| {
        CudaCompilerState.destroy(state);
        return err;
    };
    defer function.deinit();
    try ffi.set_global("tvm_callback_cuda_compile", &function, true);
}

const CudaCompilerState = struct {
    context: ?*anyopaque,
    compiler: CudaCompiler,
    destructor: ?CudaCompilerDestructor,

    fn invoke(
        context: ?*anyopaque,
        args: *const ffi.CallbackArgs,
    ) !ffi.CallbackOutput {
        const self: *CudaCompilerState = @ptrCast(@alignCast(context orelse
            return error.MissingCallbackState));
        if (args.len() != 2) return error.InvalidCallbackArity;

        var arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
        defer arena.deinit();
        const allocator = arena.allocator();
        const source = try args.get(0).dupe_string(allocator);
        try args.get(1).require_instance("target.Target");
        const compiled = try self.compiler(self.context, allocator, source);
        return .{ .owned = try ffi.OwnedValue.from_string(compiled) };
    }

    fn destroy(context: ?*anyopaque) void {
        const self: *CudaCompilerState = @ptrCast(@alignCast(context orelse return));
        if (self.destructor) |destructor| destructor(self.context);
        std.heap.c_allocator.destroy(self);
    }
};

/// Apply the TIR lowering pipeline and compile for one target.
///
/// Pass ordering follows `python/tvm/tir/pipeline.py::default_tir_pipeline`.
/// Host and device partitioning follows `python/tvm/tir/build.py::build`.
pub fn lower_and_compile(
    /// Allocator used by TVM packed calls.
    allocator: std.mem.Allocator,
    /// TIR module to lower. Each pass replaces this handle in place.
    ir_mod: *IRModule,
    /// Compilation target without its host target attached.
    target: Target,
    /// Selects host-only or CUDA host-device code generation.
    target_kind: TargetKind,
) !RuntimeModule {
    // MakePackedAPI requires target->GetHost() to return a host target.
    //  Without it, the function remains unchanged and buffer_map is not cleared.
    var host_target = switch (target_kind) {
        .cpu => target,
        .cuda => try Target.create(allocator, .cpu, 0),
    };
    defer if (target_kind == .cuda) host_target.deinit();

    var composite_target = try target.with_host(allocator, host_target);
    defer composite_target.deinit();

    try ir_mod.apply_pass(allocator, .{ .bind_target = .{ .target = composite_target } });
    try apply_pipeline(allocator, ir_mod, &default_tir_pipeline);

    switch (target_kind) {
        .cpu => {
            try finalize_host_module(allocator, ir_mod);
            return compile_cpu_module(allocator, ir_mod.*, target);
        },
        .cuda => return compile_cuda_module(allocator, ir_mod.*, target),
    }
}

const default_tir_pipeline = [_]TirPass{
    .lower_cross_thread_reduction,
    .lower_init_block,
    .plan_and_update_buffer_allocation,
    .convert_blocks_to_opaque,
    .lift_thread_binding,
    .manifest_shared_memory_local_stage,
    .{ .compact_buffer_alloc = .{ .is_strict = true } },
    .lower_auto_copy,
    .unify_thread_binding,
    .lower_match_buffer,
    .simplify,
    .inject_permuted_layout,
    .annotate_irregular_loop,
    .inject_software_pipeline,
    .transform_mma_buffer_layout,
    .lower_opaque_block,
    .flatten_buffer,
    .bf16_compute_legalize,
    .{ .narrow_data_type = .{ .target_bits = 32 } },
    .loop_partition,
    .{ .vectorize_loop = .{ .enable = true } },
    .inject_virtual_thread,
    .inject_double_buffer,
    .storage_rewrite,
    .hoist_if_then_else,
    .unroll_loop,
    .renormalize_split_pattern,
    .simplify,
    .remove_no_op,
    .rewrite_unsafe_select,
    .{ .common_subexpr_elim = .{ .enable_cse = true, .enable_equiv = false } },
    .{ .fp8_compute_legalize = .{ .promote_dtype = "float16" } },
    .verify_vtcm_limit,
    .lower_vtcm_alloc,
    .verify_memory,
    .annotate_entry_func,
    .{ .thread_sync = .{ .scope = "shared" } },
    .{ .thread_sync = .{ .scope = "shared.dyn" } },
    .{ .thread_sync = .{ .scope = "warp" } },
    .infer_fragment,
    .lower_thread_allreduce,
    .annotate_device_regions,
    .split_host_device,
    .merge_shared_memory_allocations,
    .make_packed_api,
    .fp8_storage_legalize,
    .bf16_storage_legalize,
    .lower_device_kernel_launch,
};

fn apply_pipeline(
    allocator: std.mem.Allocator,
    ir_mod: *IRModule,
    passes: []const TirPass,
) !void {
    for (passes) |pass| {
        ir_mod.apply_pass(allocator, pass) catch |err| {
            log.err("required TVM pass {s} failed", .{pass.name()});
            return err;
        };
    }
}

/// Compile a CPU runtime module through `target.build.llvm`.
fn compile_cpu_module(allocator: std.mem.Allocator, ir_mod: IRModule, target: Target) !RuntimeModule {
    return try ffi.call_global_take(RuntimeModule, allocator, "target.build.llvm", &.{
        ir_mod.as_value(),
        target.as_value(),
    });
}

/// Compile and link the host and device parts of a CUDA runtime module.
fn compile_cuda_module(allocator: std.mem.Allocator, ir_mod: IRModule, target: Target) !RuntimeModule {
    // `python/tvm/tir/build.py::split_host_device_mods` performs the same
    //  partition before target code generation.
    var device_mod = try filter_module(allocator, ir_mod, .device);
    defer device_mod.deinit();

    try apply_pipeline(allocator, &device_mod, &device_finalization_pipeline);

    var device_built = try ffi.call_global_take(
        RuntimeModule,
        allocator,
        "target.build.cuda",
        &.{ device_mod.as_value(), target.as_value() },
    );
    defer device_built.deinit();

    var host_mod = try filter_module(allocator, ir_mod, .host);
    defer host_mod.deinit();

    try finalize_host_module(allocator, &host_mod);

    var host_target = try Target.create(allocator, .cpu, 0);
    defer host_target.deinit();
    var host_built = try ffi.call_global_take(
        RuntimeModule,
        allocator,
        "target.build.llvm",
        &.{ host_mod.as_value(), host_target.as_value() },
    );
    errdefer host_built.deinit();

    try ffi.call_global_void(allocator, "ffi.ModuleImportModule", &.{
        host_built.as_value(),
        device_built.as_value(),
    });

    return host_built;
}

fn finalize_host_module(allocator: std.mem.Allocator, ir_mod: *IRModule) !void {
    try apply_pipeline(allocator, ir_mod, &host_finalization_pipeline);
}

const host_finalization_pipeline = [_]TirPass{
    .lower_tvm_builtin,
    .lower_custom_datatypes,
    .lower_intrin,
    .lower_device_storage_access_info,
    .combine_context_call,
};

const device_finalization_pipeline = [_]TirPass{
    .lower_warp_memory,
    .simplify,
    .lower_custom_datatypes,
    .lower_device_storage_access_info,
    .lower_intrin,
};

const FilterKind = enum { host, device };

/// Filter an IRModule to keep only host or device functions.
///
/// Classification follows TVM's build driver: `llvm` and `c` targets are host
///  targets. A function without a target uses the default `llvm` host target.
fn filter_module(allocator: std.mem.Allocator, ir_mod: IRModule, kind: FilterKind) !IRModule {
    const callback_fn: ffi.Callback = switch (kind) {
        .host => &filter_host_callback,
        .device => &filter_device_callback,
    };

    var filter_func = try ffi.PackedFunction.create(null, callback_fn, null);
    defer filter_func.deinit();

    const pass = TirPass{ .filter = .{ .predicate = filter_func.as_value() } };
    var pass_value = try pass.create(allocator);
    defer pass_value.deinit();

    var result = try ffi.call_global(allocator, "transform.RunPass", &.{
        pass_value.borrow(),
        ir_mod.as_value(),
    });
    errdefer result.deinit();
    return try IRModule.take(&result);
}

fn filter_host_callback(
    handle: ?*anyopaque,
    args: *const ffi.CallbackArgs,
) !ffi.CallbackOutput {
    return filter_by_target(handle, args, true);
}

fn filter_device_callback(
    handle: ?*anyopaque,
    args: *const ffi.CallbackArgs,
) !ffi.CallbackOutput {
    return filter_by_target(handle, args, false);
}

fn filter_by_target(
    handle: ?*anyopaque,
    args: *const ffi.CallbackArgs,
    want_host: bool,
) !ffi.CallbackOutput {
    _ = handle;
    if (args.len() != 1) {
        log.err("filter callback: expected 1 arg, got {d}", .{args.len()});
        return error.InvalidCallbackArity;
    }

    var arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
    defer arena.deinit();
    const is_host = function_is_host(
        arena.allocator(),
        args.get(0),
    ) catch |err| {
        log.err("failed to classify a lowered TVM function: {s}", .{@errorName(err)});
        return err;
    };
    return .{ .boolean = is_host == want_host };
}

fn function_is_host(allocator: std.mem.Allocator, func: Value) !bool {
    var attrs = (try tir.get_func_attrs(allocator, func)) orelse return true;
    defer attrs.deinit();

    var target_key = try ffi.OwnedValue.from_string("target");
    defer target_key.deinit();
    if (!try attrs.contains(allocator, target_key.borrow())) return true;

    var target = try attrs.get(allocator, target_key.borrow());
    defer target.deinit();
    var exported_value = try ffi.call_global(
        allocator,
        "target.TargetExport",
        &.{target.borrow()},
    );
    defer exported_value.deinit();
    var exported = try container.Map.retain(exported_value.borrow());
    defer exported.deinit();

    var kind_key = try ffi.OwnedValue.from_string("kind");
    defer kind_key.deinit();
    var kind_value = try exported.get(allocator, kind_key.borrow());
    defer kind_value.deinit();
    const kind = try kind_value.borrow().dupe_string(allocator);
    defer allocator.free(kind);
    return std.mem.eql(u8, kind, "llvm") or std.mem.eql(u8, kind, "c");
}
