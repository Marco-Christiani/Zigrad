//! TVM kernel provider.
//!
//! The provider accepts PR matrix-multiply functions and emits kernel artifacts
//!  through MetaSchedule autotuning. TVM C types remain internal.
const std = @import("std");
const contraction = @import("../pr/analysis/contraction.zig");
const pattern = @import("../pr/analysis/pattern.zig");

const Cache = @import("../cache.zig").Cache;
const device_mod = @import("../device.zig");
const kernel = @import("../kernel.zig");
const pr = @import("../pr/pr.zig");
const TypedPtr = @import("../utils/rtti.zig").TypedPtr;
const config = @import("config.zig");
const mm = @import("matmul.zig");
const tvm_runtime = @import("runtime.zig");
const TvmDispatchState = @import("dispatch.zig").TvmDispatchState;

const log = std.log.scoped(.@"zg/tvm_provider");

pub const TvmProvider = struct {
    /// I/O context used by cache and tuning operations.
    io: std.Io,
    /// Target compiler and linker configuration.
    compile_config: config.CompileConfig,
    /// Artifact and tuning cache.
    cache: Cache,
    /// Maximum candidates measured for one cache miss.
    max_trials: u32 = 64,
    /// Candidates submitted per tuning iteration.
    trials_per_iter: u32 = 16,

    /// Dispatch state owning the TVM module cache.
    ///
    /// The state must outlive runtime capabilities returned by this provider.
    dispatch_state: *TvmDispatchState,

    /// Inputs required to initialize a TVM provider.
    pub const InitOptions = struct {
        /// Target and compiler inputs resolved by application composition.
        compile: config.CompileConfig,

        /// Maximum measured candidates.
        max_trials: u32 = 64,

        /// Candidates submitted per tuning iteration.
        trials_per_iter: u32 = 16,
    };

    /// Initialize a provider after loading its configured TVM runtime.
    pub fn init(
        /// I/O context used by cache and tuning operations.
        io: std.Io,
        /// Artifact and tuning cache.
        cache: Cache,
        /// Runtime state shared by compiled artifacts.
        dispatch_state: *TvmDispatchState,
        /// Compiler target and tuning limits.
        options: InitOptions,
    ) tvm_runtime.Error!TvmProvider {
        try tvm_runtime.ensure_loaded(.compiler);
        return .{
            .io = io,
            .compile_config = options.compile,
            .cache = cache,
            .max_trials = options.max_trials,
            .trials_per_iter = options.trials_per_iter,
            .dispatch_state = dispatch_state,
        };
    }

    /// Return a `KernelProvider` backed by this provider state.
    pub fn kernel_provider(self: *TvmProvider) kernel.KernelProvider {
        return .{
            .name = "tvm",
            .compiler = .{
                .context = @ptrCast(self),
                .vtable = &.{
                    .compile = compile_impl,
                    .discover = discover_impl,
                },
            },
            .runtime = .{
                .context = TypedPtr.init(self.dispatch_state),
                .vtable = &.{
                    .dispatch = &TvmDispatchState.dispatch,
                    .prepare = &TvmDispatchState.prepare,
                },
            },
        };
    }

    fn discover_impl(
        _: *anyopaque,
        func: pr.Function,
        matches: *std.ArrayList(kernel.Match),
        allocator: std.mem.Allocator,
    ) kernel.DiscoverError!void {
        for (func.ops, 0..) |op, op_index| {
            if (validate_matmul_op(op) == null) continue;
            try matches.append(allocator, .{ .start = op_index, .end = op_index + 1 });
        }
    }

    fn compile_impl(ptr: *anyopaque, func: pr.Function, device: device_mod.Device, allocator: std.mem.Allocator) kernel.CompileError!kernel.Artifact {
        const self: *TvmProvider = @ptrCast(@alignCast(ptr));
        return try self.compile(func, device, allocator);
    }

    /// Compile one supported matrix-multiply function into a kernel artifact.
    ///
    /// A cached artifact is reused when present. A cache miss runs the TVM
    ///  matmul tuner and stores the selected artifact.
    fn compile(
        self: *TvmProvider,
        func: pr.Function,
        device: device_mod.Device,
        allocator: std.mem.Allocator,
    ) kernel.CompileError!kernel.Artifact {
        if (!self.compile_config.target.accepts(device)) {
            return error.Unsupported;
        }
        const mm_shape = validate_matmul_function(func) orelse return error.Unsupported;

        log.info("compiling matmul kernel: {s} ({d}x{d}x{d})", .{
            func.name, mm_shape.m, mm_shape.n, mm_shape.k,
        });

        const cached = try self.load_artifact(allocator, mm_shape, device);
        if (cached) |artifact| return make_kernel_artifact(artifact);

        const result = try self.tune(allocator, mm_shape, device);

        const artifact = try self.load_artifact(
            allocator,
            mm_shape,
            device,
        ) orelse return error.CompileFailed;

        log.info("compiled kernel: {s} ({d:.2} us over {d} samples, {d} bytes)", .{
            func.name, result.best_time_us, result.sample_count, artifact.bytes.len,
        });

        return make_kernel_artifact(artifact);
    }

    fn load_artifact(
        self: *TvmProvider,
        allocator: std.mem.Allocator,
        shape: mm.Shape,
        device: device_mod.Device,
    ) kernel.CompileError!?mm.CachedArtifact {
        return mm.load_artifact(
            self.io,
            allocator,
            &self.cache,
            shape,
            self.compile_config.compiler_fingerprint,
            self.compile_config.target,
            device,
        ) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => {
                log.err("failed to read the TVM artifact cache: {s}", .{@errorName(err)});
                return error.CompileFailed;
            },
        };
    }

    fn tune(
        self: *TvmProvider,
        allocator: std.mem.Allocator,
        shape: mm.Shape,
        device: device_mod.Device,
    ) kernel.CompileError!mm.TuneResult {
        return mm.tune(self.io, allocator, &self.cache, shape, .{
            .compile = self.compile_config,
            .device = device,
            .max_trials = self.max_trials,
            .trials_per_iter = self.trials_per_iter,
        }) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            error.TvmLoadFailed => {
                log.err("TVM runtime unavailable: {s}", .{@errorName(err)});
                return error.ProviderLoadFailed;
            },
            error.TvmCallFailed,
            error.TvmFrontendCallFailed,
            error.TvmInvalidErrorState,
            error.TvmFunctionNotFound,
            error.TvmValueAbsent,
            error.TvmTypeMismatch,
            error.TvmTypeInfoMissing,
            error.TvmFieldNotFound,
            error.TvmFieldUnreadable,
            => {
                log.err("TVM API call failed during tuning: {s}", .{@errorName(err)});
                return error.ProviderCallFailed;
            },
            else => {
                log.err("TVM tuning failed: {s}", .{@errorName(err)});
                return error.CompileFailed;
            },
        };
    }
};

/// Validate that a function describes a single rank-two matrix multiplication.
///
/// Returns the matrix dimensions, or null when the function is unsupported.
fn validate_matmul_function(func: pr.Function) ?mm.Shape {
    if (func.ops.len != 1) return null;
    if (func.params.len != 2 or func.returns.len != 1) return null;
    const op = func.ops[0];
    const shape = validate_matmul_op(op) orelse return null;
    if (op.inputs[0].value != func.params[0]) return null;
    if (op.inputs[1].value != func.params[1]) return null;
    if (op.outputs[0] != func.returns[0]) return null;
    return shape;
}

fn validate_matmul_op(op: *const pr.Op) ?mm.Shape {
    if (!(pattern.Operation{
        .input_count = 2,
        .output_count = 1,
        .first_output_rank = 2,
    }).matches(op)) return null;

    switch (op.params) {
        .mm => {},
        .dot_general => |dg| {
            if (!contraction.is_matrix_matmul(dg)) return null;
        },
        else => return null,
    }

    const a = op.inputs[0].value.as_tensor();
    const b = op.inputs[1].value.as_tensor();
    const c_tensor = op.outputs[0].as_tensor();

    if (a.shape.rank() != 2 or b.shape.rank() != 2 or c_tensor.shape.rank() != 2) return null;

    const m = a.shape.dims[0];
    const k = a.shape.dims[1];
    const n = b.shape.dims[1];

    if (b.shape.dims[0] != k) return null;
    if (c_tensor.shape.dims[0] != m or c_tensor.shape.dims[1] != n) return null;

    if (a.dtype != b.dtype or a.dtype != c_tensor.dtype) return null;
    switch (a.dtype) {
        .f16, .f32 => {},
        else => return null,
    }

    return .{ .m = m, .n = n, .k = k, .dtype = a.dtype };
}

fn make_kernel_artifact(artifact: mm.CachedArtifact) kernel.Artifact {
    return .{
        .data = artifact.bytes,
    };
}

test "TVM discovery recognizes matrix matmul" {
    const testing = std.testing;
    var program = pr.Program.init(testing.allocator);
    defer program.deinit();
    var builder = try pr.FunctionBuilder.init(&program, "main");
    defer builder.deinit();
    const lhs = try builder.param_tensor(.f32, &.{ 4, 8 });
    const rhs = try builder.param_tensor(.f32, &.{ 8, 2 });
    const output = try builder.mm(lhs, rhs);
    const func = try builder.finish(.{ .returns = &.{output} });

    var matches = std.ArrayList(kernel.Match).empty;
    defer matches.deinit(testing.allocator);
    try TvmProvider.discover_impl(undefined, func, &matches, testing.allocator);
    try testing.expectEqualSlices(
        kernel.Match,
        &.{pattern.Range{ .start = 0, .end = 1 }},
        matches.items,
    );
}

test "TVM discovery recognizes f16 matrix matmul" {
    const testing = std.testing;
    var program = pr.Program.init(testing.allocator);
    defer program.deinit();
    var builder = try pr.FunctionBuilder.init(&program, "main");
    defer builder.deinit();
    const lhs = try builder.param_tensor(.f16, &.{ 16, 16 });
    const rhs = try builder.param_tensor(.f16, &.{ 16, 16 });
    const output = try builder.mm(lhs, rhs);
    const func = try builder.finish(.{ .returns = &.{output} });

    var matches = std.ArrayList(kernel.Match).empty;
    defer matches.deinit(testing.allocator);
    try TvmProvider.discover_impl(undefined, func, &matches, testing.allocator);
    try testing.expectEqualSlices(
        kernel.Match,
        &.{pattern.Range{ .start = 0, .end = 1 }},
        matches.items,
    );
}

test "TVM compilation requires a complete matrix matmul function" {
    const testing = std.testing;

    {
        var program = pr.Program.init(testing.allocator);
        defer program.deinit();
        var builder = try pr.FunctionBuilder.init(&program, "wrong_return");
        defer builder.deinit();
        const lhs = try builder.param_tensor(.f32, &.{ 4, 4 });
        const rhs = try builder.param_tensor(.f32, &.{ 4, 4 });
        _ = try builder.mm(lhs, rhs);
        const func = try builder.finish(.{ .returns = &.{lhs} });
        try testing.expectEqual(null, validate_matmul_function(func));
    }

    {
        var program = pr.Program.init(testing.allocator);
        defer program.deinit();
        var builder = try pr.FunctionBuilder.init(&program, "wrong_parameter_order");
        defer builder.deinit();
        const lhs = try builder.param_tensor(.f32, &.{ 4, 4 });
        const rhs = try builder.param_tensor(.f32, &.{ 4, 4 });
        const output = try builder.mm(rhs, lhs);
        const func = try builder.finish(.{ .returns = &.{output} });
        try testing.expectEqual(null, validate_matmul_function(func));
    }
}
