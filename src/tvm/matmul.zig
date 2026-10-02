//! TVM matmul tuning and execution.

const std = @import("std");

const device_mod = @import("../device.zig");
const DType = @import("../dtype.zig").DType;
const dlpack = @import("../c/dlpack.zig");
const ffi = @import("../c/tvm/ffi.zig");
const tvm_compile = @import("../c/tvm/compile.zig");
const tvm_runtime = @import("../c/tvm/runtime.zig");
const ms = @import("../c/tvm/meta_schedule.zig");
const tir = @import("../c/tvm/tir.zig");
const te = @import("../c/tvm/te.zig");
const Cache = @import("../cache.zig").Cache;
const artifact = @import("artifact.zig");
const config = @import("config.zig");
pub const TargetKind = config.TargetKind;
const integration_runtime = @import("runtime.zig");
const tune_mod = @import("tune.zig");

const log = std.log.scoped(.@"zg/tvm_matmul");

/// Matrix-multiplication signature supported by the TVM provider.
pub const Shape = struct {
    /// Left-hand matrix row count.
    m: i64,
    /// Right-hand matrix column count.
    n: i64,
    /// Contracting dimension.
    k: i64,
    /// Shared input and output element type.
    dtype: DType = .f32,
};

/// Options for tuning one matrix multiplication shape.
pub const TuneOptions = struct {
    /// Target-specific compiler inputs resolved by application composition.
    compile: config.CompileConfig,

    /// Device used for target detection and candidate measurement.
    device: device_mod.Device,

    /// Delete existing work artifacts before tuning.
    retune: bool = false,

    /// Maximum measured candidates.
    max_trials: u32 = 64,

    /// Candidates submitted per tuning iteration.
    trials_per_iter: u32 = 16,

    /// Repetition and cache policy for TVM schedule measurements.
    measurement: tvm_runtime.TimeEvaluatorOptions = .{},
};

/// Selected result from one tuning run.
pub const TuneResult = struct {
    /// Mean of the selected candidate's evaluator results.
    best_time_us: f64,
    /// Evaluator repeats for the selected candidate.
    sample_count: usize,
};

/// Tune one matrix multiplication and publish the selected artifact.
pub fn tune(
    /// I/O context used for cache files and candidate measurements.
    io: std.Io,
    /// Allocator used by TVM calls and temporary tuning state.
    allocator: std.mem.Allocator,
    /// Root for artifacts and tuning records.
    artifact_cache: *const Cache,
    /// Matrix dimensions and element type.
    shape: Shape,
    /// Compiler inputs, device, and tuning limits.
    options: TuneOptions,
) !TuneResult {
    try integration_runtime.ensure_loaded(.compiler);

    const target = options.compile.target;
    if (!target.accepts(options.device)) return error.TargetDeviceMismatch;
    var tvm_target = try tir.Target.resolve(
        allocator,
        target,
        options.device.ordinal,
    );
    defer tvm_target.deinit();
    const location = try cache_location(
        io,
        artifact_cache,
        shape,
        options.compile.compiler_fingerprint,
        tvm_target.description,
        true,
    );
    var tuning_lock = try acquire_tuning_lock(io, &location.lock);
    defer tuning_lock.deinit();

    if (options.retune) {
        try reset_tuning_records(io, &location.work);
    }

    var ir_module = try build_ir_module(allocator, shape);
    defer ir_module.deinit();
    const shape_a = [_]i64{ shape.m, shape.k };
    const shape_b = [_]i64{ shape.k, shape.n };
    const shape_c = [_]i64{ shape.m, shape.n };
    const shapes = [_][]const i64{ &shape_a, &shape_b, &shape_c };

    var tuned = try tune_mod.tune(
        io,
        allocator,
        ir_module,
        tvm_target.target,
        shape.dtype,
        &shapes,
        .{
            .compile = options.compile,
            .device = options.device,
            .gpu_arch = tvm_target.gpu_arch,
            .work_cache = &location.work,
            .max_trials = options.max_trials,
            .trials_per_iter = options.trials_per_iter,
            .measurement = options.measurement,
        },
    );
    defer tuned.deinit();

    var compiled = try tvm_compile.lower_and_compile(
        allocator,
        &tuned.module,
        tvm_target.target,
        target,
    );
    defer compiled.deinit();
    try artifact.publish(
        compiled,
        io,
        allocator,
        &location.work,
        target,
        options.compile.linker,
    );
    return .{
        .best_time_us = tuned.mean_time_seconds * 1e6,
        .sample_count = tuned.sample_count,
    };
}

/// Bytes and cache key for one published matmul artifact.
///
/// The caller frees `bytes` with the allocator passed to `load_artifact`.
pub const CachedArtifact = struct {
    /// Cache key derived from the compiler fingerprint, target, and shape.
    key: artifact.CacheKey,
    /// Shared-library bytes. Free with the allocator passed to `load_artifact`.
    bytes: []u8,
};

/// Read a published matmul artifact.
pub fn load_artifact(
    /// I/O context used to read the artifact.
    io: std.Io,
    /// Allocator used for the returned artifact bytes.
    allocator: std.mem.Allocator,
    /// Root containing TVM artifacts.
    artifact_cache: *const Cache,
    /// Matrix dimensions and element type.
    shape: Shape,
    /// Cache fingerprint for compiler inputs that affect generated artifacts.
    compiler_fingerprint: []const u8,
    /// Compilation target.
    target: TargetKind,
    /// Device used to complete the target description.
    device: device_mod.Device,
) !?CachedArtifact {
    try integration_runtime.ensure_loaded(.compiler);
    if (!target.accepts(device)) return error.TargetDeviceMismatch;
    var target_description = try tir.Target.describe(
        allocator,
        target,
        device.ordinal,
    );
    defer target_description.deinit();
    const location = try cache_location(
        io,
        artifact_cache,
        shape,
        compiler_fingerprint,
        target_description.description,
        false,
    );
    const bytes = try artifact.read_cached_bytes(
        io,
        allocator,
        &location.work,
    ) orelse return null;
    return .{ .key = location.key, .bytes = bytes };
}

const CachedMatmulState = struct {
    allocator: std.mem.Allocator,
    target: TargetKind,
    device: device_mod.Device,
    shape: Shape,
    artifact: artifact.LoadedArtifact,
};

/// Loaded matmul artifact and execution state.
pub const CachedMatmul = opaque {
    /// Load the published implementation for one shape and target.
    pub fn load(
        /// I/O context used to read the artifact.
        io: std.Io,
        /// Allocator used until `deinit`.
        allocator: std.mem.Allocator,
        /// Root containing TVM artifacts.
        artifact_cache: *const Cache,
        /// Matrix dimensions and element type.
        shape: Shape,
        /// Cache fingerprint for compiler inputs that affect generated artifacts.
        compiler_fingerprint: []const u8,
        /// Compilation target.
        target: TargetKind,
        /// Device used to complete the target description and execute.
        device: device_mod.Device,
    ) !?*CachedMatmul {
        try integration_runtime.ensure_loaded(.runtime);

        if (!target.accepts(device)) return error.TargetDeviceMismatch;
        var target_description = try tir.Target.describe(
            allocator,
            target,
            device.ordinal,
        );
        defer target_description.deinit();
        const location = try cache_location(
            io,
            artifact_cache,
            shape,
            compiler_fingerprint,
            target_description.description,
            false,
        );
        var loaded = try artifact.load_cached(
            io,
            allocator,
            &location.work,
        ) orelse return null;
        errdefer loaded.deinit();

        const state = try allocator.create(CachedMatmulState);
        state.* = .{
            .allocator = allocator,
            .target = target,
            .device = device,
            .shape = shape,
            .artifact = loaded,
        };
        return @ptrCast(state);
    }

    /// Release the loaded module and wrapper state.
    pub fn deinit(self: *CachedMatmul) void {
        const state = state_from(self);
        const allocator = state.allocator;
        state.artifact.deinit();
        allocator.destroy(state);
    }

    /// Execute the loaded f32 matrix multiplication.
    pub fn execute(
        self: *CachedMatmul,
        lhs: []const f32,
        rhs: []const f32,
        output: []f32,
    ) !void {
        const state = state_from(self);
        const shape = state.shape;
        if (shape.dtype != .f32) return error.UnsupportedDType;
        const counts = try element_counts(shape);
        if (lhs.len != counts.lhs or
            rhs.len != counts.rhs or
            output.len != counts.output)
        {
            return error.InvalidShape;
        }

        var shape_lhs = [_]i64{ shape.m, shape.k };
        var shape_rhs = [_]i64{ shape.k, shape.n };
        var shape_output = [_]i64{ shape.m, shape.n };

        const allocator = state.allocator;
        switch (state.target) {
            .cpu => {
                var dl_lhs = dlpack.ManagedTensor.borrowing(
                    dlpack.Tensor.init_contiguous(f32, @constCast(lhs), &shape_lhs),
                );
                var dl_rhs = dlpack.ManagedTensor.borrowing(
                    dlpack.Tensor.init_contiguous(f32, @constCast(rhs), &shape_rhs),
                );
                var dl_output = dlpack.ManagedTensor.borrowing(
                    dlpack.Tensor.init_contiguous(f32, output, &shape_output),
                );

                var tvm_lhs = try tvm_runtime.Tensor.from_dlpack(&dl_lhs);
                defer tvm_lhs.deinit();
                var tvm_rhs = try tvm_runtime.Tensor.from_dlpack(&dl_rhs);
                defer tvm_rhs.deinit();
                var tvm_output = try tvm_runtime.Tensor.from_dlpack(&dl_output);
                defer tvm_output.deinit();

                try state.artifact.invoke(allocator, &.{
                    tvm_lhs,
                    tvm_rhs,
                    tvm_output,
                });
            },
            .cuda => {
                var tvm_lhs = try tvm_runtime.Tensor.allocate(
                    allocator,
                    std.mem.sliceAsBytes(lhs),
                    &shape_lhs,
                    .f32_,
                    .cuda,
                    state.device.ordinal,
                );
                defer tvm_lhs.deinit();
                var tvm_rhs = try tvm_runtime.Tensor.allocate(
                    allocator,
                    std.mem.sliceAsBytes(rhs),
                    &shape_rhs,
                    .f32_,
                    .cuda,
                    state.device.ordinal,
                );
                defer tvm_rhs.deinit();

                const output_init = try allocator.alloc(f32, output.len);
                defer allocator.free(output_init);
                @memset(output_init, 0);
                var tvm_output = try tvm_runtime.Tensor.allocate(
                    allocator,
                    std.mem.sliceAsBytes(output_init),
                    &shape_output,
                    .f32_,
                    .cuda,
                    state.device.ordinal,
                );
                defer tvm_output.deinit();

                try state.artifact.invoke(allocator, &.{
                    tvm_lhs,
                    tvm_rhs,
                    tvm_output,
                });
                try tvm_output.copy_to_host(allocator, std.mem.sliceAsBytes(output));
            },
        }
    }

    fn state_from(self: *CachedMatmul) *CachedMatmulState {
        return @ptrCast(@alignCast(self));
    }
};

const ElementCounts = struct {
    lhs: usize,
    rhs: usize,
    output: usize,
};

const CacheLocation = struct {
    work: Cache,
    lock: Cache,
    key: artifact.CacheKey,
};

const TuningLock = struct {
    io: std.Io,
    file: std.Io.File,

    fn deinit(self: *TuningLock) void {
        self.file.close(self.io);
        self.* = undefined;
    }
};

fn cache_location(
    io: std.Io,
    artifact_cache: *const Cache,
    shape: Shape,
    compiler_fingerprint: []const u8,
    target_description: []const u8,
    create: bool,
) !CacheLocation {
    try validate_shape(shape);
    const base = try artifact_cache.subdir(io, "tvm", .{ .create = create });
    const key = artifact.matmul_cache_key(
        compiler_fingerprint,
        target_description,
        shape.dtype,
        shape.m,
        shape.n,
        shape.k,
    );
    var lock_name_buffer: [80]u8 = undefined;
    const lock_name = std.fmt.bufPrint(&lock_name_buffer, "{s}.lock", .{key.slice()}) catch
        unreachable;
    return .{
        .work = try base.subdir(io, key.slice(), .{ .create = create }),
        .lock = try base.join(lock_name),
        .key = key,
    };
}

fn acquire_tuning_lock(io: std.Io, path: *const Cache) !TuningLock {
    const file = std.Io.Dir.cwd().createFile(io, path.path(), .{
        .truncate = false,
        .lock = .exclusive,
        .lock_nonblocking = true,
    }) catch |err| switch (err) {
        error.WouldBlock => return error.TuningInProgress,
        else => return err,
    };
    return .{ .io = io, .file = file };
}

fn reset_tuning_records(io: std.Io, work_cache: *const Cache) !void {
    const cwd = std.Io.Dir.cwd();
    for ([_][]const u8{ "workload.json", "tuning_record.json" }) |name| {
        const path = try work_cache.join(name);
        cwd.deleteFile(io, path.path()) catch |err| switch (err) {
            error.FileNotFound => {},
            else => return err,
        };
    }
}

fn validate_shape(shape: Shape) error{InvalidShape}!void {
    if (shape.m <= 0 or shape.n <= 0 or shape.k <= 0)
        return error.InvalidShape;
}

fn element_counts(shape: Shape) error{InvalidShape}!ElementCounts {
    try validate_shape(shape);
    const m_size = std.math.cast(usize, shape.m) orelse return error.InvalidShape;
    const n_size = std.math.cast(usize, shape.n) orelse return error.InvalidShape;
    const k_size = std.math.cast(usize, shape.k) orelse return error.InvalidShape;
    return .{
        .lhs = std.math.mul(usize, m_size, k_size) catch return error.InvalidShape,
        .rhs = std.math.mul(usize, k_size, n_size) catch return error.InvalidShape,
        .output = std.math.mul(usize, m_size, n_size) catch return error.InvalidShape,
    };
}

fn build_ir_module(allocator: std.mem.Allocator, shape: Shape) !tir.IRModule {
    try validate_shape(shape);
    const lhs_shape = [_]i64{ shape.m, shape.k };
    const rhs_shape = [_]i64{ shape.k, shape.n };

    var lhs = try te.placeholder(allocator, &lhs_shape, shape.dtype, "A");
    defer lhs.deinit();
    var rhs = try te.placeholder(allocator, &rhs_shape, shape.dtype, "B");
    defer rhs.deinit();
    var output = try te.matmul(allocator, &lhs, &rhs, .{});
    defer output.deinit();
    var function = try te.create_prim_func(allocator, &.{ &lhs, &rhs, &output });
    defer function.deinit();
    return try tir.IRModule.from_entry(allocator, "main", &function);
}

test "TVM compiles a CPU matmul TIR module" {
    try integration_runtime.configure(.{ .surface = .compiler });
    try integration_runtime.ensure_loaded(.compiler);

    var ir_module = try build_ir_module(
        std.testing.allocator,
        .{ .m = 2, .n = 3, .k = 4, .dtype = .f32 },
    );
    defer ir_module.deinit();
    var target = try tir.Target.create(std.testing.allocator, .cpu, 0);
    defer target.deinit();
    var compiled = try tvm_compile.lower_and_compile(
        std.testing.allocator,
        &ir_module,
        target,
        .cpu,
    );
    defer compiled.deinit();

    var main = try compiled.get_function(std.testing.allocator, "main", false);
    defer main.deinit();
}

test "TVM generates an sm80 f16 matmul design space" {
    const cuda_intrinsics = @import("cuda_intrinsics.zig");
    const logger_callback = struct {
        fn call(
            _: ?*anyopaque,
            _: *const ffi.CallbackArgs,
        ) !ffi.CallbackOutput {
            return .none;
        }
    }.call;

    const intrinsic_path = std.process.Environ.getAlloc(
        std.testing.environ,
        std.testing.allocator,
        config.cuda_intrinsics_path_env,
    ) catch |err| switch (err) {
        error.EnvironmentVariableMissing => return error.SkipZigTest,
        else => return err,
    };
    defer std.testing.allocator.free(intrinsic_path);
    try integration_runtime.configure(.{ .surface = .compiler });
    try integration_runtime.ensure_loaded(.compiler);
    try cuda_intrinsics.ensure_registered(
        std.testing.io,
        std.testing.allocator,
        intrinsic_path,
    );

    var ir_module = try build_ir_module(
        std.testing.allocator,
        .{ .m = 128, .n = 128, .k = 128, .dtype = .f16 },
    );
    defer ir_module.deinit();
    var target = try tir.Target.from_description(
        std.testing.allocator,
        "cuda -arch=sm_80 -max_shared_memory_per_block=49152 " ++
            "-max_threads_per_block=1024 -thread_warp_size=32 " ++
            "-registers_per_block=65536 -l2_cache_size_bytes=41943040",
    );
    defer target.deinit();
    var space_generator = try ms.SpaceGenerator.post_order_apply(std.testing.allocator);
    defer space_generator.deinit();
    var search_strategy = try ms.SearchStrategy.replay_trace(std.testing.allocator, .{});
    defer search_strategy.deinit();
    var logger = try ffi.PackedFunction.create(null, logger_callback, null);
    defer logger.deinit();
    var context = try ms.TuneContext.init(std.testing.allocator, .{
        .module = ir_module,
        .target = target,
        .generator = &space_generator,
        .strategy = &search_strategy,
        .task_name = "main",
        .logger = &logger,
    });
    defer context.deinit();

    var schedules = try space_generator.generate(
        std.testing.allocator,
        ir_module,
    );
    defer schedules.deinit();
    try std.testing.expect(try schedules.len(std.testing.allocator) > 0);
    try std.testing.expect(try design_space_uses_tensorization(
        std.testing.allocator,
        schedules,
    ));
}

fn design_space_uses_tensorization(
    allocator: std.mem.Allocator,
    schedules: ms.Schedules,
) !bool {
    for (0..try schedules.len(allocator)) |schedule_index| {
        var schedule = try schedules.get(allocator, schedule_index);
        defer schedule.deinit();
        var trace = try schedule.python_trace(allocator);
        defer trace.deinit();
        for (trace.lines) |line| {
            if (std.mem.indexOf(u8, line, "meta_schedule.auto_tensorize") != null)
                return true;
        }
    }
    return false;
}

test element_counts {
    try std.testing.expectEqualDeep(
        ElementCounts{ .lhs = 6, .rhs = 6, .output = 4 },
        try element_counts(.{ .m = 2, .n = 2, .k = 3 }),
    );
    try std.testing.expectError(
        error.InvalidShape,
        element_counts(.{ .m = 0, .n = 2, .k = 3 }),
    );
}
