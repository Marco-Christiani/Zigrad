//! Main benchmark orchestration for matmul performance testing.
const std = @import("std");
const zg = @import("zigrad");
const gemm = @import("../gemm.zig");
const config = @import("config.zig");
const stats = @import("stats.zig");
const backend_adapter = @import("backend_adapter.zig");
const Cache = zg.Cache;

const BenchmarkConfig = config.BenchmarkConfig;
const DType = config.DType;
const Shape = config.Shape;
const Implementation = config.Implementation;
const BenchmarkResult = config.BenchmarkResult;

const log = std.log.scoped(.benchmark);

pub const DeviceKind = config.DeviceKind;

const syms = zg.utils.Symbols.unicode;

/// Benchmark harness for matmul implementations.
pub const Harness = struct {
    io: std.Io,
    environ: *const std.process.Environ.Map,
    allocator: std.mem.Allocator,
    config: BenchmarkConfig,
    results: std.ArrayList(BenchmarkResult),
    rng: std.Random.DefaultPrng,
    cache: Cache,

    // XLA/PJRT contexts initialize on first use.
    xla_cpu_ctx: ?backend_adapter.XlaContext = null,
    xla_gpu_ctx: ?backend_adapter.XlaContext = null,
    iree_ctx: ?backend_adapter.IreeContext = null,

    // Separate maps keep CPU and CUDA modules under different keys.
    tvm_cpu_cache: std.AutoHashMap(Shape, *zg.tvm.CachedMatmul),
    tvm_gpu_cache: std.AutoHashMap(Shape, *zg.tvm.CachedMatmul),

    pub fn init(io: std.Io, environ: *const std.process.Environ.Map, allocator: std.mem.Allocator, cfg: BenchmarkConfig) !Harness {
        return Harness{
            .io = io,
            .environ = environ,
            .allocator = allocator,
            .config = cfg,
            .results = try .initCapacity(allocator, 16),
            .rng = .init(cfg.seed),
            .cache = try .init(io, environ, .{}),
            .tvm_cpu_cache = .init(allocator),
            .tvm_gpu_cache = .init(allocator),
        };
    }

    pub fn deinit(self: *Harness) void {
        self.results.deinit(self.allocator);

        if (self.xla_cpu_ctx) |*ctx| ctx.deinit();
        if (self.xla_gpu_ctx) |*ctx| ctx.deinit();
        if (self.iree_ctx) |*ctx| ctx.deinit();

        deinit_tvm_cache(&self.tvm_cpu_cache);
        deinit_tvm_cache(&self.tvm_gpu_cache);
    }

    fn deinit_tvm_cache(cache: *std.AutoHashMap(Shape, *zg.tvm.CachedMatmul)) void {
        var iter = cache.iterator();
        while (iter.next()) |entry| {
            entry.value_ptr.*.deinit();
        }
        cache.deinit();
    }

    /// Run all benchmarks according to the configuration.
    pub fn run(self: *Harness) !void {
        log.info("starting benchmark suite: {d} shapes x {d} implementations ({t})", .{
            self.config.shapes.len,
            self.config.implementations.len,
            self.config.dtype,
        });

        for (self.config.shapes) |shape| {
            try self.run_shape(shape);
        }

        log.info("benchmark suite complete: {d} results collected", .{self.results.items.len});
    }

    pub const TuneOpts = struct {
        max_trials: u32 = 64,
        trials_per_iter: u32 = 16,
        /// Delete existing artifacts and tune from scratch.
        retune: bool = false,
    };

    /// Tune search-based kernels (TVM) for all configured shapes and implementations.
    pub fn tune(self: *Harness, opts: TuneOpts) !void {
        var need_tvm_cpu = false;
        var need_tvm_gpu = false;
        for (self.config.implementations) |impl| switch (impl) {
            .tvm_cpu => {
                need_tvm_cpu = true;
            },
            .tvm_gpu => {
                need_tvm_gpu = true;
            },
            else => {},
        };

        if (!need_tvm_cpu and !need_tvm_gpu) {
            log.info("no tunable implementations requested, nothing to tune", .{});
            return;
        }

        if (need_tvm_cpu) {
            for (self.config.shapes) |shape| {
                try self.tune_tvm_shape(shape, .cpu, opts);
            }
        }
        if (need_tvm_gpu) {
            for (self.config.shapes) |shape| {
                try self.tune_tvm_shape(shape, .cuda, opts);
            }
        }
    }

    fn tune_tvm_shape(self: *Harness, shape: Shape, target_kind: zg.tvm.TargetKind, opts: TuneOpts) !void {
        log.info("tuning: {any} ({t}), {d} trials", .{ shape, target_kind, opts.max_trials });
        const compile_config = try zg.tvm.CompileConfig.from_environ(
            self.environ,
            target_kind,
        );
        const result = try zg.tvm.tune_matmul(
            self.io,
            self.allocator,
            &self.cache,
            try self.tvm_shape(shape),
            .{
                .compile = compile_config,
                .device = .{
                    .platform = if (target_kind == .cuda) .cuda else .cpu,
                },
                .retune = opts.retune,
                .max_trials = opts.max_trials,
                .trials_per_iter = opts.trials_per_iter,
            },
        );

        log.info("tuned: {any} ({t}), {d:.2} us over {d} samples", .{
            shape, target_kind, result.best_time_us, result.sample_count,
        });
    }

    /// Dispatch to the inner function selected by the configured dtype.
    fn run_shape(self: *Harness, shape: Shape) !void {
        return switch (self.config.dtype) {
            inline else => |dtype| self.run_shape_typed(dtype.ZigType(), shape),
        };
    }

    fn run_shape_typed(self: *Harness, comptime T: type, shape: Shape) !void {
        log.info("benchmarking shape: {any} ({s})", .{ shape, @typeName(T) });

        const elems_a: usize = @intCast(shape.m * shape.k);
        const elems_b: usize = @intCast(shape.k * shape.n);
        const elems_c: usize = @intCast(shape.m * shape.n);

        const a = try self.allocator.alloc(T, elems_a);
        defer self.allocator.free(a);
        const b = try self.allocator.alloc(T, elems_b);
        defer self.allocator.free(b);
        const c = try self.allocator.alloc(T, elems_c);
        defer self.allocator.free(c);

        fill_random(T, &self.rng, a);
        fill_random(T, &self.rng, b);

        var reference: ?[]T = null;
        defer if (reference) |ref| self.allocator.free(ref);

        if (self.config.verify_correctness) {
            const ref = try self.allocator.alloc(T, elems_c);
            @memset(ref, @as(T, 0));
            gemm.gemm(T, shape.m, shape.n, shape.k, a, @intCast(shape.k), b, @intCast(shape.n), ref, @intCast(shape.n));
            reference = ref;
        }

        for (self.config.implementations) |impl| {
            const result = try self.benchmark_impl_typed(T, impl, shape, a, b, c, reference);
            try self.results.append(self.allocator, result);

            if (result.passed) {
                log.info("  {s}: {d:.2} GFLOP/s (median {d:.2} us, error {any})", .{
                    impl.display_name(),
                    result.gflops,
                    result.median_us,
                    result.max_abs_error,
                });
            } else {
                log.err("  {s}: FAILED correctness check (error {any} > tolerance {any})", .{
                    impl.display_name(),
                    result.max_abs_error,
                    self.config.dtype.tolerance(),
                });
            }
        }
    }

    fn benchmark_impl_typed(
        self: *Harness,
        comptime T: type,
        impl: Implementation,
        shape: Shape,
        a: []const T,
        b: []const T,
        c: []T,
        reference: ?[]const T,
    ) !BenchmarkResult {
        for (0..self.config.warmup_iters) |_| {
            @memset(c, @as(T, 0));
            try self.run_kernel(T, impl, shape, a, b, c);
        }

        var times = try self.allocator.alloc(f64, self.config.bench_iters);
        defer self.allocator.free(times);

        for (0..self.config.bench_iters) |i| {
            @memset(c, @as(T, 0));

            const start = std.Io.Timestamp.now(self.io, .awake);
            try self.run_kernel(T, impl, shape, a, b, c);
            const elapsed = start.untilNow(self.io, .awake);

            times[i] = @as(f64, @floatFromInt(elapsed.toNanoseconds())) / 1000.0;
        }

        const median_us = stats.median(times);
        const mean_us = stats.mean(times);
        const stddev_us = stats.stddev(times);

        const flops = 2.0 * @as(f64, @floatFromInt(shape.m)) *
            @as(f64, @floatFromInt(shape.n)) *
            @as(f64, @floatFromInt(shape.k));
        const gflops = flops / (median_us * 1000.0);

        var max_err: f64 = 0.0;
        var passed = true;
        if (reference) |ref| {
            max_err = max_abs_error(T, ref, c);
            passed = (max_err <= self.config.dtype.tolerance());
        }

        return BenchmarkResult{
            .impl = impl,
            .shape = shape,
            .dtype = self.config.dtype,
            .median_us = median_us,
            .mean_us = mean_us,
            .stddev_us = stddev_us,
            .gflops = gflops,
            .max_abs_error = max_err,
            .passed = passed,
        };
    }

    /// Execute a single kernel invocation.
    fn run_kernel(
        self: *Harness,
        comptime T: type,
        impl: Implementation,
        shape: Shape,
        a: []const T,
        b: []const T,
        c: []T,
    ) !void {
        const lda: usize = @intCast(shape.k);
        const ldb: usize = @intCast(shape.n);
        const ldc: usize = @intCast(shape.n);
        switch (impl) {
            .blas => try gemm.blas_gemm(T, shape.m, shape.n, shape.k, a, lda, b, ldb, c, ldc),
            .zig_naive => gemm.gemm(T, shape.m, shape.n, shape.k, a, lda, b, ldb, c, ldc),
            .tvm_cpu => try self.run_tvm(T, shape, a, b, c, .cpu),
            .tvm_gpu => try self.run_tvm(T, shape, a, b, c, .gpu),
            .iree => if (comptime zg.build_options.has_iree and zg.build_options.has_mlir)
                try self.run_iree(T, shape, a, b, c)
            else
                return error.IreeDisabled,
            .xla_cpu => if (comptime zg.build_options.has_mlir)
                try self.run_xla(T, shape, a, b, c, .cpu)
            else
                return error.MlirDisabled,
            .xla_gpu => if (comptime zg.build_options.has_mlir)
                try self.run_xla(T, shape, a, b, c, .gpu)
            else
                return error.MlirDisabled,
        }
    }

    /// Execute the published TVM generation for this shape.
    fn run_tvm(
        self: *Harness,
        comptime T: type,
        shape: Shape,
        a: []const T,
        b: []const T,
        c: []T,
        device: DeviceKind,
    ) !void {
        const element = comptime zg.tvm.ElementType.from_dtype(DType.from_zig_type(T).to_pr_dtype()) orelse
            return error.UnsupportedDtype;

        const mem_cache = switch (device) {
            .cpu => &self.tvm_cpu_cache,
            .gpu => &self.tvm_gpu_cache,
        };

        const tuned = if (mem_cache.get(shape)) |module|
            module
        else blk: {
            const target_kind: zg.tvm.TargetKind = switch (device) {
                .cpu => .cpu,
                .gpu => .cuda,
            };
            const build_identity = try zg.tvm.CompileConfig.resolve_build_identity(
                self.environ,
            );
            const module = try zg.tvm.CachedMatmul.load(
                self.io,
                self.allocator,
                &self.cache,
                try self.tvm_shape(shape),
                build_identity,
                target_kind,
                .{
                    .platform = if (target_kind == .cuda) .cuda else .cpu,
                },
            ) orelse
                return error.NoTuningRecords;

            errdefer module.deinit();
            try mem_cache.put(shape, module);
            break :blk module;
        };

        try tuned.execute(
            element,
            a,
            b,
            c,
        );
    }

    fn tvm_shape(self: *const Harness, shape: Shape) !zg.tvm.MatmulShape {
        return .{ .m = shape.m, .n = shape.n, .k = shape.k, .dtype = zg.tvm.ElementType.from_dtype(self.config.dtype.to_pr_dtype()) orelse return error.UnsupportedDtype };
    }

    /// Execute XLA/PJRT matmul (compiles on-the-fly, caches per shape).
    fn run_xla(
        self: *Harness,
        comptime T: type,
        shape: Shape,
        a: []const T,
        b: []const T,
        c: []T,
        device: DeviceKind,
    ) !void {
        const ctx = switch (device) {
            .cpu => &self.xla_cpu_ctx,
            .gpu => &self.xla_gpu_ctx,
        };

        if (ctx.* == null) {
            ctx.* = try backend_adapter.XlaContext.init(
                self.io,
                self.environ,
                self.allocator,
                device,
            );
        }

        try ctx.*.?.execute(T, shape, a, b, c);
    }

    fn run_iree(
        self: *Harness,
        comptime T: type,
        shape: Shape,
        a: []const T,
        b: []const T,
        c: []T,
    ) !void {
        if (self.iree_ctx == null) {
            self.iree_ctx = try backend_adapter.IreeContext.init(
                self.io,
                self.environ,
                self.allocator,
            );
        }
        try self.iree_ctx.?.execute(T, shape, a, b, c);
    }

    /// Print benchmark results in a human-readable table format.
    pub fn print_results(self: *const Harness) !void {
        var buffer: [8192]u8 = undefined;
        var stdout_writer = std.Io.File.stdout().writer(self.io, &buffer);
        const writer = &stdout_writer.interface;

        const header_sep = "\n" ++ "=" ** 60 ++ "\n";
        try writer.print("{s}Matmul Benchmark Results ({t}){s}", .{ header_sep, self.config.dtype, header_sep });

        var seen_shapes = std.AutoHashMap(Shape, void).init(self.allocator);
        defer seen_shapes.deinit();

        for (self.config.shapes) |shape| {
            if (seen_shapes.contains(shape)) continue;
            try seen_shapes.put(shape, {});

            try writer.print("\nShape: {any}\n\n", .{shape});
            try writer.writeAll("Implementation            Median (us)    GFLOP/s    vs Naive\n");
            try writer.writeAll("---------------------------------------------------------\n");

            var naive_gflops: f64 = 1.0;
            for (self.results.items) |result| {
                if (result.shape.m == shape.m and result.shape.n == shape.n and result.shape.k == shape.k and
                    result.impl == .zig_naive)
                {
                    naive_gflops = result.gflops;
                    break;
                }
            }

            for (self.results.items) |result| {
                if (result.shape.m != shape.m or result.shape.n != shape.n or result.shape.k != shape.k) continue;

                const speedup = result.gflops / naive_gflops;
                const status = if (result.passed) syms.check else syms.x;

                try writer.print("{s} {s:<23} {d:>10.2} {d:>10.2} {d:>8.2}{s}\n", .{
                    status,
                    result.impl.display_name(),
                    result.median_us,
                    result.gflops,
                    speedup,
                    syms.mul,
                });
            }
        }

        // Summary
        try writer.writeAll(header_sep ++ "Summary" ++ header_sep);

        if (self.results.items.len > 0) {
            var best_overall: ?BenchmarkResult = null;
            var best_hand_rolled: ?BenchmarkResult = null;

            for (self.results.items) |result| {
                if (!result.passed) continue;

                if (best_overall == null or result.gflops > best_overall.?.gflops) {
                    best_overall = result;
                }

                if (result.impl == .zig_naive and
                    (best_hand_rolled == null or result.gflops > best_hand_rolled.?.gflops))
                {
                    best_hand_rolled = result;
                }
            }

            if (best_overall) |best| {
                try writer.print("Best overall: {s} ({d:.2} GFLOP/s)\n", .{
                    best.impl.display_name(),
                    best.gflops,
                });
            }

            if (best_hand_rolled) |best| {
                try writer.print("Best hand-rolled: {s} ({d:.2} GFLOP/s)\n", .{
                    best.impl.display_name(),
                    best.gflops,
                });
            }
        }

        try writer.writeAll("\n");
        try writer.flush();
    }
};

fn fill_random(comptime T: type, rng: *std.Random.DefaultPrng, slice: []T) void {
    for (slice) |*val| {
        val.* = @floatCast(rng.random().float(f32) * 2.0 - 1.0);
    }
}

fn max_abs_error(comptime T: type, expected: []const T, actual: []const T) f64 {
    std.debug.assert(expected.len == actual.len);
    var result: f64 = 0.0;
    for (expected, actual) |e, a| {
        const err = @abs(@as(f64, @floatCast(e)) - @as(f64, @floatCast(a)));
        result = @max(result, err);
    }
    return result;
}

test "harness: init and deinit" {
    const allocator = std.testing.allocator;
    var environ: std.process.Environ.Map = .init(allocator);
    defer environ.deinit();

    const cfg = BenchmarkConfig{
        .shapes = &[_]Shape{.{ .m = 4, .n = 4, .k = 4 }},
        .implementations = &[_]Implementation{.zig_naive},
        .bench_iters = 5,
    };

    var harness = try Harness.init(std.testing.io, &environ, allocator, cfg);
    defer harness.deinit();

    try std.testing.expect(harness.results.items.len == 0);
}
