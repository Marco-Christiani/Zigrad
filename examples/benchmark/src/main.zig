//! Matmul benchmark: compare kernel implementations across backends.
//!
//! Compares TVM and compiled backend GEMM
//!  against naive Zig and BLAS baselines.
//!
//! Usage:
//!   benchmark [run] [options]          Run benchmarks (default subcommand)
//!   benchmark tune [options]           Tune search-based kernels before benchmarking
//!
//! Options:
//!   --shapes=MxNxK[,MxNxK,...]   Matrix dimensions (default: 128x128x128)
//!   --impls=name[,name,...]       Implementations: zig_naive, blas, tvm_cpu,
//!                                 tvm_gpu, iree, xla_cpu, xla_gpu, all, all-cpu, all-gpu
//!                                 (default: zig_naive)
//!   --dtype=f16|f32               Element type (default: f32)
//!   --warmup=N                    Warmup iterations (default: 10)
//!   --iters=N                     Benchmark iterations (default: 100)
//!   --trials=N                    Max tuning trials per shape (tune only, default: 64)
//!   --trials-per-iter=N           Trials per iteration (tune only, default: 16)
//!   (cache dir controlled via ZG_CACHE_DIR env var, default: /tmp/zigrad-cache)
const std = @import("std");
const zg = @import("zigrad");
const gemm = @import("gemm.zig");

const benchmark = struct {
    const harness = @import("benchmark/harness.zig");
    const config = @import("benchmark/config.zig");

    const Harness = harness.Harness;
    const BenchmarkConfig = config.BenchmarkConfig;
    const DType = config.DType;
    const Shape = config.Shape;
    const Implementation = config.Implementation;
};

const log = std.log.scoped(.benchmark);

const Subcommand = enum { run, tune };

/// Positional iterator over `std.process.Init.minimal.args`.
const ArgvIterator = struct {
    argv: []const [:0]const u8,
    idx: usize = 0,
    fn next(self: *ArgvIterator) ?[:0]const u8 {
        if (self.idx >= self.argv.len) return null;
        const a = self.argv[self.idx];
        self.idx += 1;
        return a;
    }
};

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    const argv = try init.minimal.args.toSlice(gpa);
    defer gpa.free(argv);
    var args_iter: ArgvIterator = .{ .argv = argv, .idx = 1 };

    var shapes = std.ArrayList(benchmark.Shape).empty;
    defer shapes.deinit(gpa);
    var impls = std.ArrayList(benchmark.Implementation).empty;
    defer impls.deinit(gpa);
    var dtype: benchmark.DType = .f32;
    var warmup: usize = 10;
    var iters: usize = 100;
    var subcmd: Subcommand = .run;
    var max_trials: u32 = 64;
    var trials_per_iter: u32 = 16;
    var retune = false;

    while (args_iter.next()) |arg| {
        // Subcommand (bare word, no --)
        if (std.meta.stringToEnum(Subcommand, arg)) |cmd| {
            subcmd = cmd;
            continue;
        }
        if (std.mem.startsWith(u8, arg, "--shapes=")) {
            var shape_strs = std.mem.splitScalar(u8, arg["--shapes=".len..], ',');
            while (shape_strs.next()) |s| {
                try shapes.append(gpa, try parse_shape(s));
            }
            continue;
        }
        if (std.mem.startsWith(u8, arg, "--impls=")) {
            var impl_strs = std.mem.splitScalar(u8, arg["--impls=".len..], ',');
            while (impl_strs.next()) |s| {
                try append_impls(gpa, &impls, s);
            }
            continue;
        }
        if (std.mem.startsWith(u8, arg, "--dtype=")) {
            const val = arg["--dtype=".len..];
            dtype = std.meta.stringToEnum(benchmark.DType, val) orelse {
                log.err("unknown dtype: {s} (expected: f16, f32)", .{val});
                return error.InvalidArguments;
            };
            continue;
        }
        if (std.mem.startsWith(u8, arg, "--warmup=")) {
            warmup = try std.fmt.parseInt(usize, arg["--warmup=".len..], 10);
            continue;
        }
        if (std.mem.startsWith(u8, arg, "--iters=")) {
            iters = try std.fmt.parseInt(usize, arg["--iters=".len..], 10);
            continue;
        }
        if (std.mem.startsWith(u8, arg, "--trials=")) {
            max_trials = try std.fmt.parseInt(u32, arg["--trials=".len..], 10);
            continue;
        }
        if (std.mem.startsWith(u8, arg, "--trials-per-iter=")) {
            trials_per_iter = try std.fmt.parseInt(u32, arg["--trials-per-iter=".len..], 10);
            continue;
        }
        if (std.mem.eql(u8, arg, "--retune")) {
            retune = true;
            continue;
        }
        if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            print_usage();
            return;
        }
        log.err("unknown argument: {s}", .{arg});
        return error.InvalidArguments;
    }

    if (shapes.items.len == 0) try shapes.append(gpa, .{ .m = 128, .n = 128, .k = 128 });
    if (impls.items.len == 0) try impls.append(gpa, .zig_naive);

    if (comptime zg.build_options.has_tvm) {
        const surface: zg.tvm.runtime.Surface = switch (subcmd) {
            .run => .runtime,
            .tune => .compiler,
        };
        try zg.tvm.runtime.configure(.from_environ(init.environ_map, surface));
    }

    const cfg = benchmark.BenchmarkConfig{
        .shapes = shapes.items,
        .implementations = impls.items,
        .dtype = dtype,
        .warmup_iters = warmup,
        .bench_iters = iters,
    };

    var harness_inst = try benchmark.Harness.init(io, init.environ_map, gpa, cfg);
    defer harness_inst.deinit();

    switch (subcmd) {
        .tune => try harness_inst.tune(.{
            .max_trials = max_trials,
            .trials_per_iter = trials_per_iter,
            .retune = retune,
        }),
        .run => {
            try harness_inst.run();
            try harness_inst.print_results();
        },
    }
}

/// Append implementations matching a name or group shorthand.
fn append_impls(
    gpa: std.mem.Allocator,
    impls: *std.ArrayList(benchmark.Implementation),
    name: []const u8,
) !void {
    const impl_map = std.StaticStringMap(enum { all, all_cpu, all_gpu }).initComptime(.{
        .{ "all", .all },
        .{ "all-cpu", .all_cpu },
        .{ "all-gpu", .all_gpu },
    });
    if (impl_map.get(name)) |group| {
        for (std.meta.tags(benchmark.Implementation)) |e| {
            if (!implementation_available(e)) continue;
            const dominated = switch (group) {
                .all => true,
                .all_cpu => !e.is_gpu(),
                .all_gpu => e.is_gpu(),
            };
            if (dominated) try impls.append(gpa, e);
        }
    } else if (std.meta.stringToEnum(benchmark.Implementation, name)) |impl| {
        try impls.append(gpa, impl);
    } else {
        log.err("unknown implementation: {s}", .{name});
        return error.InvalidArguments;
    }
}

fn implementation_available(impl: benchmark.Implementation) bool {
    return switch (impl) {
        .blas => gemm.has_blas,
        else => true,
    };
}

fn parse_shape(s: []const u8) !benchmark.Shape {
    var parts = std.mem.splitScalar(u8, s, 'x');
    const m = try std.fmt.parseInt(i64, parts.next() orelse return error.InvalidShape, 10);
    const n = try std.fmt.parseInt(i64, parts.next() orelse return error.InvalidShape, 10);
    const k = try std.fmt.parseInt(i64, parts.next() orelse return error.InvalidShape, 10);
    return .{ .m = m, .n = n, .k = k };
}

fn print_usage() void {
    const usage =
        \\Usage: benchmark [subcommand] [options]
        \\
        \\Subcommands:
        \\  run                     Run benchmarks (default)
        \\  tune                    Tune search-based kernels (TVM, etc.)
        \\
        \\Options:
        \\  --shapes=MxNxK[,...]    Matrix dimensions (default: 128x128x128)
        \\  --impls=name[,...]      zig_naive, blas, tvm_cpu, tvm_gpu, iree, xla_cpu,
        \\                          xla_gpu, all, all-cpu, all-gpu
        \\  --dtype=f16|f32         Element type (default: f32)
        \\  --warmup=N              Warmup iterations (default: 10)
        \\  --iters=N               Benchmark iterations (default: 100)
        \\  --trials=N              Max tuning trials per shape (tune only, default: 64)
        \\  --trials-per-iter=N     Trials per iteration (tune only, default: 16)
        \\  --retune                Delete existing artifacts and tune from scratch
        \\  --help, -h              Show this message
        \\
        \\  Cache dir: set ZG_CACHE_DIR env var (default: /tmp/zigrad-cache)
        \\
    ;
    _ = std.Io.File.stdout(); // suppress unused param warning if any
    // Usage is printed to stderr to avoid threading io into print_usage.
    std.debug.print("{s}", .{usage});
}

test {
    @import("std").testing.refAllDecls(@This());
    _ = @import("benchmark/harness.zig");
    _ = @import("benchmark/stats.zig");
    _ = @import("gemm.zig");
}

test "all-cpu excludes unavailable implementations" {
    var implementations = std.ArrayList(benchmark.Implementation).empty;
    defer implementations.deinit(std.testing.allocator);

    try append_impls(std.testing.allocator, &implementations, "all-cpu");
    if (!gemm.has_blas) {
        try std.testing.expect(std.mem.indexOfScalar(
            benchmark.Implementation,
            implementations.items,
            .blas,
        ) == null);
    }
}
