//! Failed IREE loading through `Loader.interface.load` returns the artifact to
//!  its caller. The production loader logs each failure at error level, which
//!  the test runner counts as a test failure, so this executable owns its log
//!  root and accepts exactly the expected failure logs.

const std = @import("std");
const zg = @import("zigrad");
const AdversarialAllocator = @import("adversarial_allocator").AdversarialAllocator;

pub const std_options: std.Options = .{ .logFn = log_fn };

const add_module =
    \\module {
    \\  func.func @main(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
    \\    %0 = stablehlo.add %a, %b : tensor<4xf32>
    \\    return %0 : tensor<4xf32>
    \\  }
    \\}
;

/// Failure the next load is expected to log.
const Expectation = enum { none, invalid_module, out_of_memory };

var expectation: Expectation = .none;
var execution_errors: usize = 0;
var abi_errors: usize = 0;
var unexpected_logs: usize = 0;

fn log_fn(
    comptime level: std.log.Level,
    comptime scope: @EnumLiteral(),
    comptime format: []const u8,
    args: anytype,
) void {
    var buffer: [1024]u8 = undefined;
    const message = std.fmt.bufPrint(&buffer, format, args) catch "<message too long>";
    const expected = switch (level) {
        .err => switch (scope) {
            .@"zg/iree_execution" => switch (expectation) {
                .invalid_module => std.mem.eql(u8, message, "IREE execution failed: IreeError"),
                .out_of_memory => std.mem.eql(u8, message, "IREE execution failed: OutOfMemory"),
                .none => false,
            },
            .@"zg/iree_runtime_abi" => expectation == .invalid_module and
                std.mem.indexOf(u8, message, "INVALID_ARGUMENT; FlatBuffer verification failed: buffer header too small") != null,
            else => false,
        },
        .warn => false,
        else => true,
    };
    if (level == .err and expected) {
        if (scope == .@"zg/iree_execution") execution_errors += 1 else abi_errors += 1;
    } else if (!expected) {
        unexpected_logs += 1;
    }
    std.debug.print("[" ++ @tagName(scope) ++ "] (" ++ @tagName(level) ++ "): {s}\n", .{message});
}

fn check(ok: bool, comptime message: []const u8) !void {
    if (ok) return;
    std.debug.print("check failed: " ++ message ++ "\n", .{});
    return error.CheckFailed;
}

var failed_loads: usize = 0;

/// Load `artifact` expecting `error_kind`, then check that the caller still
///  owns an unchanged artifact and nothing was freed.
fn expect_failed_load(
    loader: *zg.iree.Loader,
    artifact: *zg.iree.Artifact,
    adversary: *AdversarialAllocator,
    expected: Expectation,
    error_kind: anyerror,
) !void {
    const bytes = artifact.bytes;
    const allocator = artifact.allocator;
    expectation = expected;
    defer expectation = .none;
    const result = loader.interface.load(artifact);
    if (result) |_| return check(false, "load failed") else |err| try check(err == error_kind, "load error kind");
    failed_loads += 1;
    try check(artifact.bytes.ptr == bytes.ptr, "artifact pointer unchanged");
    try check(artifact.bytes.len == bytes.len, "artifact length unchanged");
    try check(artifact.allocator.ptr == allocator.ptr, "artifact allocator unchanged");
    try check(artifact.allocator.vtable == allocator.vtable, "artifact allocator vtable unchanged");
    try check(adversary.frees_of(bytes.ptr) == 0, "no free before caller cleanup");
}

/// Caller-side cleanup frees the artifact once at the VMFB alignment.
fn expect_caller_cleanup(artifact: *zg.iree.Artifact, adversary: *AdversarialAllocator) !void {
    const ptr = artifact.bytes.ptr;
    artifact.deinit();
    try check(adversary.frees_of(ptr) == 1, "one caller free");
    try check(adversary.free_alignment_of(ptr).? == .@"16", "caller free alignment");
}

fn compile(
    init: std.process.Init,
    allocator: std.mem.Allocator,
    source: []u8,
) !zg.iree.Artifact {
    var compiler: zg.iree.Compiler = .{ .config = .{
        .executable = init.environ_map.get("ZG_IREE_COMPILER_PATH") orelse "iree-compile",
        .target_backend = "llvm-cpu",
    } };
    var context: zg.CompilationCtx = .{ .allocator = allocator, .io = init.io };
    const artifact = try compiler.interface.run(
        .{ .bytes = source, .encoding = .text },
        &context,
    );
    try check(@intFromPtr(artifact.bytes.ptr) % 16 == 0, "compiled artifact is 16-byte aligned");
    return artifact;
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    var adversary: AdversarialAllocator = .{ .parent = gpa };
    defer adversary.deinit();
    const allocator = adversary.allocator();

    var iree = try zg.iree.Runtime.init(gpa, .registered, .{ .driver = "local-sync" });
    defer iree.deinit();
    var execution = zg.iree.Execution.init(gpa, &iree, .{ .driver = "local-sync" });
    var loader = zg.iree.Loader.init(&execution, "module.main");

    // Invalid bytes make IREE reject the module.
    {
        const bytes = try allocator.alignedAlloc(u8, .@"16", 64);
        @memset(bytes, 0);
        var artifact: zg.iree.Artifact = .{ .bytes = bytes, .allocator = allocator };
        try expect_failed_load(&loader, &artifact, &adversary, .invalid_module, error.IntegrationFailure);
        try expect_caller_cleanup(&artifact, &adversary);
    }

    const source = try gpa.dupe(u8, add_module);
    defer gpa.free(source);

    // A compiled module whose first loading allocation fails stays with the
    // caller.
    {
        var artifact = try compile(init, allocator, source);
        adversary.allocations = 0;
        adversary.fail_at = 0;
        defer adversary.fail_at = null;
        try expect_failed_load(&loader, &artifact, &adversary, .out_of_memory, error.OutOfMemory);
        adversary.fail_at = null;
        try expect_caller_cleanup(&artifact, &adversary);
    }

    // A compiled module loads once the failing allocation is past, and the
    // program owns the VMFB until release.
    var artifact = try compile(init, allocator, source);
    const vmfb = artifact.bytes;
    var sweep_failures: usize = 0;
    for (0..64) |ordinal| {
        adversary.allocations = 0;
        adversary.fail_at = ordinal;
        defer adversary.fail_at = null;
        expectation = .out_of_memory;
        const result = loader.interface.load(&artifact);
        expectation = .none;
        if (result) |loaded| {
            var program = loaded;
            try check(sweep_failures > 0, "an allocation failure was exercised");
            try check(adversary.frees_of(vmfb.ptr) == 0, "program holds the VMFB");
            execution.interface.release_program(&program);
            break;
        } else |err| {
            try check(err == error.OutOfMemory, "load error kind");
            failed_loads += 1;
            sweep_failures += 1;
            try check(artifact.bytes.ptr == vmfb.ptr and artifact.bytes.len == vmfb.len, "artifact unchanged");
            try check(artifact.allocator.ptr == allocator.ptr, "artifact allocator unchanged");
            try check(adversary.frees_of(vmfb.ptr) == 0, "no free after failed load");
        }
    } else return error.CheckFailed;
    try check(adversary.frees_of(vmfb.ptr) == 1, "one free at program release");
    try check(adversary.free_alignment_of(vmfb.ptr).? == .@"16", "program release alignment");

    try adversary.expect_clean();
    try check(unexpected_logs == 0, "no unexpected logs");
    try check(execution_errors == failed_loads, "one execution error per failed load");
    try check(abi_errors == 1, "one abi error");
    std.debug.print("iree loader failure: {d} failed loads, {d} execution errors, {d} abi errors\n", .{
        failed_loads, execution_errors, abi_errors,
    });
}
