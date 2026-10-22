//! IREE VM bytecode loading.

const std = @import("std");
const stablehlo = @import("../stablehlo.zig");
const Executor = @import("../execution.zig");
const compiler = @import("compiler.zig");
const execution = @import("execution.zig");
const config_mod = @import("config.zig");
const runtime = @import("runtime.zig");

const LoaderInterface = Executor.Loader(compiler.Artifact);

/// Loader for one IREE runtime and entry function.
pub const Loader = struct {
    interface: LoaderInterface,
    execution: *execution.Execution,
    entry_name: []const u8,

    const vtable: LoaderInterface.VTable = .{
        .load = load,
    };

    pub fn init(
        execution_instance: *execution.Execution,
        entry_name: []const u8,
    ) Loader {
        return .{
            .interface = .{
                .executor = &execution_instance.interface,
                .vtable = &vtable,
            },
            .execution = execution_instance,
            .entry_name = entry_name,
        };
    }

    fn load(
        interface: *LoaderInterface,
        artifact: *compiler.Artifact,
    ) Executor.Error!*anyopaque {
        const self: *Loader = @fieldParentPtr("interface", interface);
        var bytecode: runtime.Bytecode = .{
            .owned = .{
                .bytes = artifact.bytes,
                .allocator = artifact.allocator,
            },
        };
        const loaded = self.execution.runtime.load(
            artifact.allocator,
            &bytecode,
            self.entry_name,
        ) catch |err| return execution.map_error(err);
        artifact.* = undefined;
        return loaded._state;
    }
};

const AdversarialAllocator = @import("adversarial_allocator").AdversarialAllocator;

test AdversarialAllocator {
    var adversary: AdversarialAllocator = .{ .parent = std.testing.allocator };
    defer adversary.deinit();
    const gpa = adversary.allocator();

    const bytes = try gpa.alloc(u8, 24);
    try std.testing.expect(@intFromPtr(bytes.ptr) % 16 != 0);
    const wide = try gpa.alignedAlloc(u8, .@"16", 24);
    try std.testing.expectEqual(@as(usize, 0), @intFromPtr(wide.ptr) % 16);
    const word = try gpa.alignedAlloc(u8, .@"64", 8);
    try std.testing.expectEqual(@as(usize, 0), @intFromPtr(word.ptr) % 64);

    gpa.free(bytes);
    gpa.free(wide);
    try std.testing.expectEqual(@as(usize, 1), adversary.find(bytes.ptr).?.frees);
    try std.testing.expectEqual(std.mem.Alignment.@"16", adversary.find(wide.ptr).?.free_alignment.?);
    gpa.free(word);
    try adversary.expect_clean();
}

const add_module =
    \\module {
    \\  func.func @main(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
    \\    %0 = stablehlo.add %a, %b : tensor<4xf32>
    \\    return %0 : tensor<4xf32>
    \\  }
    \\}
;

/// Compiler configuration for the CPU module tests. The caller frees
///  `executable`.
fn cpu_compiler_config(gpa: std.mem.Allocator) !struct { []u8, config_mod.CompilerConfig } {
    const executable = std.process.Environ.getAlloc(
        std.testing.environ,
        gpa,
        "ZG_IREE_COMPILER_PATH",
    ) catch |err| switch (err) {
        error.EnvironmentVariableMissing => try gpa.dupe(u8, "iree-compile"),
        else => return err,
    };
    return .{ executable, .{ .executable = executable, .target_backend = "llvm-cpu" } };
}

test "compile returns 16-byte aligned VMFB under a byte-aligned allocator" {
    var adversary: AdversarialAllocator = .{ .parent = std.testing.allocator };
    defer adversary.deinit();
    const gpa = adversary.allocator();

    const executable, const config = try cpu_compiler_config(std.testing.allocator);
    defer std.testing.allocator.free(executable);

    const vmfb = compiler.compile(std.testing.io, gpa, add_module, .{
        .config = config,
        .encoding = .text,
    }) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    try std.testing.expect(vmfb.len > 16);
    try std.testing.expectEqual(@as(usize, 0), @intFromPtr(vmfb.ptr) % 16);
    try std.testing.expectEqual(std.mem.Alignment.@"16", adversary.find(vmfb.ptr).?.alignment);

    gpa.free(vmfb);
    try std.testing.expectEqual(std.mem.Alignment.@"16", adversary.free_alignment_of(vmfb.ptr).?);
    try adversary.expect_clean();
}

test "Loader keeps the compiled allocation until program release" {
    const testing = std.testing;
    const DType = @import("dtype").DType;
    var adversary: AdversarialAllocator = .{ .parent = testing.allocator };
    defer adversary.deinit();
    const gpa = adversary.allocator();

    const executable, const config = try cpu_compiler_config(testing.allocator);
    defer testing.allocator.free(executable);

    var compiler_instance: compiler.Compiler = .{ .config = config };
    var context: @import("compilation").Context = .{ .allocator = gpa, .io = testing.io };
    const source = try testing.allocator.dupe(u8, add_module);
    defer testing.allocator.free(source);
    var artifact = compiler_instance.interface.run(
        stablehlo.Artifact{ .bytes = source, .encoding = .text },
        &context,
    ) catch |err| switch (err) {
        error.CompilationFailed => return error.SkipZigTest,
        else => return err,
    };

    try testing.expectEqual(@as(usize, 0), @intFromPtr(artifact.bytes.ptr) % 16);
    const vmfb_ptr = artifact.bytes.ptr;
    try testing.expectEqual(std.mem.Alignment.@"16", adversary.find(vmfb_ptr).?.alignment);
    try testing.expectEqual(@as(usize, 0), adversary.frees_of(vmfb_ptr));

    var iree = try runtime.Runtime.init(testing.allocator, .registered, .{ .driver = "local-sync" });
    defer iree.deinit();
    var exec = execution.Execution.init(testing.allocator, &iree, .{ .driver = "local-sync" });
    var loader = Loader.init(&exec, "module.main");

    var program = try loader.interface.load(&artifact);
    try testing.expectEqual(@as(usize, 0), adversary.frees_of(vmfb_ptr));

    const lhs = [4]f32{ 1, 2, 3, 4 };
    const rhs = [4]f32{ 10, 20, 30, 40 };
    const shape = [_]i64{4};
    const executor = &exec.interface;
    const a = try executor.upload(std.mem.asBytes(&lhs), DType.f32, &shape);
    defer executor.release(a);
    const b = try executor.upload(std.mem.asBytes(&rhs), DType.f32, &shape);
    defer executor.release(b);
    var outputs: [1]Executor.Buffer = undefined;
    _ = try executor.invoke(program, &.{ a, b }, &outputs, .{});
    defer executor.release(outputs[0]);

    var result: [4]f32 = undefined;
    _ = try executor.download(outputs[0], std.mem.asBytes(&result));
    try testing.expectEqualSlices(f32, &.{ 11, 22, 33, 44 }, &result);
    try testing.expectEqual(@as(usize, 0), adversary.frees_of(vmfb_ptr));

    executor.release_program(&program);
    try testing.expectEqual(@as(usize, 1), adversary.frees_of(vmfb_ptr));
    try testing.expectEqual(std.mem.Alignment.@"16", adversary.free_alignment_of(vmfb_ptr).?);
    try adversary.expect_clean();
}

test "Runtime.load leaves owned bytecode with the caller when loading fails" {
    const testing = std.testing;
    var adversary: AdversarialAllocator = .{ .parent = testing.allocator };
    defer adversary.deinit();
    const gpa = adversary.allocator();

    const executable, const config = try cpu_compiler_config(testing.allocator);
    defer testing.allocator.free(executable);
    const vmfb = compiler.compile(testing.io, gpa, add_module, .{
        .config = config,
        .encoding = .text,
    }) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    var iree = try runtime.Runtime.init(testing.allocator, .registered, .{ .driver = "local-sync" });
    defer iree.deinit();

    // Fail each loading allocation in turn until loading succeeds.
    var failures: usize = 0;
    var loading: AdversarialAllocator = .{ .parent = testing.allocator };
    defer loading.deinit();
    for (0..64) |ordinal| {
        loading.allocations = 0;
        loading.fail_at = ordinal;
        var bytecode: runtime.Bytecode = .{ .owned = .{ .bytes = vmfb, .allocator = gpa } };
        var loaded = iree.load(loading.allocator(), &bytecode, "module.main") catch |err| {
            try testing.expectEqual(error.OutOfMemory, err);
            try testing.expectEqual(vmfb.ptr, bytecode.owned.bytes.ptr);
            try testing.expectEqual(vmfb.len, bytecode.owned.bytes.len);
            try testing.expectEqual(@as(usize, 0), adversary.frees_of(vmfb.ptr));
            failures += 1;
            continue;
        };
        try testing.expect(failures > 0);
        try testing.expectEqual(@as(usize, 0), adversary.frees_of(vmfb.ptr));
        loaded.deinit();
        break;
    } else return error.TestUnexpectedResult;

    try testing.expectEqual(@as(usize, 1), adversary.frees_of(vmfb.ptr));
    try testing.expectEqual(std.mem.Alignment.@"16", adversary.free_alignment_of(vmfb.ptr).?);
    try loading.expect_clean();
    try adversary.expect_clean();
}
