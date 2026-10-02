const std = @import("std");

const compilation = @import("compilation");
const output = @import("output");
const pr = @import("pr.zig");
const zxpr = @import("zxpr.zig");

pub const Target = zxpr.OutputTarget;
pub const Spec = zxpr.DumpSpec;
pub const Config = zxpr.Config;

/// Emits a PR program and returns the same borrowed program.
pub const Dump = struct {
    pub const Input = *pr.Program;
    pub const Output = *pr.Program;

    config: Config,

    pub fn run(self: Dump, program: Input, ctx: *compilation.Context) !Output {
        const task = struct {
            program: *const pr.Program,
            config: Config,

            pub fn emit(task_self: @This(), out: *std.Io.Writer) !void {
                try zxpr.emit_program(task_self.program, out, task_self.config);
            }
        }{
            .program = program,
            .config = self.config,
        };

        try output.write(ctx.io, self.config.target, task);
        return program;
    }
};

test "emit_program includes entry header and zxpr output" {
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main_fn");
    defer b.deinit();
    const x = try b.param_tensor(.f32, &.{1});
    const func = try b.finish(.{ .returns = &.{x} });
    _ = try program.add_function(func);

    var writer_state = std.Io.Writer.Allocating.init(testing.allocator);
    defer writer_state.deinit();

    try zxpr.emit_program(&program, &writer_state.writer, .{
        .target = .stdout,
        .entry_label = "test_main",
        .spec = .{ .zxpr = .{ .mode = .plain } },
    });
    const out = try writer_state.toOwnedSlice();
    defer testing.allocator.free(out);

    try testing.expect(std.mem.indexOf(u8, out, "entry: test_main") != null);
    try testing.expect(std.mem.indexOf(u8, out, "zxpr @0 main_fn") != null);
}

test "emit_program json format" {
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();
    const x = try b.param_tensor(.f32, &.{1});
    const func = try b.finish(.{ .returns = &.{x} });
    _ = try program.add_function(func);

    var writer_state = std.Io.Writer.Allocating.init(testing.allocator);
    defer writer_state.deinit();

    try zxpr.emit_program(&program, &writer_state.writer, .{
        .target = .stdout,
        .entry_label = "main",
        .spec = .json,
    });
    const out = try writer_state.toOwnedSlice();
    defer testing.allocator.free(out);

    // JSON format: no entry header, produces JSON object
    try testing.expect(std.mem.indexOf(u8, out, "entry:") == null);
    try testing.expect(std.mem.indexOf(u8, out, "\"name\":\"main\"") != null);
}
