const std = @import("std");
const Writer = std.Io.Writer;

pub const OutputTarget = @import("output").Target;
const json = @import("json.zig");
const pr = @import("pr.zig");
const serialize = @import("serialize.zig");
pub const Emitter = @import("zxpr/Emitter.zig");
pub const style = @import("zxpr/style.zig");

pub const ZxprMode = enum {
    auto,
    plain,
};

pub const ZxprSpec = struct {
    mode: ZxprMode = .auto,
    style_opts: style.ConfigOpts = .{},
};

pub const DumpSpec = union(enum) {
    zxpr: ZxprSpec,
    json,
    binary,
};

pub const Config = struct {
    target: OutputTarget = .stdout,
    spec: DumpSpec = .{ .zxpr = .{} },

    /// Optional label identifying which PR function was selected as entry.
    /// Printed as "entry: <name>" header in ZXPR output for user reference.
    /// Does not affect actual function names in the dumped content.
    entry_label: ?[]const u8 = null,
};

pub fn emit_program(program: *const pr.Program, writer: *Writer, cfg: Config) !void {
    switch (cfg.spec) {
        .json => try json.emit_program(program, writer),
        .binary => try serialize.emit(program, writer),
        .zxpr => |zx| {
            if (cfg.entry_label) |name| {
                try writer.print("entry: {s}\n", .{name});
            }

            const mode: style.ConfigMode = switch (zx.mode) {
                .plain => .plain,
                .auto => switch (cfg.target) {
                    .stdout => .auto_stdout,
                    .file => .plain,
                },
            };
            const zx_cfg = style.config(mode, zx.style_opts);

            for (program.functions(), program.function_ids(), 0..) |func, function_id, i| {
                if (i > 0) try writer.writeAll("\n");
                var emitter = Emitter.init(writer, func, function_id, zx_cfg);
                try emitter.emit();
            }
        },
    }
}

/// Emit ZXPR representation of a function.
pub fn emit(func: pr.Function, writer: *Writer, cfg: style.Config) !void {
    var emitter = Emitter.init(writer, func, null, cfg);
    try emitter.emit();
}

/// Emit a single param declaration: `%0: 2x3<f32>`
pub fn emit_param_line(v: *const pr.Var, writer: *Writer) !void {
    // Create a minimal emitter just for formatting
    const dummy_func = pr.Function{
        .name = "",
        .params = &.{},
        .returns = &.{},
        .ops = &.{},
        .regions = &.{},
        .var_count = 0,
    };
    var emitter = Emitter.init(writer, dummy_func, null, style.config(.plain, .{}));
    try emitter.emit_var_with_type(v);
}

/// Emit a single op binding: `%2: 2x2<f32> = mm[M=2, K=3, N=2](%0, %1)  ; vjp`
pub fn emit_op_line(op: *const pr.Op, writer: *Writer) !void {
    const dummy_func = pr.Function{
        .name = "",
        .params = &.{},
        .returns = &.{},
        .ops = &.{},
        .regions = &.{},
        .var_count = 0,
    };
    var emitter = Emitter.init(writer, dummy_func, null, style.config(.plain, .{}));
    try emitter.emit_binding(op);
}

/// Emit a region block: header + ops + footer.
pub fn emit_region_block(func: pr.Function, region: pr.Region, writer: *Writer) !void {
    var emitter = Emitter.init(writer, func, null, style.config(.plain, .{}));
    try emitter.emit_region_start(0, region);
    for (region.op_ids) |op_id| {
        const op = func.op_by_id(op_id) orelse continue;
        try writer.writeAll("  ");
        try emitter.emit_binding(op);
        try writer.writeAll("\n");
    }
    try emitter.emit_region_end(0);
}

test {
    std.testing.refAllDecls(@This());
}

test "zxpr format" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const a = try b.param_tensor(.f32, &.{ 2, 3 });
    const c = try b.param_tensor(.f32, &.{ 3, 2 });
    const d = try b.mm(a, c);
    const func = try b.finish(.{ .returns = &.{d} });

    var buf: [512]u8 = undefined;
    var w: Writer = .fixed(&buf);
    try emit(func, &w, style.config(.plain, .{}));

    const result = w.buffered();
    try std.testing.expect(std.mem.indexOf(u8, result, "zxpr main {") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "; params (2)") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "%0: 2x3<f32>") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "let\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "mm[M=2, K=3, N=2") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "!vjp") == null);
    try std.testing.expect(std.mem.indexOf(u8, result, "!jvp") == null);

    try std.testing.expect(std.mem.indexOf(u8, result, "in %2") != null);
}

test "zxpr value ids have no rendering ceiling" {
    const value = pr.Var{
        .id = std.math.maxInt(u32),
        .aval = .{ .tensor = .{
            .dtype = .f32,
            .shape = .{ .dims = &.{} },
        } },
    };

    var buf: [64]u8 = undefined;
    var writer: Writer = .fixed(&buf);
    try emit_param_line(&value, &writer);

    try std.testing.expectEqualStrings("%4294967295: f32", writer.buffered());
}

test "zxpr with transpose shows permutation" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "t");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{ 2, 3 });
    const y = try b.transpose(x, &.{ 1, 0 });
    const func = try b.finish(.{ .returns = &.{y} });

    var buf: [256]u8 = undefined;
    var w: Writer = .fixed(&buf);
    try emit(func, &w, style.config(.plain, .{}));

    const result = w.buffered();
    try std.testing.expect(std.mem.indexOf(u8, result, "transpose[perm=[1, 0]]") != null);
}

test "zxpr kernelize region annotations" {
    const kernel = @import("kernel");
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "k");
    defer b.deinit();

    const a = try b.param_tensor(.f32, &.{ 2, 2 });
    const c = try b.param_tensor(.f32, &.{ 2, 2 });

    try b.push_region("tvm-kernel", &.{
        kernel.provider_annotation("tvm"),
        .{ .name = "example.note", .value = .{ .string = "line\n\"quoted\"" } },
        .{ .name = "example.payload", .value = .{ .bytes = &.{ 0, 127, 255 } } },
    });
    const add1 = try b.add(a, c);
    const add2 = try b.add(add1, c);
    try b.pop_region();

    const out = try b.add(add2, c);
    var func = try b.finish(.{ .returns = &.{out} });
    func.annotations = &.{.{ .name = "example.function", .value = .unit }};

    var buf: [512]u8 = undefined;
    var w: Writer = .fixed(&buf);
    try emit(func, &w, style.config(.plain, .{}));

    const result = w.buffered();
    try std.testing.expect(std.mem.indexOf(u8, result, "zxpr k[example.function]") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "> tvm-kernel[zigrad.kernel.provider=\"tvm\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "example.note=\"line\\n\\\"quoted\\\"\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "example.payload=0x007fff") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "<") != null);
}

test "program zxpr exposes monotonic function identities" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    const saved = program.checkpoint_appends();
    var removed_builder = try pr.FunctionBuilder.init(&program, "removed");
    defer removed_builder.deinit();
    _ = try program.add_function(try removed_builder.finish(.{ .returns = &.{} }));
    program.restore_appends(saved);

    var callee_builder = try pr.FunctionBuilder.init(&program, "callee");
    defer callee_builder.deinit();
    const callee_input = try callee_builder.param_tensor(.f32, &.{});
    const callee_id = try program.add_function(try callee_builder.finish(.{ .returns = &.{callee_input} }));

    var caller_builder = try pr.FunctionBuilder.init(&program, "caller");
    defer caller_builder.deinit();
    const caller_input = try caller_builder.param_tensor(.f32, &.{});
    const call_op = try caller_builder.call(callee_id, &.{caller_input});
    _ = try program.add_function(try caller_builder.finish(.{ .returns = call_op.outputs }));

    var writer: Writer.Allocating = .init(std.testing.allocator);
    defer writer.deinit();
    try emit_program(&program, &writer.writer, .{
        .target = .{ .file = "" },
        .spec = .{ .zxpr = .{ .mode = .plain } },
    });

    const output = writer.written();
    try std.testing.expect(std.mem.indexOf(u8, output, "zxpr @1 callee") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "callee=@1") != null);
}

test "emit_program binary format parses" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();
    const x = try b.param_tensor(.f32, &.{1});
    const func = try b.finish(.{ .returns = &.{x} });
    _ = try program.add_function(func);

    var output_writer: Writer.Allocating = .init(std.testing.allocator);
    defer output_writer.deinit();
    try emit_program(&program, &output_writer.writer, .{ .spec = .binary });
    const bytes = try output_writer.toOwnedSlice();
    defer std.testing.allocator.free(bytes);

    var parsed = try serialize.parse(std.testing.allocator, bytes);
    defer parsed.deinit();
    try std.testing.expectEqualStrings("main", parsed.functions()[0].name);
}
