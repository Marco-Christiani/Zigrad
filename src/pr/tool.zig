//! Read-only command support for serialized PR programs.

const std = @import("std");
const json = @import("json.zig");
const pr = @import("pr.zig");
const serialize = @import("serialize.zig");
const zxpr = @import("zxpr.zig");

const Allocator = std.mem.Allocator;
const Writer = std.Io.Writer;
const max_input_bytes = 1024 * 1024 * 1024;

/// Textual rendering produced from a serialized PR program.
pub const RenderFormat = enum {
    zxpr,
    json,
};

/// Read-only operation to apply to a serialized PR program.
pub const Action = union(enum) {
    render: RenderFormat,
    info,
};

/// Parse a serialized PR file and write the selected result to standard output.
pub fn run(io: std.Io, allocator: Allocator, path: []const u8, action: Action) !void {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(
        io,
        path,
        allocator,
        .limited(max_input_bytes),
    );
    defer allocator.free(bytes);

    var stdout_buffer: [8192]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buffer);
    switch (action) {
        .render => |format| {
            var program = try serialize.parse(allocator, bytes);
            defer program.deinit();
            try emit_render(&program, format, &stdout_writer.interface);
        },
        .info => {
            const header = try serialize.read_header(bytes);
            if (!header.is_supported()) {
                try emit_info(header, bytes.len, null, &stdout_writer.interface);
            } else {
                var program = serialize.parse(allocator, bytes) catch |err| {
                    try emit_info(header, bytes.len, null, &stdout_writer.interface);
                    try stdout_writer.interface.print("parse_error: {s}\n", .{@errorName(err)});
                    try stdout_writer.interface.flush();
                    return err;
                };
                defer program.deinit();
                try emit_info(header, bytes.len, &program, &stdout_writer.interface);
            }
        },
    }
    try stdout_writer.interface.flush();
}

fn emit_render(program: *const pr.Program, format: RenderFormat, writer: *Writer) !void {
    switch (format) {
        .zxpr => try zxpr.emit_program(program, writer, .{
            .spec = .{ .zxpr = .{ .mode = .plain } },
        }),
        .json => try json.emit_program(program, writer),
    }
}

fn emit_info(header: serialize.Header, input_size: usize, program: ?*const pr.Program, writer: *Writer) !void {
    try writer.print("wire: {f} supported={}\n", .{ header, header.is_supported() });
    try writer.print("size: {Bi:.2}\n", .{input_size});
    if (program) |parsed| {
        var entry_resolution_mode: enum { set, infer } = .infer;
        const entry: []const u8 = blk: {
            if (parsed.entry) |entry_id| {
                entry_resolution_mode = .set;
                break :blk parsed.get_function_by_id(entry_id).?.name;
            } else {
                entry_resolution_mode = .infer;
                const entry_id = parsed.resolve_entry() catch |err| switch (err) {
                    inline else => |e| break :blk "unresolved:" ++ @errorName(e),
                };
                break :blk parsed.get_function_by_id(entry_id).?.name;
            }
        };
        try writer.print("entry: {s}\n", .{entry});
        try writer.print("entry_resolution_mode: {t}\n", .{entry_resolution_mode});
        try writer.print("functions: {d}\n", .{parsed.functions().len});
        for (parsed.functions()) |func| {
            try writer.print(
                "function[{?d}] {s}: {d} ops, {d} values\n",
                .{ parsed.get_function_id(func.name), func.name, func.ops.len, func.var_count },
            );
            for (func.ops) |op| {
                switch (op.params) {
                    .call => |cp| try writer.print(" -> {d}\n", .{cp.callee}),
                    else => {},
                }
            }
        }
    }
}

test emit_render {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();
    const x = try b.param_tensor(.f32, &.{1});
    const y = try b.param_tensor(.f32, &.{1});
    const sum = try b.add(x, y);
    const func = try b.finish(.{ .returns = &.{sum} });
    _ = try program.add_function(func);

    for ([_]RenderFormat{ .zxpr, .json }) |format| {
        var output: Writer.Allocating = .init(std.testing.allocator);
        defer output.deinit();
        try emit_render(&program, format, &output.writer);

        const expected = switch (format) {
            .zxpr => "zxpr @0 main",
            .json => "\"name\":\"main\"",
        };
        try std.testing.expect(std.mem.indexOf(u8, output.written(), expected) != null);
    }
}

test emit_info {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "summary");
    defer b.deinit();
    const x = try b.param_tensor(.f32, &.{1});
    const y = try b.add(x, x);
    const func = try b.finish(.{ .returns = &.{y} });
    _ = try program.add_function(func);
    const header = serialize.Header{
        .magic = serialize.magic[0..serialize.magic.len].*,
        .version = serialize.version,
        .schema_hash = serialize.schema_hash,
    };

    var output: Writer.Allocating = .init(std.testing.allocator);
    defer output.deinit();
    try emit_info(header, 123, &program, &output.writer);
    var lines = std.mem.tokenizeScalar(u8, output.written(), '\n');

    try std.testing.expectStringStartsWith(lines.next().?, "wire: magic=ZGPRWIRE");
    try std.testing.expectEqualStrings("size: 123B", lines.next().?);
    try std.testing.expectEqualStrings("entry: summary", lines.next().?);
    try std.testing.expectEqualStrings("entry_resolution_mode: infer", lines.next().?);
    try std.testing.expectEqualStrings("functions: 1", lines.next().?);
    try std.testing.expectEqualStrings("function[0] summary: 1 ops, 2 values", lines.next().?);
}

test "emit_info reports an unsupported header without a program" {
    const header = serialize.Header{
        .magic = serialize.magic[0..serialize.magic.len].*,
        .version = serialize.version - 1,
        .schema_hash = serialize.schema_hash,
    };
    var output: Writer.Allocating = .init(std.testing.allocator);
    defer output.deinit();

    try emit_info(header, 42, null, &output.writer);

    try std.testing.expect(std.mem.indexOf(u8, output.written(), "supported=false") != null);
    try std.testing.expect(std.mem.indexOf(u8, output.written(), "functions:") == null);
}
