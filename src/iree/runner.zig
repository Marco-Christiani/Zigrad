//! Standalone runner for IREE VMFB artifacts.
//!
//! The binary shares Zigrad's IREE runtime lifecycle and adds only command
//!  parsing, input construction, and output rendering.

const std = @import("std");
const config = @import("config.zig");
const iree = @import("runtime.zig");

const log = std.log.scoped(.@"zg/iree_runner");

const Arguments = struct {
    vmfb_path: []const u8,
    function_name: []const u8,
    driver: []const u8,
    input_specs: []const []const u8,
};

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const allocator = init.gpa;

    var stderr_buffer: [1024]u8 = undefined;
    var stderr_writer = std.Io.File.stderr().writer(io, &stderr_buffer);
    const stderr = &stderr_writer.interface;
    defer stderr.flush() catch {};

    var input_specs: std.ArrayList([]const u8) = .empty;
    defer input_specs.deinit(allocator);
    const arguments = try parse_arguments(
        init.minimal.args,
        allocator,
        &input_specs,
        stderr,
    );

    const vmfb_bytes = try read_file(io, allocator, arguments.vmfb_path);
    var bytecode: iree.Bytecode = .{ .owned = .{
        .bytes = vmfb_bytes,
        .allocator = allocator,
    } };
    log.info("loaded {d} bytes from {s}", .{ vmfb_bytes.len, arguments.vmfb_path });

    var runtime = try iree.Runtime.init(
        allocator,
        .registered,
        .{ .driver = arguments.driver },
    );
    defer runtime.deinit();

    var executable = runtime.load(
        allocator,
        &bytecode,
        arguments.function_name,
    ) catch |err| {
        bytecode.deinit();
        return err;
    };
    defer executable.deinit();

    var inputs: std.ArrayList(iree.Buffer) = .empty;
    defer {
        for (inputs.items) |*input| input.deinit();
        inputs.deinit(allocator);
    }
    for (arguments.input_specs) |spec| {
        try inputs.append(
            allocator,
            try parse_input(allocator, &runtime, spec),
        );
    }

    var invocation = try executable.invoke(allocator, inputs.items);
    defer invocation.deinit();
    log.info("executed {s}", .{arguments.function_name});

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch @panic("failed to flush stdout");

    if (invocation.outputs.len == 0) {
        try stdout.writeAll("(no outputs)\n");
        return;
    }

    for (invocation.outputs, 0..) |*output, output_index| {
        const element_type = try output.element_type();
        const byte_count = try std.math.mul(
            usize,
            output.element_count(),
            element_type.byte_width(),
        );
        const bytes = try allocator.alloc(u8, byte_count);
        defer allocator.free(bytes);
        try runtime.read_buffer(output, bytes);

        try stdout.print("output[{d}]: [", .{output_index});
        try print_values(stdout, element_type, bytes);
        try stdout.writeAll("]\n");
    }
}

fn parse_arguments(
    process_args: std.process.Args,
    allocator: std.mem.Allocator,
    input_specs: *std.ArrayList([]const u8),
    stderr: *std.Io.Writer,
) !Arguments {
    var vmfb_path: ?[]const u8 = null;
    var function_name: []const u8 = "module.main";
    var driver: []const u8 = config.default_driver;

    var iterator = process_args.iterate();
    _ = iterator.next();
    while (iterator.next()) |argument| {
        if (std.mem.startsWith(u8, argument, "--function=")) {
            function_name = argument["--function=".len..];
        } else if (std.mem.startsWith(u8, argument, "--driver=")) {
            driver = argument["--driver=".len..];
        } else if (std.mem.startsWith(u8, argument, "--input=")) {
            try input_specs.append(
                allocator,
                argument["--input=".len..],
            );
        } else if (std.mem.startsWith(u8, argument, "-")) {
            try stderr.print("unknown option: {s}\n", .{argument});
            return error.UnknownOption;
        } else if (vmfb_path == null) {
            vmfb_path = argument;
        } else {
            try stderr.print("unexpected argument: {s}\n", .{argument});
            return error.UnexpectedArgument;
        }
    }

    const path = vmfb_path orelse {
        try stderr.writeAll(
            \\usage: iree-runner <vmfb-path> [options]
            \\
            \\  --function=<name>  Fully qualified function name
            \\  --driver=<name>    IREE HAL driver
            \\  --input=<spec>     Repeatable tensor input
            \\
            \\Input format: <dim0>x<dim1>x...x<dtype>[=v0,v1,...]
            \\Supported dtypes: f32, f64, i32, i64
            \\
        );
        return error.MissingArgument;
    };

    return .{
        .vmfb_path = path,
        .function_name = function_name,
        .driver = driver,
        .input_specs = input_specs.items,
    };
}

fn read_file(
    io: std.Io,
    allocator: std.mem.Allocator,
    path: []const u8,
) ![]align(iree.bytecode_alignment_bytes) u8 {
    var file = if (std.fs.path.isAbsolute(path))
        try std.Io.Dir.openFileAbsolute(io, path, .{})
    else
        try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);

    const length = try file.length(io);
    const bytes = try allocator.alignedAlloc(
        u8,
        .fromByteUnits(iree.bytecode_alignment_bytes),
        @intCast(length),
    );
    errdefer allocator.free(bytes);
    _ = try file.readPositionalAll(io, bytes, 0);
    return bytes;
}

const element_type_map = .{
    .{ iree.ElementType.f32, f32 },
    .{ iree.ElementType.f64, f64 },
    .{ iree.ElementType.i32, i32 },
    .{ iree.ElementType.i64, i64 },
};

fn print_values(
    writer: *std.Io.Writer,
    element_type: iree.ElementType,
    bytes: []const u8,
) !void {
    inline for (element_type_map) |entry| {
        if (element_type == entry[0]) {
            const T = entry[1];
            const values: []const T = @alignCast(std.mem.bytesAsSlice(T, bytes));
            for (values, 0..) |value, index| {
                if (index > 0) try writer.writeAll(", ");
                try writer.print("{d}", .{value});
            }
            return;
        }
    }
    try writer.print("<{d} bytes, unsupported element type>", .{bytes.len});
}

const input_type_map = std.StaticStringMap(iree.ElementType).initComptime(.{
    .{ "f32", .f32 },
    .{ "f64", .f64 },
    .{ "i32", .i32 },
    .{ "i64", .i64 },
});

fn parse_input(
    allocator: std.mem.Allocator,
    runtime: *iree.Runtime,
    spec: []const u8,
) !iree.Buffer {
    const equals = std.mem.indexOfScalar(u8, spec, '=');
    const shape_and_type = if (equals) |index| spec[0..index] else spec;
    const values: ?[]const u8 = if (equals) |index| spec[index + 1 ..] else null;

    var parts: std.ArrayList([]const u8) = .empty;
    defer parts.deinit(allocator);
    var iterator = std.mem.splitScalar(u8, shape_and_type, 'x');
    while (iterator.next()) |part| {
        if (part.len > 0) try parts.append(allocator, part);
    }
    if (parts.items.len < 2) {
        log.err("input requires at least one dimension and a dtype: {s}", .{spec});
        return error.InvalidInputSpec;
    }

    const type_name = parts.items[parts.items.len - 1];
    const element_type = input_type_map.get(type_name) orelse {
        log.err("unsupported input dtype '{s}'", .{type_name});
        return error.UnsupportedDtype;
    };

    const dimensions = parts.items[0 .. parts.items.len - 1];
    const shape = try allocator.alloc(i64, dimensions.len);
    defer allocator.free(shape);

    var element_count: usize = 1;
    for (dimensions, 0..) |dimension, index| {
        const value = std.fmt.parseInt(i64, dimension, 10) catch {
            log.err("invalid input dimension '{s}'", .{dimension});
            return error.InvalidDimension;
        };
        if (value <= 0) return error.InvalidDimension;
        shape[index] = value;
        element_count = std.math.mul(
            usize,
            element_count,
            @intCast(value),
        ) catch return error.DimensionOverflow;
    }

    const byte_count = std.math.mul(
        usize,
        element_count,
        element_type.byte_width(),
    ) catch return error.DimensionOverflow;
    const data = try allocator.alloc(u8, byte_count);
    defer allocator.free(data);

    if (values) |text| {
        try parse_values(text, element_type, data, element_count);
    } else {
        @memset(data, 0);
    }

    return try runtime.create_buffer(data, element_type, shape);
}

fn parse_values(
    text: []const u8,
    element_type: iree.ElementType,
    data: []u8,
    element_count: usize,
) !void {
    var iterator = std.mem.splitScalar(u8, text, ',');
    var index: usize = 0;
    while (iterator.next()) |item| {
        const value = std.mem.trim(u8, item, " ");
        if (value.len == 0) continue;
        if (index >= element_count) return error.TooManyValues;
        try parse_value(element_type, value, data, index);
        index += 1;
    }
    if (index != element_count) return error.NotEnoughValues;
}

fn parse_value(
    element_type: iree.ElementType,
    text: []const u8,
    data: []u8,
    index: usize,
) !void {
    inline for (element_type_map) |entry| {
        if (element_type == entry[0]) {
            const T = entry[1];
            const value: T = switch (@typeInfo(T)) {
                .float => try std.fmt.parseFloat(T, text),
                .int => try std.fmt.parseInt(T, text, 10),
                else => unreachable,
            };
            @as(*align(1) T, @ptrCast(data.ptr + index * @sizeOf(T))).* = value;
            return;
        }
    }
    return error.UnsupportedDtype;
}
