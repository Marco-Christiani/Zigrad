//! Out-of-process IREE compilation behind a pure Zig contract.

const std = @import("std");
const compilation = @import("compilation");
const stablehlo = @import("../stablehlo.zig");
const CompilerConfig = @import("config.zig").CompilerConfig;

const log = std.log.scoped(.@"zg/iree_compiler");

/// Encoding of the supplied MLIR source.
pub const SourceEncoding = enum {
    text,
    bytecode,
};

/// Options for one StableHLO or MLIR to VMFB compilation.
pub const CompileOptions = struct {
    /// Compiler executable, target, and temporary-file policy.
    config: CompilerConfig,

    /// Encoding of `source`.
    encoding: SourceEncoding,
};

/// IREE VM bytecode released with `deinit`.
pub const Artifact = struct {
    bytes: []u8,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *Artifact) void {
        self.allocator.free(self.bytes);
        self.* = undefined;
    }
};

const CompilerInterface = compilation.Compiler(stablehlo.Artifact, Artifact);

/// StableHLO to IREE VM bytecode compiler.
pub const Compiler = struct {
    interface: CompilerInterface = .{ .vtable = &vtable },

    config: CompilerConfig,

    const vtable: CompilerInterface.VTable = .{
        .compile = compile_interface,
    };

    fn compile_interface(
        interface: *CompilerInterface,
        source: stablehlo.Artifact,
        ctx: *compilation.Context,
    ) compilation.Error!Artifact {
        const self: *Compiler = @fieldParentPtr("interface", interface);
        const encoding: SourceEncoding = switch (source.encoding) {
            .text => .text,
            .binary => .bytecode,
        };
        const bytes = compile(ctx.io, ctx.allocator, source.bytes, .{
            .config = self.config,
            .encoding = encoding,
        }) catch |err| {
            log.err("IREE compilation failed: {s}", .{@errorName(err)});
            return switch (err) {
                error.OutOfMemory => error.OutOfMemory,
                else => error.CompilationFailed,
            };
        };
        return .{
            .bytes = bytes,
            .allocator = ctx.allocator,
        };
    }
};

/// Compile MLIR source into VMFB bytes allocated by `allocator`.
pub fn compile(
    io: std.Io,
    allocator: std.mem.Allocator,
    source: []const u8,
    options: CompileOptions,
) ![]u8 {
    const config = options.config;
    const backend_argument = try std.fmt.allocPrint(
        allocator,
        "--iree-hal-target-backends={s}",
        .{config.target_backend},
    );
    defer allocator.free(backend_argument);
    const input_argument = try std.fmt.allocPrint(
        allocator,
        "--iree-input-type={s}",
        .{config.input_type},
    );
    defer allocator.free(input_argument);

    const suffix: []const u8 = switch (options.encoding) {
        .text => ".mlir",
        .bytecode => ".mlirbc",
    };
    var input_path_buffer: [std.fs.max_path_bytes]u8 = undefined;
    var output_path_buffer: [std.fs.max_path_bytes]u8 = undefined;
    const input_path = try write_temporary_file(
        io,
        source,
        config.temporary_directory,
        suffix,
        &input_path_buffer,
    );
    defer std.Io.Dir.cwd().deleteFile(io, input_path) catch {};
    const output_path = try make_output_path(input_path, &output_path_buffer);
    defer std.Io.Dir.cwd().deleteFile(io, output_path) catch {};

    var arguments = std.ArrayList([]const u8).empty;
    defer arguments.deinit(allocator);
    try arguments.append(allocator, config.executable);
    try arguments.append(allocator, backend_argument);
    try arguments.append(allocator, input_argument);
    try arguments.appendSlice(allocator, config.extra_arguments);
    try arguments.append(allocator, input_path);
    try arguments.append(allocator, "-o");
    try arguments.append(allocator, output_path);

    log.debug("compiling {d} bytes with '{s}'", .{ source.len, config.executable });
    const result = try std.process.run(allocator, io, .{
        .argv = arguments.items,
        .expand_arg0 = .expand,
    });
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);

    switch (result.term) {
        .exited => |code| if (code != 0) {
            log.err("IREE compiler exited with code {d}: {s}", .{ code, result.stderr });
            return error.CompileFailed;
        },
        else => {
            log.err("IREE compiler terminated abnormally", .{});
            return error.CompileFailed;
        },
    }

    const vmfb = try std.Io.Dir.cwd().readFileAlloc(
        io,
        output_path,
        allocator,
        .limited(256 * 1024 * 1024),
    );
    log.debug("compiled {d} bytes of MLIR into {d} bytes of VMFB", .{
        source.len,
        vmfb.len,
    });
    return vmfb;
}

fn write_temporary_file(
    io: std.Io,
    data: []const u8,
    directory: []const u8,
    suffix: []const u8,
    path_buffer: *[std.fs.max_path_bytes]u8,
) ![]const u8 {
    const separator: []const u8 = if (std.mem.endsWith(u8, directory, "/")) "" else "/";
    for (0..8) |attempt| {
        const timestamp = std.Io.Timestamp.now(io, .awake);
        const path = std.fmt.bufPrint(
            path_buffer,
            "{s}{s}zigrad-iree-{x}-{d}{s}",
            .{
                directory,
                separator,
                @as(u64, @truncate(@as(u96, @bitCast(timestamp.nanoseconds)))),
                attempt,
                suffix,
            },
        ) catch return error.PathTooLong;

        var file = std.Io.Dir.cwd().createFile(io, path, .{
            .exclusive = true,
        }) catch |err| {
            if (err == error.PathAlreadyExists) continue;
            return err;
        };
        defer file.close(io);
        try file.writeStreamingAll(io, data);
        return path;
    }
    return error.TemporaryFileCollision;
}

fn make_output_path(
    input_path: []const u8,
    path_buffer: *[std.fs.max_path_bytes]u8,
) ![]const u8 {
    const stem = if (std.mem.lastIndexOfScalar(u8, input_path, '.')) |dot|
        input_path[0..dot]
    else
        input_path;
    return std.fmt.bufPrint(path_buffer, "{s}.vmfb", .{stem}) catch
        return error.PathTooLong;
}

test "make_output_path replaces the source suffix" {
    var buffer: [std.fs.max_path_bytes]u8 = undefined;
    try std.testing.expectEqualStrings(
        "/tmp/input.vmfb",
        try make_output_path("/tmp/input.mlirbc", &buffer),
    );
}
