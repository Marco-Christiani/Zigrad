//! CLI support for compiling serialized PR to IREE VM bytecode.

const std = @import("std");
const zg = @import("zigrad");

pub const Options = struct {
    /// Serialized PR file to compile.
    input: []const u8,

    /// Destination for the emitted VMFB artifact.
    output: ?[]const u8 = null,

    /// IREE compilation target override.
    target: ?[]const u8 = null,

    /// PR function compiled as the IREE entry point.
    entry: ?[]const u8 = null,

    /// Additional arguments passed to `iree-compile`.
    compiler_arguments: []const []const u8 = &.{},

    /// Optional destination and format for PR output.
    pr: ?zg.pr_dump.Config = null,

    /// Optional destination for the MLIR passed to IREE.
    mlir: ?zg.output.Config = null,
};

pub fn run(
    io: std.Io,
    allocator: std.mem.Allocator,
    environ: *const std.process.Environ.Map,
    opts: Options,
) !void {
    var config = zg.iree.Config.from_environ(environ);
    if (opts.target) |target|
        config.compiler.target_backend = target;

    config.compiler.extra_arguments = opts.compiler_arguments;

    const serialized = try std.Io.Dir.cwd().readFileAlloc(
        io,
        opts.input,
        allocator,
        .unlimited,
    );
    defer allocator.free(serialized);
    var program = try zg.pr.serialize.parse(allocator, serialized);
    defer program.deinit();
    const entry_id = try select_entry(&program, opts.entry);
    const entry = program.get_function_by_id(entry_id) orelse unreachable;
    const entry_label = entry.name;

    var ctx = zg.CompilationCtx{
        .allocator = allocator,
        .io = io,
    };
    var compiler = zg.iree.Compiler{ .config = config.compiler };
    var pipeline = zg.Pipeline.init(allocator);
    defer pipeline.deinit();
    try pipeline.add(zg.Validate{});
    if (opts.pr) |selected| {
        var output_config = selected;
        output_config.entry_label = output_config.entry_label orelse entry_label;
        try pipeline.add(zg.pr_dump.Dump{ .config = output_config });
    }
    try pipeline.add(zg.outline.Pass{});
    try pipeline.add(zg.mlir.stablehlo.Lower{
        .config = .{
            .encoding = if (opts.mlir == null) .binary else .text,
        },
    });
    if (opts.mlir) |selected| {
        var output_config = selected;
        output_config.entry_label = output_config.entry_label orelse entry_label;
        try pipeline.add(zg.stablehlo.Dump{ .config = output_config });
    }
    try pipeline.add(&compiler.interface);
    var vmfb = try pipeline.run(
        zg.iree.Artifact,
        &program,
        &ctx,
    );
    defer vmfb.deinit();

    const output_path = opts.output orelse "program.vmfb";
    var output = try std.Io.Dir.cwd().createFile(io, output_path, .{ .truncate = true });
    defer output.close(io);
    try output.writeStreamingAll(io, vmfb.bytes);
    std.log.info("wrote {d} bytes VMFB -> {s}", .{ vmfb.bytes.len, output_path });
}

fn select_entry(program: *zg.pr.Program, requested_name: ?[]const u8) !zg.pr.FunctionId {
    if (requested_name) |name| {
        const requested = program.get_function_id(name) orelse return error.EntryNotFound;
        try program.set_entry(requested);
    }
    return try program.resolve_entry();
}
