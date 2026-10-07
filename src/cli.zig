const std = @import("std");
const zg = @import("zigrad");
pub const schema = @import("cli/schema.zig");
pub const metadata = @import("cli/render.zig");

/// Options that apply to every command.
pub const GlobalOpts = struct {
    /// Emit PR to stdout or a file.
    dump_pr: ?zg.pr.dump.Config = null,
    /// Emit MLIR to stdout or a file.
    dump_mlir: ?zg.output.Config = null,
    /// Emit optimized HLO from the PJRT executable.
    dump_optimized_hlo: ?zg.output.Config = null,
    /// Emit the kernelization report to stdout or a file.
    dump_kernels: ?zg.output.Config = null,
    /// Reduce command output.
    quiet: bool = false,
};

/// Options for `tvm tune`.
pub const TvmTuneOpts = struct {
    /// Matmul dimensions in MxNxK form.
    shape: ?[]const u8 = "128x128x128",
    /// Maximum number of tuning trials.
    trials: ?u32 = 64,
    /// Number of trials submitted per tuning iteration.
    trials_per_iter: ?u32 = 16,
    /// Select the CUDA target.
    cuda: bool = false,

    /// Select the CUDA target through the `gpu` alias.
    gpu: bool = false,
    /// Select the CPU target.
    cpu: bool = false,
};

/// Options for `tvm run`.
pub const TvmRunOpts = struct {
    /// Matmul dimensions in MxNxK form.
    shape: ?[]const u8 = null,
    /// Select the CUDA target.
    cuda: bool = false,

    /// Select the CUDA target through the `gpu` alias.
    gpu: bool = false,
    /// Select the CPU target.
    cpu: bool = false,
};

/// Options for rendering TVM-annotated PR.
pub const TvmRenderOpts = struct {
    /// Render the program with every available color palette.
    sweep_palettes: bool = false,
    /// Render with one named color palette.
    palette: ?[]const u8 = null,
};

/// Options for `demo kernel-provider`.
pub const KernelProviderDemoOpts = struct {
    /// Comma-separated provider names.
    provider: ?[]const u8 = "tvm",
    /// Matmul dimensions in MxNxK form.
    shape: ?[]const u8 = "128x128x128",
    /// Element data type used by the candidate region.
    dtype: ?[]const u8 = "f32",
    /// Candidate graph family.
    pattern: ?[]const u8 = "epilogue",
};

/// Terminal backend used by an executable demo.
pub const DemoBackend = enum {
    pjrt,
    iree,
};

/// Options shared by demos without scenario-specific configuration.
pub const DemoOpts = struct {
    /// Terminal backend used to compile and execute the scenario.
    backend: DemoBackend = .pjrt,
};

/// Options for the small training demos.
pub const TrainDemoOpts = struct {
    /// Terminal backend used to compile and execute the scenario.
    backend: DemoBackend = .pjrt,
    /// Number of warmup iterations.
    warmup: ?u32 = null,
    /// Number of measured training steps.
    steps: ?u32 = null,
};

/// Options for `iree compile`.
pub const IreeCompileOpts = struct {
    /// Serialized PR file to compile.
    path: []const u8,
    /// Destination for the VMFB artifact.
    output: ?[]const u8 = null,
    /// IREE compilation target.
    target: ?[]const u8 = null,
    /// PR function compiled as the IREE entry point.
    entry: ?[]const u8 = null,
    /// Arguments following `--`, passed to `iree-compile`.
    compiler_arguments: []const []const u8 = &.{},
};

/// Options for rendering a serialized PR file.
pub const PrRenderOpts = struct {
    /// Serialized PR file to read.
    path: []const u8,
    /// Output representation.
    format: zg.pr.tool.RenderFormat = .zxpr,
};

/// Options for inspecting a serialized PR file.
pub const PrInfoOpts = struct {
    /// Serialized PR file to read.
    path: []const u8,
};

/// Options for a PJRT executable-cache operation.
pub const CacheOpts = struct {
    /// Cache artifact to read or write.
    path: []const u8,
};

/// Commands that operate on PR artifacts.
pub const PrCommand = union(enum) {
    print_demo,
    render: PrRenderOpts,
    info: PrInfoOpts,
};

/// Options for `tvm symbols`.
pub const TvmSymbolsOpts = struct {
    /// Load compiler symbols in addition to runtime symbols.
    load_compiler: bool = false,
};

/// Commands for the optional TVM integration.
pub const TvmCommand = union(enum) {
    symbols: TvmSymbolsOpts,
    check_load,
    tune: TvmTuneOpts,
    run: TvmRunOpts,
    render_matmul: TvmRenderOpts,
    render_attention: TvmRenderOpts,
};

/// Commands for the optional IREE integration.
pub const IreeCommand = union(enum) {
    compile: IreeCompileOpts,
};

/// PJRT executable-cache operations.
pub const CacheCommand = union(enum) {
    save: CacheOpts,
    run: CacheOpts,
};

/// Commands that directly exercise the optional PJRT integration.
pub const PjrtCommand = union(enum) {
    aot_demo,
    cache: CacheCommand,
};

/// Executable Zigrad demos.
pub const DemoCommand = union(enum) {
    basic: DemoOpts,
    custom_call_negative: DemoOpts,
    kernel_provider: KernelProviderDemoOpts,
    vjp: DemoOpts,
    train: TrainDemoOpts,
    llm_train: TrainDemoOpts,
};

/// Root command groups.
pub const Command = union(enum) {
    pr: PrCommand,
    tvm: TvmCommand,
    iree: IreeCommand,
    pjrt: PjrtCommand,
    demo: DemoCommand,
};

/// Parsed command-line invocation.
pub const Invocation = struct {
    global: GlobalOpts,
    command: Command,
};

/// Process-argument parse result released with `deinit`.
///
/// Command string slices remain valid until `deinit` is called.
pub const Parsed = struct {
    arena: std.heap.ArenaAllocator,
    invocation: Invocation,

    /// Release parsed arguments and invalidate their string slices.
    pub fn deinit(self: *Parsed) void {
        self.arena.deinit();
    }
};

const Cursor = struct {
    args: []const []const u8,
    index: usize = 0,

    fn next(self: *Cursor) ?[]const u8 {
        if (self.index == self.args.len) return null;
        defer self.index += 1;
        return self.args[self.index];
    }

    fn require(self: *Cursor) ![]const u8 {
        return self.next() orelse error.MissingArgument;
    }

    fn expect_end(self: *const Cursor) !void {
        if (self.index != self.args.len) return error.UnexpectedArgument;
    }
};

const LongOption = struct {
    name: []const u8,
    value: ?[]const u8,
};

/// Parse process arguments into arena-backed command values.
///
/// `Parsed.deinit` invalidates every string slice in the result.
pub fn parse(env: zg.RuntimeEnv, args: std.process.Args) !Parsed {
    var arena = std.heap.ArenaAllocator.init(env.allocator);
    errdefer arena.deinit();
    const allocator = arena.allocator();

    var process_args = try std.process.Args.Iterator.initAllocator(args, allocator);
    defer process_args.deinit();
    _ = process_args.next();

    var tokens: std.ArrayList([]const u8) = .empty;
    defer tokens.deinit(allocator);
    while (process_args.next()) |arg|
        try tokens.append(allocator, arg);

    if (schema.help_target(tokens.items)) |command| {
        try write_help(env.io, command);
        return error.HelpShown;
    }
    const invocation = parse_tokens(tokens.items) catch |err| {
        try write_parse_error(env.io, err, tokens.items);
        return error.InvalidArguments;
    };

    return .{
        .arena = arena,
        .invocation = invocation,
    };
}

fn parse_tokens(args: []const []const u8) !Invocation {
    if (args.len == 0) return error.MissingCommand;

    var cursor = Cursor{ .args = args };
    var global: GlobalOpts = .{};
    try parse_global_options(&cursor, &global);

    const root_arg = cursor.next() orelse return error.MissingCommand;
    const root = schema.find_child(&schema.root, root_arg) orelse
        return error.UnknownCommandGroup;
    const command: Command = switch (root.id) {
        .pr => .{ .pr = try parse_pr(&cursor) },
        .tvm => .{ .tvm = try parse_tvm(&cursor) },
        .iree => .{ .iree = try parse_iree(&cursor) },
        .pjrt => .{ .pjrt = try parse_pjrt(&cursor) },
        .demo => .{ .demo = try parse_demo(&cursor) },
        else => unreachable,
    };
    return .{ .global = global, .command = command };
}

fn parse_global_options(cursor: *Cursor, opts: *GlobalOpts) !void {
    comptime validate_option_struct(GlobalOpts, &schema.root);

    while (cursor.index < cursor.args.len) {
        const arg = cursor.args[cursor.index];
        const option = parse_long_option(arg) orelse return;
        cursor.index += 1;

        const negated = std.mem.startsWith(u8, option.name, "no-");
        const raw_name = if (negated) option.name["no-".len..] else option.name;
        const option_spec = schema.find_option(&schema.root, raw_name) orelse
            return error.UnknownGlobalOption;
        if (negated and !option_spec.negatable) return error.UnknownGlobalOption;
        if (negated and option.value != null) return error.UnexpectedOptionValue;

        var normalized_buf: [64]u8 = undefined;
        if (option_spec.long_name.len > normalized_buf.len)
            return error.UnknownGlobalOption;
        for (option_spec.long_name, 0..) |byte, index|
            normalized_buf[index] = if (byte == '-') '_' else byte;
        const normalized = normalized_buf[0..option_spec.long_name.len];

        const Field = std.meta.FieldEnum(GlobalOpts);
        const field = std.meta.stringToEnum(Field, normalized) orelse
            return error.UnknownGlobalOption;
        switch (field) {
            .dump_pr => opts.dump_pr = parse_dump_pr_value(option.value orelse ""),
            .dump_mlir => opts.dump_mlir = dump_config(option.value),
            .dump_optimized_hlo => opts.dump_optimized_hlo = dump_config(option.value),
            .dump_kernels => opts.dump_kernels = dump_config(option.value),
            .quiet => opts.quiet = try parse_flag(option.value, !negated),
        }
    }
}

fn parse_pr(cursor: *Cursor) !PrCommand {
    const command = try require_subcommand(.pr, cursor);
    return switch (command.id) {
        .pr_print_demo => result: {
            try cursor.expect_end();
            break :result .print_demo;
        },
        .pr_render => result: {
            const path = try require_positional(cursor);
            var opts = PrRenderOpts{ .path = path };
            try parse_options(PrRenderOpts, .pr_render, cursor, &opts);
            break :result .{ .render = opts };
        },
        .pr_info => result: {
            const path = try require_positional(cursor);
            try cursor.expect_end();
            break :result .{ .info = .{ .path = path } };
        },
        else => unreachable,
    };
}

fn parse_tvm(cursor: *Cursor) !TvmCommand {
    const command = try require_subcommand(.tvm, cursor);
    return switch (command.id) {
        .tvm_symbols => .{ .symbols = try parse_default_options(TvmSymbolsOpts, .tvm_symbols, cursor) },
        .tvm_check_load => result: {
            try cursor.expect_end();
            break :result .check_load;
        },
        .tvm_tune => .{ .tune = try parse_default_options(TvmTuneOpts, .tvm_tune, cursor) },
        .tvm_run => .{ .run = try parse_default_options(TvmRunOpts, .tvm_run, cursor) },
        .tvm_render_matmul => .{ .render_matmul = try parse_default_options(TvmRenderOpts, .tvm_render_matmul, cursor) },
        .tvm_render_attention => .{ .render_attention = try parse_default_options(TvmRenderOpts, .tvm_render_attention, cursor) },
        else => unreachable,
    };
}

fn parse_iree(cursor: *Cursor) !IreeCommand {
    const command = try require_subcommand(.iree, cursor);
    return switch (command.id) {
        .iree_compile => result: {
            const path = try require_positional(cursor);
            var opts = IreeCompileOpts{ .path = path };
            const option_start = cursor.index;
            var option_end = cursor.args.len;
            for (cursor.args[option_start..], option_start..) |arg, index| {
                if (std.mem.eql(u8, arg, "--")) {
                    option_end = index;
                    opts.compiler_arguments = cursor.args[index + 1 ..];
                    break;
                }
            }
            var option_cursor = Cursor{ .args = cursor.args[option_start..option_end] };
            try parse_options(IreeCompileOpts, .iree_compile, &option_cursor, &opts);
            cursor.index = cursor.args.len;
            break :result .{ .compile = opts };
        },
        else => unreachable,
    };
}

fn parse_pjrt(cursor: *Cursor) !PjrtCommand {
    const command = try require_subcommand(.pjrt, cursor);
    return switch (command.id) {
        .pjrt_aot_demo => result: {
            try cursor.expect_end();
            break :result .aot_demo;
        },
        .pjrt_cache => .{ .cache = try parse_cache(cursor) },
        else => unreachable,
    };
}

fn parse_cache(cursor: *Cursor) !CacheCommand {
    const command = try require_subcommand(.pjrt_cache, cursor);
    const path = try require_positional(cursor);
    try cursor.expect_end();
    const opts = CacheOpts{ .path = path };
    return switch (command.id) {
        .pjrt_cache_save => .{ .save = opts },
        .pjrt_cache_run => .{ .run = opts },
        else => unreachable,
    };
}

fn parse_demo(cursor: *Cursor) !DemoCommand {
    const command = try require_subcommand(.demo, cursor);
    return switch (command.id) {
        .demo_basic => .{ .basic = try parse_default_options(DemoOpts, .demo_basic, cursor) },
        .demo_custom_call_negative => .{ .custom_call_negative = try parse_default_options(DemoOpts, .demo_custom_call_negative, cursor) },
        .demo_kernel_provider => .{ .kernel_provider = try parse_default_options(KernelProviderDemoOpts, .demo_kernel_provider, cursor) },
        .demo_vjp => .{ .vjp = try parse_default_options(DemoOpts, .demo_vjp, cursor) },
        .demo_train => .{ .train = try parse_default_options(TrainDemoOpts, .demo_train, cursor) },
        .demo_llm_train => .{ .llm_train = try parse_default_options(TrainDemoOpts, .demo_llm_train, cursor) },
        else => unreachable,
    };
}

fn require_subcommand(
    parent_id: schema.CommandId,
    cursor: *Cursor,
) !*const schema.Command {
    const parent = schema.find_by_id(parent_id) orelse unreachable;
    const name = cursor.next() orelse return error.MissingCommand;
    return schema.find_child(parent, name) orelse error.UnknownCommand;
}

fn parse_default_options(
    comptime T: type,
    comptime command_id: schema.CommandId,
    cursor: *Cursor,
) !T {
    var opts: T = .{};
    try parse_options(T, command_id, cursor, &opts);
    return opts;
}

fn parse_options(
    comptime T: type,
    comptime command_id: schema.CommandId,
    cursor: *Cursor,
    opts: *T,
) !void {
    const command = comptime schema.find_by_id(command_id) orelse
        @compileError("CLI parser references an unknown command id");
    comptime validate_option_struct(T, command);

    while (cursor.next()) |arg| {
        const option = parse_long_option(arg) orelse return error.UnexpectedArgument;
        const negated = std.mem.startsWith(u8, option.name, "no-");
        const raw_name = if (negated) option.name["no-".len..] else option.name;
        const option_spec = schema.find_option(command, raw_name) orelse
            return error.UnknownOption;
        if (negated and !option_spec.negatable) return error.UnknownOption;

        var normalized_buf: [64]u8 = undefined;
        if (option_spec.long_name.len > normalized_buf.len) return error.UnknownOption;
        for (option_spec.long_name, 0..) |c, index|
            normalized_buf[index] = if (c == '-') '_' else c;
        const normalized = normalized_buf[0..option_spec.long_name.len];

        const Field = std.meta.FieldEnum(T);
        const field = std.meta.stringToEnum(Field, normalized) orelse return error.UnknownOption;
        switch (field) {
            inline else => |field_tag| {
                const field_name = @tagName(field_tag);
                const FieldType = @FieldType(T, field_name);
                if (comptime FieldType == []const []const u8) {
                    unreachable;
                } else {
                    @field(opts.*, field_name) = try parse_option_value(
                        FieldType,
                        cursor,
                        option.value,
                        negated,
                    );
                }
            },
        }
    }
}

fn validate_option_struct(comptime T: type, comptime command: *const schema.Command) void {
    inline for (command.options) |option| {
        if (command == &schema.root and std.mem.eql(u8, option.long_name, "help"))
            continue;
        comptime var found = false;
        inline for (@field(@typeInfo(T), @tagName(@typeInfo(T))).field_names) |field_name| {
            if (schema_name_matches_field(option.long_name, field_name))
                found = true;
        }
        if (!found)
            @compileError("CLI option --" ++ option.long_name ++
                " has no matching field in " ++ @typeName(T));
    }

    inline for (@field(@typeInfo(T), @tagName(@typeInfo(T))).field_names, @field(@typeInfo(T), @tagName(@typeInfo(T))).field_types) |field_name, field_type| {
        if (field_type == []const []const u8 and
            std.mem.eql(u8, field_name, "compiler_arguments"))
            continue;
        comptime var found = false;
        inline for (command.options) |option| {
            if (schema_name_matches_field(option.long_name, field_name))
                found = true;
        }
        inline for (command.positionals) |positional| {
            if (schema_name_matches_field(positional.name, field_name))
                found = true;
        }
        if (!found)
            @compileError("CLI field " ++ field_name ++ " in " ++ @typeName(T) ++
                " has no matching schema entry");
    }
}

fn schema_name_matches_field(
    comptime schema_name: []const u8,
    comptime field_name: []const u8,
) bool {
    if (schema_name.len != field_name.len) return false;
    for (schema_name, field_name) |schema_byte, field_byte| {
        const normalized = if (schema_byte == '-')
            '_'
        else
            std.ascii.toLower(schema_byte);
        if (normalized != field_byte) return false;
    }
    return true;
}

fn parse_option_value(
    comptime T: type,
    cursor: *Cursor,
    inline_value: ?[]const u8,
    negated: bool,
) !T {
    if (T == bool) {
        if (negated) {
            if (inline_value != null) return error.UnexpectedOptionValue;
            return false;
        }
        return try parse_flag(inline_value, true);
    }
    if (negated) return error.UnknownOption;

    const raw = inline_value orelse value: {
        const next = try cursor.require();
        if (std.mem.startsWith(u8, next, "--")) return error.MissingOptionValue;
        break :value next;
    };
    return try parse_value(T, raw);
}

fn parse_value(comptime T: type, raw: []const u8) !T {
    if (T == []const u8) return raw;

    return switch (@typeInfo(T)) {
        .int => std.fmt.parseInt(T, raw, 10),
        .optional => |info| try parse_value(info.child, raw),
        .@"enum" => std.meta.stringToEnum(T, raw) orelse error.InvalidOptionValue,
        else => @compileError("unsupported CLI option type: " ++ @typeName(T)),
    };
}

fn parse_flag(value: ?[]const u8, default: bool) !bool {
    const raw = value orelse return default;
    const values = std.StaticStringMap(bool).initComptime(.{
        .{ "true", true },
        .{ "false", false },
    });
    return values.get(raw) orelse error.InvalidOptionValue;
}

fn parse_long_option(arg: []const u8) ?LongOption {
    if (!std.mem.startsWith(u8, arg, "--") or arg.len == 2) return null;
    const body = arg[2..];
    const equals = std.mem.indexOfScalar(u8, body, '=') orelse
        return .{ .name = body, .value = null };
    return .{
        .name = body[0..equals],
        .value = body[equals + 1 ..],
    };
}

fn require_positional(cursor: *Cursor) ![]const u8 {
    const value = try cursor.require();
    if (std.mem.startsWith(u8, value, "--")) return error.MissingArgument;
    return value;
}

fn dump_config(value: ?[]const u8) zg.output.Config {
    return if (value) |path|
        .{ .target = .{ .file = path } }
    else
        .{ .target = .stdout };
}

const stdout_dump_specs = std.StaticStringMap(zg.pr.dump.Spec).initComptime(.{
    .{ "", @as(zg.pr.dump.Spec, .{ .zxpr = .{ .mode = .auto } }) },
    .{ "auto", @as(zg.pr.dump.Spec, .{ .zxpr = .{ .mode = .auto } }) },
    .{ "plain", @as(zg.pr.dump.Spec, .{ .zxpr = .{ .mode = .plain } }) },
    .{ "json", @as(zg.pr.dump.Spec, .json) },
    .{ "binary", @as(zg.pr.dump.Spec, .binary) },
});

fn parse_dump_pr_value(value: []const u8) zg.pr.dump.Config {
    if (stdout_dump_specs.get(value)) |spec|
        return .{ .target = .stdout, .spec = spec };

    const spec: zg.pr.dump.Spec =
        if (std.mem.endsWith(u8, value, ".json"))
            .json
        else if (std.mem.endsWith(u8, value, ".zgpr"))
            .binary
        else
            .{ .zxpr = .{ .mode = .auto } };
    return .{ .target = .{ .file = value }, .spec = spec };
}

fn write_help(io: std.Io, command: *const schema.Command) !void {
    var stdout_buffer: [8192]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buffer);
    try metadata.write_help(&stdout_writer.interface, command);
    try stdout_writer.interface.flush();
}

fn write_parse_error(
    io: std.Io,
    parse_error: anyerror,
    args: []const []const u8,
) !void {
    const message = switch (parse_error) {
        error.MissingCommand => "missing command",
        error.UnknownCommandGroup => "unknown command group",
        error.UnknownGlobalOption => "unknown global option",
        error.UnknownCommand => "unknown command",
        error.MissingArgument => "missing argument",
        error.UnexpectedArgument => "unexpected argument",
        error.UnknownOption => "unknown option",
        error.UnexpectedOptionValue => "unexpected option value",
        error.MissingOptionValue => "missing option value",
        error.InvalidOptionValue, error.InvalidCharacter, error.Overflow => "invalid option value",
        else => @errorName(parse_error),
    };
    const command = schema.command_target(args);

    var stderr_buffer: [1024]u8 = undefined;
    var stderr_writer = std.Io.File.stderr().writer(io, &stderr_buffer);
    try stderr_writer.interface.print("zigrad: {s}\nTry '", .{message});
    try metadata.write_command_path(&stderr_writer.interface, command);
    try stderr_writer.interface.writeAll(" --help'.\n");
    try stderr_writer.interface.flush();
}

test "parse_tokens groups commands without feature-dependent names" {
    const args = [_][]const u8{
        "--dump-pr=program.zgpr",
        "--quiet",
        "demo",
        "train",
        "--backend=iree",
        "--warmup=2",
        "--steps",
        "4",
    };
    const invocation = try parse_tokens(&args);

    try std.testing.expect(invocation.global.quiet);
    const dump_pr = invocation.global.dump_pr orelse return error.TestExpectedEqual;
    switch (dump_pr.target) {
        .file => |path| try std.testing.expectEqualStrings("program.zgpr", path),
        else => return error.TestExpectedEqual,
    }
    switch (invocation.command) {
        .demo => |demo| switch (demo) {
            .train => |opts| {
                try std.testing.expectEqual(DemoBackend.iree, opts.backend);
                try std.testing.expectEqual(@as(?u32, 2), opts.warmup);
                try std.testing.expectEqual(@as(?u32, 4), opts.steps);
            },
            else => return error.TestExpectedEqual,
        },
        else => return error.TestExpectedEqual,
    }
}

test "parse_tokens configures compiler output destinations" {
    const args = [_][]const u8{
        "--dump-mlir=backend-input.mlir",
        "--dump-optimized-hlo",
        "--dump-kernels=kernel-report.txt",
        "demo",
        "kernel-provider",
        "--provider=mirage",
    };
    const invocation = try parse_tokens(&args);

    const mlir = invocation.global.dump_mlir orelse return error.TestExpectedEqual;
    switch (mlir.target) {
        .file => |path| try std.testing.expectEqualStrings("backend-input.mlir", path),
        else => return error.TestExpectedEqual,
    }
    const optimized_hlo = invocation.global.dump_optimized_hlo orelse
        return error.TestExpectedEqual;
    try std.testing.expectEqual(zg.output.Target.stdout, optimized_hlo.target);
    const kernels = invocation.global.dump_kernels orelse return error.TestExpectedEqual;
    switch (kernels.target) {
        .file => |path| try std.testing.expectEqualStrings("kernel-report.txt", path),
        else => return error.TestExpectedEqual,
    }
}

test "parse_tokens distinguishes demo backends from IREE targets" {
    {
        const args = [_][]const u8{ "demo", "basic", "--backend=pjrt" };
        const invocation = try parse_tokens(&args);
        switch (invocation.command) {
            .demo => |demo| switch (demo) {
                .basic => |opts| try std.testing.expectEqual(DemoBackend.pjrt, opts.backend),
                else => return error.TestExpectedEqual,
            },
            else => return error.TestExpectedEqual,
        }
    }
    {
        const args = [_][]const u8{
            "iree",
            "compile",
            "model.zgpr",
            "--target=llvm-cpu",
            "--entry=forward",
            "--",
            "--iree-llvmcpu-target-cpu=cortex-a72",
        };
        const invocation = try parse_tokens(&args);
        switch (invocation.command) {
            .iree => |command| switch (command) {
                .compile => |opts| {
                    try std.testing.expectEqualStrings("model.zgpr", opts.path);
                    try std.testing.expectEqualStrings(
                        "llvm-cpu",
                        opts.target orelse return error.TestExpectedEqual,
                    );
                    try std.testing.expectEqualStrings(
                        "forward",
                        opts.entry orelse return error.TestExpectedEqual,
                    );
                    try std.testing.expectEqualSlices(
                        []const u8,
                        &.{"--iree-llvmcpu-target-cpu=cortex-a72"},
                        opts.compiler_arguments,
                    );
                },
            },
            else => return error.TestExpectedEqual,
        }
    }
}

test "parse_tokens parses nested artifact commands" {
    {
        const args = [_][]const u8{ "pr", "render", "model.zgpr", "--format=json" };
        const invocation = try parse_tokens(&args);
        switch (invocation.command) {
            .pr => |command| switch (command) {
                .render => |opts| {
                    try std.testing.expectEqualStrings("model.zgpr", opts.path);
                    try std.testing.expectEqual(zg.pr.tool.RenderFormat.json, opts.format);
                },
                else => return error.TestExpectedEqual,
            },
            else => return error.TestExpectedEqual,
        }
    }
    {
        const args = [_][]const u8{ "pjrt", "cache", "save", "artifact.bin" };
        const invocation = try parse_tokens(&args);
        switch (invocation.command) {
            .pjrt => |command| switch (command) {
                .cache => |cache| switch (cache) {
                    .save => |opts| try std.testing.expectEqualStrings("artifact.bin", opts.path),
                    else => return error.TestExpectedEqual,
                },
                else => return error.TestExpectedEqual,
            },
            else => return error.TestExpectedEqual,
        }
    }
}

test "parse_tokens supports global boolean negation" {
    const args = [_][]const u8{
        "--quiet",
        "--no-quiet",
        "pr",
        "print-demo",
    };
    const invocation = try parse_tokens(&args);
    try std.testing.expect(!invocation.global.quiet);
}

test "parse_tokens rejects values on negated booleans" {
    const args = [_][]const u8{
        "--no-quiet=true",
        "pr",
        "print-demo",
    };
    try std.testing.expectError(error.UnexpectedOptionValue, parse_tokens(&args));
}

test "schema resolves every command id" {
    inline for (@field(@typeInfo(schema.CommandId), @tagName(@typeInfo(schema.CommandId))).field_values) |field_value| {
        const id: schema.CommandId = @fromBackingInt(@intCast(field_value));
        try std.testing.expect(schema.find_by_id(id) != null);
    }
}

test {
    _ = metadata;
}
