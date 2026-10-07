const std = @import("std");

/// Stable identifiers for commands in the Zigrad CLI.
pub const CommandId = enum {
    root,
    pr,
    pr_print_demo,
    pr_render,
    pr_info,
    tvm,
    tvm_symbols,
    tvm_check_load,
    tvm_tune,
    tvm_run,
    tvm_render_matmul,
    tvm_render_attention,
    iree,
    iree_compile,
    pjrt,
    pjrt_aot_demo,
    pjrt_cache,
    pjrt_cache_save,
    pjrt_cache_run,
    demo,
    demo_basic,
    demo_custom_call_negative,
    demo_kernel_provider,
    demo_vjp,
    demo_train,
    demo_llm_train,
};

/// Description of one command-line option.
pub const Option = struct {
    long_name: []const u8,
    short_name: ?u8 = null,
    description: []const u8,
    value_name: ?[]const u8 = null,
    value_optional: bool = false,
    negatable: bool = false,
    choices: []const []const u8 = &.{},
};

/// Description of one positional argument.
pub const Positional = struct {
    name: []const u8,
    description: []const u8,
    optional: bool = false,
};

/// One node in the command tree.
pub const Command = struct {
    id: CommandId,
    name: []const u8,
    summary: []const u8,
    description: ?[]const u8 = null,
    options: []const Option = &.{},
    positionals: []const Positional = &.{},
    subcommands: []const Command = &.{},
    requirement: ?[]const u8 = null,
    hidden: bool = false,
};

pub const global_options = [_]Option{
    .{
        .long_name = "dump-pr",
        .description = "Emit PR to stdout or a file",
        .value_name = "SPEC",
        .value_optional = true,
    },
    .{
        .long_name = "dump-mlir",
        .description = "Emit the MLIR input passed to the backend",
        .value_name = "PATH",
        .value_optional = true,
    },
    .{
        .long_name = "dump-optimized-hlo",
        .description = "Emit optimized HLO produced by the PJRT backend",
        .value_name = "PATH",
        .value_optional = true,
    },
    .{
        .long_name = "dump-kernels",
        .description = "Emit the kernelization report to stdout or a file",
        .value_name = "PATH",
        .value_optional = true,
    },
    .{
        .long_name = "quiet",
        .description = "Reduce command output",
        .negatable = true,
    },
    .{
        .long_name = "help",
        .short_name = 'h',
        .description = "Show help for the selected command",
    },
};

const format_option = [_]Option{.{
    .long_name = "format",
    .description = "Select the serialized PR rendering format",
    .value_name = "FORMAT",
    .choices = &.{ "zxpr", "json" },
}};

const tvm_target_options = [_]Option{
    .{
        .long_name = "cuda",
        .description = "Select the CUDA target",
        .negatable = true,
    },
    .{
        .long_name = "gpu",
        .description = "Select the CUDA target",
        .negatable = true,
    },
    .{
        .long_name = "cpu",
        .description = "Select the CPU target",
        .negatable = true,
    },
};

const tvm_tune_options = [_]Option{
    .{
        .long_name = "shape",
        .description = "Set matmul dimensions in MxNxK form",
        .value_name = "MxNxK",
    },
    .{
        .long_name = "trials",
        .description = "Set the maximum number of tuning trials",
        .value_name = "COUNT",
    },
    .{
        .long_name = "trials-per-iter",
        .description = "Set the number of trials submitted per tuning iteration",
        .value_name = "COUNT",
    },
} ++ tvm_target_options;

const tvm_symbols_options = [_]Option{.{
    .long_name = "load-compiler",
    .description = "Load the TVM compiler lib in addition to the FFI lib before listing symbols",
    .negatable = true,
}};

const tvm_run_options = [_]Option{.{
    .long_name = "shape",
    .description = "Set matmul dimensions in MxNxK form",
    .value_name = "MxNxK",
}} ++ tvm_target_options;

const tvm_render_options = [_]Option{
    .{
        .long_name = "sweep-palettes",
        .description = "Render with every available color palette",
        .negatable = true,
    },
    .{
        .long_name = "palette",
        .description = "Render with one named color palette",
        .value_name = "NAME",
    },
};

const iree_compile_options = [_]Option{
    .{
        .long_name = "output",
        .description = "Write the VMFB artifact to PATH",
        .value_name = "PATH",
    },
    .{
        .long_name = "target",
        .description = "Select the IREE compilation target",
        .value_name = "NAME",
    },
    .{
        .long_name = "entry",
        .description = "Select the PR function compiled as the entry point",
        .value_name = "NAME",
    },
};

const demo_backend_options = [_]Option{.{
    .long_name = "backend",
    .description = "Select the terminal backend",
    .value_name = "NAME",
    .choices = &.{ "pjrt", "iree" },
}};

const kernel_provider_options = [_]Option{
    .{
        .long_name = "provider",
        .description = "Select comma-separated kernel provider names",
        .value_name = "NAMES",
    },
    .{
        .long_name = "shape",
        .description = "Set matmul dimensions in MxNxK form",
        .value_name = "MxNxK",
    },
    .{
        .long_name = "dtype",
        .description = "Set the candidate region element data type",
        .value_name = "DTYPE",
        .choices = &.{ "f16", "bf16", "f32" },
    },
    .{
        .long_name = "pattern",
        .description = "Select the candidate graph family",
        .value_name = "PATTERN",
        .choices = &.{ "epilogue", "mlp" },
    },
};

const train_options = demo_backend_options ++ [_]Option{
    .{
        .long_name = "warmup",
        .description = "Set the number of warmup iterations",
        .value_name = "COUNT",
    },
    .{
        .long_name = "steps",
        .description = "Set the number of measured training steps",
        .value_name = "COUNT",
    },
};

const path_argument = [_]Positional{.{
    .name = "PATH",
    .description = "Artifact path",
}};

const pr_children = [_]Command{
    .{
        .id = .pr_print_demo,
        .name = "print-demo",
        .summary = "Print the demo program as PR",
    },
    .{
        .id = .pr_render,
        .name = "render",
        .summary = "Render a serialized PR artifact",
        .options = &format_option,
        .positionals = &path_argument,
    },
    .{
        .id = .pr_info,
        .name = "info",
        .summary = "Inspect a serialized PR artifact",
        .positionals = &path_argument,
    },
};

const tvm_children = [_]Command{
    .{
        .id = .tvm_symbols,
        .name = "symbols",
        .summary = "List TVM FFI symbols",
        .options = &tvm_symbols_options,
        .requirement = "TVM",
    },
    .{
        .id = .tvm_check_load,
        .name = "check-load",
        .summary = "Check that the TVM compiler can load",
        .requirement = "TVM",
    },
    .{
        .id = .tvm_tune,
        .name = "tune",
        .summary = "Tune a matmul",
        .options = &tvm_tune_options,
        .requirement = "TVM",
    },
    .{
        .id = .tvm_run,
        .name = "run",
        .summary = "Run a tuned matmul",
        .options = &tvm_run_options,
        .requirement = "TVM",
    },
    .{
        .id = .tvm_render_matmul,
        .name = "render-matmul",
        .summary = "Render TVM-annotated matmul PR",
        .options = &tvm_render_options,
    },
    .{
        .id = .tvm_render_attention,
        .name = "render-attention",
        .summary = "Render TVM-annotated attention PR",
        .options = &tvm_render_options,
    },
};

const iree_children = [_]Command{
    .{
        .id = .iree_compile,
        .name = "compile",
        .summary = "Compile a serialized PR artifact to VMFB",
        .description = "Arguments following -- are passed to iree-compile.",
        .options = &iree_compile_options,
        .positionals = &path_argument,
        .requirement = "IREE and MLIR",
    },
};

const cache_children = [_]Command{
    .{
        .id = .pjrt_cache_save,
        .name = "save",
        .summary = "Write a PJRT executable artifact",
        .positionals = &path_argument,
        .requirement = "PJRT and MLIR",
    },
    .{
        .id = .pjrt_cache_run,
        .name = "run",
        .summary = "Load a PJRT executable artifact",
        .positionals = &path_argument,
        .requirement = "PJRT",
    },
};

const pjrt_children = [_]Command{
    .{
        .id = .pjrt_aot_demo,
        .name = "aot-demo",
        .summary = "Run the PJRT AOT demo",
        .requirement = "PJRT and MLIR",
    },
    .{
        .id = .pjrt_cache,
        .name = "cache",
        .summary = "Operate on PJRT executable artifacts",
        .subcommands = &cache_children,
        .requirement = "PJRT",
    },
};

const demo_children = [_]Command{
    .{
        .id = .demo_basic,
        .name = "basic",
        .summary = "Run the basic matrix computation",
        .options = &demo_backend_options,
        .requirement = "MLIR plus PJRT or IREE",
    },
    .{
        .id = .demo_custom_call_negative,
        .name = "custom-call-negative",
        .summary = "Exercise missing custom-call handling",
        .options = &demo_backend_options,
        .requirement = "MLIR plus PJRT or IREE",
    },
    .{
        .id = .demo_kernel_provider,
        .name = "kernel-provider",
        .summary = "Exercise kernel-provider dispatch",
        .options = &kernel_provider_options,
        .requirement = "PJRT and MLIR",
    },
    .{
        .id = .demo_vjp,
        .name = "vjp",
        .summary = "Run the reverse-mode AD demo",
        .options = &demo_backend_options,
        .requirement = "MLIR plus PJRT or IREE",
    },
    .{
        .id = .demo_train,
        .name = "train",
        .summary = "Run the training demo",
        .options = &train_options,
        .requirement = "MLIR plus PJRT or IREE",
    },
    .{
        .id = .demo_llm_train,
        .name = "llm-train",
        .summary = "Run the small LLM training demo",
        .options = &train_options,
        .requirement = "MLIR plus PJRT or IREE",
    },
};

const root_children = [_]Command{
    .{
        .id = .pr,
        .name = "pr",
        .summary = "Operate on Program Representation artifacts",
        .subcommands = &pr_children,
    },
    .{
        .id = .tvm,
        .name = "tvm",
        .summary = "Exercise the optional TVM integration",
        .subcommands = &tvm_children,
    },
    .{
        .id = .iree,
        .name = "iree",
        .summary = "Exercise the optional IREE integration",
        .subcommands = &iree_children,
    },
    .{
        .id = .pjrt,
        .name = "pjrt",
        .summary = "Exercise the optional PJRT integration",
        .subcommands = &pjrt_children,
    },
    .{
        .id = .demo,
        .name = "demo",
        .summary = "Run executable Zigrad scenarios",
        .subcommands = &demo_children,
    },
};

/// Root of the Zigrad command tree.
pub const root = Command{
    .id = .root,
    .name = "zigrad",
    .summary = "Zigrad core and integration development CLI",
    .description = "Integration commands remain visible when their integration is disabled. Invoking one reports the required build capability.",
    .options = &global_options,
    .subcommands = &root_children,
};

pub fn find_child(command: *const Command, name: []const u8) ?*const Command {
    for (command.subcommands) |*child| {
        if (std.mem.eql(u8, child.name, name)) return child;
    }
    return null;
}

pub fn find_option(command: *const Command, long_name: []const u8) ?*const Option {
    for (command.options) |*option| {
        if (std.mem.eql(u8, option.long_name, long_name)) return option;
    }
    return null;
}

pub fn find_by_id(id: CommandId) ?*const Command {
    return find_by_id_in(&root, id);
}

fn find_by_id_in(command: *const Command, id: CommandId) ?*const Command {
    if (command.id == id) return command;
    for (command.subcommands) |*child| {
        if (find_by_id_in(child, id)) |found| return found;
    }
    return null;
}

/// Resolve the deepest command path present in an argument list.
pub fn command_target(args: []const []const u8) *const Command {
    var command = &root;
    for (args) |arg| {
        if (std.mem.startsWith(u8, arg, "-")) continue;
        if (find_child(command, arg)) |child| command = child;
    }
    return command;
}

/// Resolve the most specific command preceding a help option.
pub fn help_target(args: []const []const u8) ?*const Command {
    var command = &root;
    for (args) |arg| {
        if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help"))
            return command;
        if (std.mem.startsWith(u8, arg, "-")) continue;
        if (find_child(command, arg)) |child| command = child;
    }
    return if (args.len == 0) command else null;
}

test "schema command ids are unique" {
    var seen = std.EnumSet(CommandId).empty;
    try expect_unique_ids(&root, &seen);
    try std.testing.expectEqual(@field(@typeInfo(CommandId), @tagName(@typeInfo(CommandId))).field_names.len, seen.count());
}

fn expect_unique_ids(command: *const Command, seen: *std.EnumSet(CommandId)) !void {
    try std.testing.expect(!seen.contains(command.id));
    seen.insert(command.id);
    for (command.subcommands) |*child| try expect_unique_ids(child, seen);
}
