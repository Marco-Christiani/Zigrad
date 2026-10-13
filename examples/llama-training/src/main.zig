//! LLaMA 3.2 training and inference through selectable Zigrad backends.

const std = @import("std");
const zg = @import("zigrad");

const training = @import("training.zig");
const log = std.log.scoped(.@"zg/example/llama_training");

const default_weights_path = "./weights/llama-3.2-1b-instruct/model.safetensors";

const BackendKind = enum {
    pjrt,
    iree,
};

pub const std_options: std.Options = .{
    .log_level = .info,
};

pub fn main(init: std.process.Init) !void {
    const env = zg.RuntimeEnv.from_init(init);
    const args = try init.minimal.args.toSlice(env.allocator);
    defer env.allocator.free(args);

    const output = observation_output(args[1..]);
    execute(env, args[1..]) catch |err| {
        log.err("llama-training: {s}", .{@errorName(err)});
        if (output) |path| {
            try training.write_observation(env.io, env.allocator, path, .{ .result = .{ .refused = .{
                .code = training.refusal_code(err),
                .message = @errorName(err),
            } } });
        }
        return err;
    };
}

const ParsedArgs = struct {
    backend: BackendKind = .pjrt,
    options: training.Options,
    observation: ?training.ObservationArgs = null,
    explicit_weights: bool = false,
    help: bool = false,
};

fn observation_output(args: []const []const u8) ?[]const u8 {
    for (args) |arg| if (value(arg, "--observation-output=")) |path| return path;
    return null;
}

fn parse_observation_args(args: []const []const u8, default_weights: []const u8) !ParsedArgs {
    var parsed = ParsedArgs{ .options = .{ .weights_path = default_weights } };
    var manifest: ?[]const u8 = null;
    var hash: ?[]const u8 = null;
    var output: ?[]const u8 = null;
    var explicit = std.StaticBitSet(7).empty;
    var seen = std.StringHashMap(void).init(std.heap.page_allocator);
    defer seen.deinit();
    for (args) |arg| {
        const key = arg[0 .. std.mem.indexOfScalar(u8, arg, '=') orelse arg.len];
        const entry = try seen.getOrPut(key);
        if (entry.found_existing) {
            if (@import("builtin").is_test) log.debug("duplicate argument '{s}'", .{key}) else log.err("duplicate argument '{s}'", .{key});
            return error.InvalidArgument;
        }
        if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            parsed.help = true;
        } else if (value(arg, "--backend=")) |name| {
            parsed.backend = std.meta.stringToEnum(BackendKind, name) orelse return error.InvalidArgument;
            explicit.set(0);
        } else if (value(arg, "--mode=")) |name| {
            parsed.options.mode = std.meta.stringToEnum(training.Mode, name) orelse return error.InvalidArgument;
            explicit.set(1);
        } else if (value(arg, "--dtype=")) |name| {
            const dtype = std.meta.stringToEnum(zg.DType, name) orelse return error.InvalidArgument;
            parsed.options.dtype = switch (dtype) {
                .bf16, .f16, .f32, .f64 => dtype,
                else => return error.InvalidArgument,
            };
            explicit.set(2);
        } else if (value(arg, "--sequence-length=")) |raw| {
            parsed.options.sequence_length = std.fmt.parseInt(i64, raw, 10) catch return error.InvalidArgument;
            explicit.set(3);
        } else if (value(arg, "--batch=")) |raw| {
            parsed.options.batch = std.fmt.parseInt(i64, raw, 10) catch return error.InvalidArgument;
            explicit.set(4);
        } else if (value(arg, "--warmup-steps=")) |raw| {
            parsed.options.warmup_steps = std.fmt.parseInt(usize, raw, 10) catch return error.InvalidArgument;
            explicit.set(5);
        } else if (value(arg, "--steps=")) |raw| {
            parsed.options.steps = std.fmt.parseInt(usize, raw, 10) catch return error.InvalidArgument;
            explicit.set(6);
        } else if (value(arg, "--weights=")) |path| {
            parsed.options.weights_path = path;
            parsed.explicit_weights = true;
        } else if (value(arg, "--input-manifest=")) |path| {
            manifest = path;
        } else if (value(arg, "--input-manifest-sha256=")) |sha| {
            hash = sha;
        } else if (value(arg, "--observation-output=")) |path| {
            output = path;
        } else if (std.mem.eql(u8, arg, "--quiet")) {
            parsed.options.quiet = true;
        } else {
            log.err("unknown argument '{s}'", .{arg});
            return error.InvalidArgument;
        }
    }
    if (manifest != null or hash != null or output != null) {
        if (manifest == null or hash == null or output == null or manifest.?.len == 0 or hash.?.len == 0 or output.?.len == 0) {
            if (@import("builtin").is_test) log.debug("observation requires input manifest, manifest SHA256, and output path", .{}) else log.err("observation requires input manifest, manifest SHA256, and output path", .{});
            return error.InvalidArgument;
        }
        const w = training.checked_workload;
        if ((explicit.isSet(0) and parsed.backend != .pjrt) or (explicit.isSet(1) and parsed.options.mode != .training) or
            (explicit.isSet(2) and parsed.options.dtype != .f32) or (explicit.isSet(3) and parsed.options.sequence_length != w.sequence_length) or
            (explicit.isSet(4) and parsed.options.batch != w.batch_size) or (explicit.isSet(5) and parsed.options.warmup_steps != w.untimed_updates) or
            (explicit.isSet(6) and parsed.options.steps != 1))
        {
            if (@import("builtin").is_test) log.debug("explicit CLI setting conflicts with checked observation policy", .{}) else log.err("explicit CLI setting conflicts with checked observation policy", .{});
            return error.InvalidArgument;
        }
        parsed.observation = .{ .input_manifest = manifest.?, .input_manifest_sha256 = hash.?, .output = output.? };
        parsed.options.mode = .training;
        parsed.options.dtype = .f32;
        parsed.options.batch = w.batch_size;
        parsed.options.sequence_length = w.sequence_length;
        parsed.options.warmup_steps = w.untimed_updates;
        parsed.options.steps = 1;
    }
    return parsed;
}

fn execute(env: zg.RuntimeEnv, args: []const []const u8) !void {
    var parsed = try parse_observation_args(args, env.environ.get("ZG_LLAMA_SAFETENSORS_PATH") orelse default_weights_path);
    if (parsed.help) {
        std.debug.print("{s}", .{usage});
        return;
    }
    if (parsed.observation) |request| {
        var input = try training.load_input_manifest(env.io, env.allocator, request, if (parsed.explicit_weights) parsed.options.weights_path else null);
        defer input.deinit();
        parsed.options.weights_path = input.checkpoint_path;
        return run_pjrt(env, parsed.options, &input, request.output);
    }
    return switch (parsed.backend) {
        .pjrt => run_pjrt(env, parsed.options, null, null),
        .iree => if (comptime zg.build_options.has_iree) run_iree(env, parsed.options) else unavailable("IREE", "-Diree=true"),
    };
}

fn run_pjrt(env: zg.RuntimeEnv, options: training.Options, input: ?*const training.CheckedInput, output: ?[]const u8) !void {
    if (comptime !zg.build_options.has_pjrt) return unavailable("PJRT", "-Dpjrt=true");
    const plugin_path = env.environ.get("PJRT_PLUGIN_PATH") orelse {
        log.err("set PJRT_PLUGIN_PATH to a PJRT plugin", .{});
        return error.MissingPlugin;
    };
    const client_options = try zg.pjrt.config.from_environ(env.environ);
    var client = try zg.pjrt.Client.init(env.allocator, plugin_path, client_options);
    defer client.deinit();
    const devices = try client.get_devices(env.allocator);
    defer env.allocator.free(devices);
    if (devices.len == 0) return error.NoDevices;

    var execution = try zg.pjrt.Execution.init(&client, devices[0], .{});
    var backend = zg.pjrt.Backend.init(&execution, .{});
    var ctx = zg.CompilationCtx{
        .allocator = env.allocator,
        .io = env.io,
        .device = execution.interface.device,
    };
    var run_options = options;
    if (input) |checked| {
        if (!client.platform.eql(.cpu) or !execution.interface.device.platform.eql(.cpu)) {
            log.err("checked observation requires a CPU client and executor", .{});
            return error.UnsupportedDevice;
        }
        run_options.observation = .{ .input = checked, .input_readiness = .{ .context = &client, .wait = wait_uploaded_inputs }, .output = output.?, .device = .{
            .platform_name = client.platform.name,
            .device_kind = try client.device_kind(&devices[0]),
            .ordinal = try devices[0].get_id(client.api),
        } };
    }
    return try training.run(&ctx, &backend.interface, run_options);
}

fn wait_uploaded_inputs(context: *anyopaque, buffers: []const zg.Executor.Buffer) !void {
    const client: *zg.pjrt.Client = @ptrCast(@alignCast(context));
    for (buffers) |buffer| {
        var raw = zg.pjrt.Buffer{ .pjrt_buffer = @ptrCast(@alignCast(buffer.handle)) };
        var ready = try raw.ready_event(client.api);
        defer client.deinit_event(&ready);
        try client.await_event(&ready);
    }
}

fn run_iree(env: zg.RuntimeEnv, options: training.Options) !void {
    const config = zg.iree.Config.from_environ(env.environ);
    var runtime = try zg.iree.Runtime.init(env.allocator, .registered, config.runtime);
    defer runtime.deinit();
    var execution = zg.iree.Execution.init(env.allocator, &runtime, config.runtime);
    var backend = zg.iree.Backend.init(&execution, config.compiler, "module.main");
    var ctx = zg.CompilationCtx{
        .allocator = env.allocator,
        .io = env.io,
        .device = execution.interface.device,
    };
    return try training.run(&ctx, &backend.interface, options);
}

fn unavailable(comptime name: []const u8, comptime build_option: []const u8) error{BackendUnavailable} {
    log.err("{s} is unavailable, rebuild with {s}", .{ name, build_option });
    return error.BackendUnavailable;
}

fn value(argument: []const u8, prefix: []const u8) ?[]const u8 {
    if (!std.mem.startsWith(u8, argument, prefix)) return null;
    return argument[prefix.len..];
}

const usage =
    \\Usage: llama-training [options]
    \\
    \\Options:
    \\  --backend=pjrt|iree
    \\  --mode=training|inference
    \\  --dtype=bf16|f16|f32|f64
    \\  --sequence-length=N
    \\  --batch=N
    \\  --warmup-steps=N
    \\  --steps=N
    \\  --weights=PATH
    \\  --input-manifest=PATH
    \\  --input-manifest-sha256=SHA256
    \\  --observation-output=PATH
    \\  --quiet
    \\  --help, -h
    \\
    \\The default checkpoint path is ./weights/llama-3.2-1b-instruct/model.safetensors.
    \\ZG_LLAMA_SAFETENSORS_PATH overrides it when --weights is absent.
    \\PJRT requires PJRT_PLUGIN_PATH. IREE reads ZG_IREE_* configuration.
;

test parse_observation_args {
    const complete = [_][]const u8{ "--input-manifest=input.json", "--input-manifest-sha256=hash", "--observation-output=result.json" };
    const parsed = try parse_observation_args(&complete, "weights");
    try std.testing.expectEqual(zg.DType.f32, parsed.options.dtype);
    try std.testing.expectEqual(training.checked_workload.untimed_updates, parsed.options.warmup_steps);
    try std.testing.expectError(error.InvalidArgument, parse_observation_args(complete[0..2], "weights"));
    try std.testing.expectError(error.InvalidArgument, parse_observation_args(&(complete ++ .{"--dtype=bf16"}), "weights"));
    try std.testing.expectError(error.InvalidArgument, parse_observation_args(&(complete ++ .{"--batch=2"}), "weights"));
    try std.testing.expectError(error.InvalidArgument, parse_observation_args(&(complete ++ .{"--warmup-steps=1"}), "weights"));
    try std.testing.expectError(error.InvalidArgument, parse_observation_args(&(complete ++ .{"--input-manifest=again"}), "weights"));
}
