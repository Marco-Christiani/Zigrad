const std = @import("std");
const contraction = @import("pr").analysis.contraction;
const pattern = @import("pr").analysis.pattern;
const device = @import("device");
const kernel = @import("kernel");
const pr = @import("pr");
const dispatch_mod = @import("dispatch.zig");
const artifact_mod = @import("artifact.zig");
const config = @import("config.zig");
const mirage = @import("../c/mirage/api.zig");
const TypedPtr = @import("rtti").TypedPtr;

const log = std.log.scoped(.@"zg/mirage_provider");
const gated_mlp_op_count: usize = 5;
const max_function_ops: usize = gated_mlp_op_count;

// Alignment required by offsets in Mirage DTensor workspace plans.
//
// TODO(mirage): Read this value from transpile metadata when the C API exposes
//  it.
const workspace_alignment: usize = 128;

pub const MirageProvider = struct {
    /// Runtime state used to dispatch generated CUDA artifacts.
    dispatch_state: *dispatch_mod.MirageDispatchState,
    /// Symbolic-search configuration used during compilation.
    search: SearchOptions,

    pub const SearchOptions = struct {
        /// Named upstream search configuration.
        preset: mirage.SearchPreset = .default,
        /// Search limit in seconds. A non-positive value uses Mirage's default.
        time_limit_seconds: f64 = 0,
    };

    pub const InitOptions = struct {
        /// Mirage adapter loading policy.
        runtime: config.RuntimeConfig = .{},
        /// Symbolic-search policy.
        search: SearchOptions = .{},
    };

    /// Initialize the provider and load its adapter dependency.
    pub fn init(
        /// Runtime state retained by the returned provider.
        dispatch_state: *dispatch_mod.MirageDispatchState,
        /// Adapter loading and symbolic-search configuration.
        options: InitOptions,
    ) mirage.MirageError!MirageProvider {
        try mirage.load_runtime(options.runtime.adapter.path);
        return .{
            .dispatch_state = dispatch_state,
            .search = options.search,
        };
    }

    /// Return a `KernelProvider` backed by this provider state.
    pub fn kernel_provider(self: *MirageProvider) kernel.KernelProvider {
        return .{
            .name = "mirage",
            .compiler = .{
                .context = @ptrCast(self),
                .vtable = &.{
                    .compile = compile_impl,
                    .discover = discover_impl,
                },
            },
            .runtime = .{
                .context = TypedPtr.init(self.dispatch_state),
                .vtable = &.{
                    .dispatch = &dispatch_mod.MirageDispatchState.dispatch,
                },
            },
        };
    }

    fn discover_impl(
        _: *anyopaque,
        func: pr.Function,
        matches: *std.ArrayList(kernel.Match),
        allocator: std.mem.Allocator,
    ) kernel.DiscoverError!void {
        for (0..func.ops.len) |root| {
            if (match_gated_mlp(func, root)) |matched|
                try append_unique_match(matches, allocator, matched);
        }
        for (0..func.ops.len) |start| {
            const matched = match_connected(func, start) orelse continue;
            try append_unique_match(matches, allocator, matched);
        }
    }

    fn match_connected(func: pr.Function, start: usize) ?kernel.Match {
        return pattern.connected_range(func, start, .{
            .accepts = is_supported_op,
            .contains = is_supported_matmul,
            .max_ops = max_function_ops,
        });
    }

    fn compile_impl(ptr: *anyopaque, func: pr.Function, selected_device: device.Device, allocator: std.mem.Allocator) kernel.CompileError!kernel.Artifact {
        const self: *MirageProvider = @ptrCast(@alignCast(ptr));
        return try compile(func, selected_device, allocator, self.search);
    }

    fn compile(
        func: pr.Function,
        selected_device: device.Device,
        allocator: std.mem.Allocator,
        search: SearchOptions,
    ) kernel.CompileError!kernel.Artifact {
        if (!selected_device.platform.eql(.cuda)) return error.Unsupported;
        if (func.ops.len > max_function_ops) {
            log.debug(
                "function '{s}' has {d} ops (> {d}); skipping mirage compile",
                .{ func.name, func.ops.len, max_function_ops },
            );
            return error.Unsupported;
        }

        const graph = mirage.Graph.init() catch |err| return map_mirage_api_error(err);
        defer graph.deinit();

        var tensor_map = std.AutoHashMap(*const pr.Var, mirage.Tensor).init(allocator);
        defer tensor_map.deinit();

        try lower_function_graph(func, graph, &tensor_map);

        return try optimize_and_transpile(
            func.name,
            selected_device,
            allocator,
            graph,
            search,
        );
    }

    /// Optimize the graph and transpile the selected graph to CUDA source.
    fn optimize_and_transpile(
        target_name: []const u8,
        selected_device: device.Device,
        allocator: std.mem.Allocator,
        graph: *mirage.Graph,
        search: SearchOptions,
    ) kernel.CompileError!kernel.Artifact {
        const mirage_device = mirage.Device.init(selected_device.ordinal) catch |err|
            return map_mirage_api_error(err);
        defer mirage_device.deinit();

        const optimize_opts = mirage.OptimizeOptions{
            .preset = search.preset,
            .time_limit_seconds = search.time_limit_seconds,
        };

        const optimized = mirage.optimize(mirage_device, graph, &optimize_opts) catch |err| {
            if (err == error.MirageApiUnsupported) {
                log.debug("mirage symbolic optimization is unsupported for function '{s}'", .{target_name});
                return error.Unsupported;
            }
            if (err == error.MirageNotFound) {
                log.debug("mirage found no optimized graph for function '{s}'", .{target_name});
                return error.Unsupported;
            }
            return map_mirage_api_error(err);
        };
        defer optimized.deinit();

        var source = mirage.transpile(allocator, optimized, null) catch |err| {
            if (err == error.MirageApiUnsupported) {
                log.debug("mirage transpile unsupported for function '{s}'", .{target_name});
                return error.Unsupported;
            }
            return map_mirage_api_error(err);
        };
        defer source.deinit();

        const cuda_code = source.code;
        if (cuda_code.len == 0) {
            log.err("mirage transpile returned empty source for '{s}'", .{target_name});
            return error.ProviderCallFailed;
        }

        const buf_size = source.workspace_size;
        if (buf_size != 0) {
            log.debug("mirage workspace for '{s}': {d} bytes", .{ target_name, buf_size });
        }

        // A source with no custom kernels cannot launch through NVRTC.
        //
        // This occurs for library operations such as a standalone cuBLAS matmul.
        //  Returning Unsupported leaves the operation with the backend.
        const num_kernels = source.kernels.len;
        if (num_kernels == 0) {
            log.debug("mirage transpile produced 0 custom kernels for '{s}'; falling back to backend", .{target_name});
            return error.Unsupported;
        }

        // Filter source for NVRTC (strip host code, replace runtime.h).
        const filtered = try dispatch_mod.filter_source_for_nvrtc(allocator, cuda_code);
        defer allocator.free(filtered);

        // Build kernel descriptors from Layer 2 metadata.
        var kernel_descs = try std.ArrayList(artifact_mod.KernelDesc).initCapacity(allocator, num_kernels);
        defer {
            for (kernel_descs.items) |k| {
                allocator.free(k.args);
                allocator.free(k.func_name);
            }
            kernel_descs.deinit(allocator);
        }

        for (source.kernels) |meta| {
            var args = try allocator.alloc(artifact_mod.KernelArg, meta.args.len);
            errdefer allocator.free(args);
            for (meta.args, 0..) |arg, ai| {
                args[ai] = .{
                    .source = switch (arg.source) {
                        .input => .input,
                        .output => .output,
                        .workspace => .buf,
                    },
                    .index_or_offset = arg.index_or_offset,
                };
            }

            const func_name = try allocator.dupe(u8, meta.func_name);
            errdefer allocator.free(func_name);
            try kernel_descs.append(allocator, .{
                .func_name = func_name,
                .smem_bytes = @intCast(meta.smem_bytes),
                .grid_dim = meta.grid_dim,
                .block_dim = meta.block_dim,
                .args = args,
            });
        }

        // Serialize the artifact.
        const artifact_data = artifact_mod.encode(allocator, .{
            .source = filtered,
            .buf_size = @intCast(buf_size),
            .kernels = kernel_descs.items,
        }) catch {
            log.err("failed to encode artifact for '{s}'", .{target_name});
            return error.ProviderCallFailed;
        };

        return .{
            .data = artifact_data,
            .workspace_bytes = buf_size,
            .workspace_alignment = workspace_alignment,
        };
    }
};

fn append_unique_match(
    matches: *std.ArrayList(kernel.Match),
    allocator: std.mem.Allocator,
    matched: kernel.Match,
) std.mem.Allocator.Error!void {
    for (matches.items) |existing| {
        if (existing.start == matched.start and existing.end == matched.end) return;
    }
    try matches.append(allocator, matched);
}

fn is_supported_matmul(op: *const pr.Op) bool {
    if (op.inputs.len != 2 or op.outputs.len != 1) return false;
    return switch (op.params) {
        .mm, .bmm => true,
        .dot_general => |dg| contraction.is_canonical_batched_matmul(
            dg,
            op.inputs[0].value.as_tensor().shape.rank(),
            op.inputs[1].value.as_tensor().shape.rank(),
        ),
        else => false,
    };
}

fn is_supported_pointwise(op: *const pr.Op) bool {
    if (op.outputs.len != 1) return false;
    return switch (op.params) {
        .exp, .log, .logistic => op.inputs.len == 1,
        .add, .multiply, .divide => op.inputs.len == 2,
        else => false,
    };
}

fn is_supported_op(op: *const pr.Op) bool {
    return is_supported_matmul(op) or is_supported_pointwise(op);
}

fn lower_function_graph(
    func: pr.Function,
    graph: *mirage.Graph,
    tensor_map: *std.AutoHashMap(*const pr.Var, mirage.Tensor),
) kernel.CompileError!void {
    for (func.params) |in_var| {
        const tensor = in_var.as_tensor();

        const dtype = dtype_to_mirage(tensor.dtype) orelse return error.Unsupported;
        if (tensor.shape.dims.len == 0 or tensor.shape.dims.len > mirage.max_rank)
            return error.Unsupported;

        var dims: [mirage.max_rank]i64 = .{ 0, 0, 0, 0 };
        for (tensor.shape.dims, 0..) |dim, idx| {
            if (dim == 0 or dim > std.math.maxInt(i64)) return error.Unsupported;
            dims[idx] = @intCast(dim);
        }

        const spec = mirage.TensorSpec{
            .dtype = dtype,
            .dims = dims[0..tensor.shape.dims.len],
        };

        const handle = graph.new_input(&spec) catch |err| return map_mirage_api_error(err);
        try tensor_map.put(in_var, handle);
    }

    var op_index: usize = 0;
    while (op_index < func.ops.len) {
        if (match_silu(func, op_index)) |matched| {
            const input = tensor_map.get(matched.input) orelse return error.Unsupported;
            const output = try emit_unary(graph, .silu, input);
            try tensor_map.put(matched.output, output);
            op_index += 2;
            continue;
        }

        const op = func.ops[op_index];
        if (op.outputs.len != 1) return error.Unsupported;

        const out_tensor = try lower_op(graph, op, tensor_map);
        try tensor_map.put(op.outputs[0], out_tensor);
        op_index += 1;
    }

    for (func.returns) |out_var| {
        const out_tensor = tensor_map.get(out_var) orelse return error.Unsupported;
        graph.mark_output(out_tensor) catch |err| return map_mirage_api_error(err);
    }
}

const SiluMatch = struct {
    input: *const pr.Var,
    output: *const pr.Var,
};

const gated_mlp_pattern = pattern.Graph{
    .nodes = &.{
        .{ .operation = .{ .primitive = .mm, .input_count = 2, .output_count = 1 } },
        .{ .operation = .{ .primitive = .mm, .input_count = 2, .output_count = 1 } },
        .{
            .operation = .{ .primitive = .logistic, .input_count = 1, .output_count = 1 },
            .inputs = &.{.{ .node_output = .{ .node = 0 } }},
        },
        .{
            .operation = .{ .primitive = .multiply, .input_count = 2, .output_count = 1 },
            .inputs = &.{
                .{ .node_output = .{ .node = 0 } },
                .{ .node_output = .{ .node = 2 } },
            },
            .input_order = .unordered,
        },
        .{
            .operation = .{ .primitive = .multiply, .input_count = 2, .output_count = 1 },
            .inputs = &.{
                .{ .node_output = .{ .node = 3 } },
                .{ .node_output = .{ .node = 1 } },
            },
            .input_order = .unordered,
        },
    },
    .root = 4,
};

fn match_gated_mlp(func: pr.Function, root: usize) ?pattern.Range {
    var captures: [gated_mlp_op_count]?*const pr.Op = undefined;
    if (!gated_mlp_pattern.match(func.ops[root], &captures)) return null;
    return pattern.dense_range(func, &captures);
}

fn match_silu(func: pr.Function, start: usize) ?SiluMatch {
    _ = pattern.sequence(func, start, &.{
        .{ .primitive = .logistic, .input_count = 1, .output_count = 1 },
        .{ .primitive = .multiply, .input_count = 2, .output_count = 1 },
    }) orelse return null;
    const logistic = func.ops[start];
    const multiply = func.ops[start + 1];

    const input = logistic.inputs[0].value;
    const activation = logistic.outputs[0];
    if (activation.only_user() != multiply or
        !pattern.binary_operands(multiply, input, activation, .unordered)) return null;
    return .{ .input = input, .output = multiply.outputs[0] };
}

fn lower_op(
    graph: *mirage.Graph,
    op: *const pr.Op,
    tensor_map: *const std.AutoHashMap(*const pr.Var, mirage.Tensor),
) kernel.CompileError!mirage.Tensor {
    switch (op.params) {
        .mm, .bmm => {
            if (op.inputs.len != 2) return error.Unsupported;
            const lhs = tensor_map.get(op.inputs[0].value) orelse return error.Unsupported;
            const rhs = tensor_map.get(op.inputs[1].value) orelse return error.Unsupported;
            return try emit_matmul(graph, lhs, rhs);
        },
        .dot_general => |dg| {
            if (op.inputs.len != 2) return error.Unsupported;

            const lhs_tensor = op.inputs[0].value.as_tensor();
            const rhs_tensor = op.inputs[1].value.as_tensor();

            if (!contraction.is_canonical_batched_matmul(
                dg,
                lhs_tensor.shape.rank(),
                rhs_tensor.shape.rank(),
            )) return error.Unsupported;

            const lhs = tensor_map.get(op.inputs[0].value) orelse return error.Unsupported;
            const rhs = tensor_map.get(op.inputs[1].value) orelse return error.Unsupported;
            return try emit_matmul(graph, lhs, rhs);
        },
        .exp => {
            if (op.inputs.len != 1) return error.Unsupported;
            const input = tensor_map.get(op.inputs[0].value) orelse return error.Unsupported;
            return try emit_unary(graph, .exp, input);
        },
        .log => {
            if (op.inputs.len != 1) return error.Unsupported;
            const input = tensor_map.get(op.inputs[0].value) orelse return error.Unsupported;
            return try emit_unary(graph, .log, input);
        },
        .add => {
            if (op.inputs.len != 2) return error.Unsupported;
            const lhs = tensor_map.get(op.inputs[0].value) orelse return error.Unsupported;
            const rhs = tensor_map.get(op.inputs[1].value) orelse return error.Unsupported;
            return try emit_binary(graph, .add, lhs, rhs);
        },
        .multiply => {
            if (op.inputs.len != 2) return error.Unsupported;
            const lhs = tensor_map.get(op.inputs[0].value) orelse return error.Unsupported;
            const rhs = tensor_map.get(op.inputs[1].value) orelse return error.Unsupported;
            return try emit_binary(graph, .mul, lhs, rhs);
        },
        .divide => {
            if (op.inputs.len != 2) return error.Unsupported;
            const lhs = tensor_map.get(op.inputs[0].value) orelse return error.Unsupported;
            const rhs = tensor_map.get(op.inputs[1].value) orelse return error.Unsupported;
            return try emit_binary(graph, .div, lhs, rhs);
        },
        else => return error.Unsupported,
    }
}

fn emit_matmul(graph: *mirage.Graph, lhs: mirage.Tensor, rhs: mirage.Tensor) kernel.CompileError!mirage.Tensor {
    return graph.matmul(lhs, rhs) catch |err| {
        log.err("mirage graph.matmul rejected (lhs={d}, rhs={d}): {s}", .{ lhs.index, rhs.index, @errorName(err) });
        return map_mirage_api_error(err);
    };
}

fn emit_unary(
    graph: *mirage.Graph,
    op: mirage.UnaryOp,
    input: mirage.Tensor,
) kernel.CompileError!mirage.Tensor {
    return graph.unary(op, input) catch |err| {
        log.err("mirage graph.unary({s}) rejected (input={d}): {s}", .{ @tagName(op), input.index, @errorName(err) });
        return map_mirage_api_error(err);
    };
}

fn emit_binary(
    graph: *mirage.Graph,
    op: mirage.BinaryOp,
    lhs: mirage.Tensor,
    rhs: mirage.Tensor,
) kernel.CompileError!mirage.Tensor {
    return graph.binary(op, lhs, rhs) catch |err| {
        log.err("mirage graph.binary({s}) rejected (lhs={d}, rhs={d}): {s}", .{ @tagName(op), lhs.index, rhs.index, @errorName(err) });
        return map_mirage_api_error(err);
    };
}

fn dtype_to_mirage(dtype: pr.DType) ?mirage.DType {
    return switch (dtype) {
        .f16 => .f16,
        .bf16 => .bf16,
        .f32 => .f32,
        .f64 => .f64,
        else => null,
    };
}

test dtype_to_mirage {
    try std.testing.expectEqual(mirage.DType.f16, dtype_to_mirage(.f16));
    try std.testing.expectEqual(mirage.DType.bf16, dtype_to_mirage(.bf16));
    try std.testing.expectEqual(mirage.DType.f32, dtype_to_mirage(.f32));
    try std.testing.expectEqual(mirage.DType.f64, dtype_to_mirage(.f64));
    try std.testing.expectEqual(@as(?mirage.DType, null), dtype_to_mirage(.i32));
}

fn map_mirage_api_error(err: mirage.MirageError) kernel.CompileError {
    return switch (err) {
        error.MirageUnavailable => error.ProviderLoadFailed,
        error.MirageInvalidArgument => error.Unsupported,
        error.MirageInternalError => error.ProviderCallFailed,
        error.MirageApiUnsupported => error.Unsupported,
        error.MirageNotFound => error.Unsupported,
        error.OutOfMemory => error.OutOfMemory,
    };
}

test "Mirage discovery grows a connected supported region" {
    const testing = std.testing;
    var program = pr.Program.init(testing.allocator);
    defer program.deinit();
    var builder = try pr.FunctionBuilder.init(&program, "main");
    defer builder.deinit();
    const lhs = try builder.param_tensor(.f32, &.{ 4, 8 });
    const rhs = try builder.param_tensor(.f32, &.{ 8, 2 });
    const bias = try builder.param_tensor(.f32, &.{ 4, 2 });
    const mm = try builder.mm(lhs, rhs);
    const sum = try builder.add(mm, bias);
    const output = try builder.exp(sum);
    const func = try builder.finish(.{ .returns = &.{output} });

    const matched = MirageProvider.match_connected(func, 0) orelse
        return error.TestUnexpectedResult;
    try testing.expectEqual(@as(usize, 3), matched.len());
}

test "Mirage discovery covers pointwise prefixes and connected branches" {
    const testing = std.testing;

    {
        var program = pr.Program.init(testing.allocator);
        defer program.deinit();
        var builder = try pr.FunctionBuilder.init(&program, "prefix");
        defer builder.deinit();
        const lhs = try builder.param_tensor(.f32, &.{ 4, 8 });
        const rhs = try builder.param_tensor(.f32, &.{ 8, 2 });
        const transformed = try builder.exp(lhs);
        const output = try builder.mm(transformed, rhs);
        const func = try builder.finish(.{ .returns = &.{output} });

        const matched = MirageProvider.match_connected(func, 0) orelse
            return error.TestUnexpectedResult;
        try testing.expectEqual(@as(usize, 2), matched.len());
    }

    {
        var program = pr.Program.init(testing.allocator);
        defer program.deinit();
        var builder = try pr.FunctionBuilder.init(&program, "branch");
        defer builder.deinit();
        const lhs = try builder.param_tensor(.f32, &.{ 4, 8 });
        const rhs = try builder.param_tensor(.f32, &.{ 8, 2 });
        const mm = try builder.mm(lhs, rhs);
        const exponent = try builder.exp(mm);
        const logarithm = try builder.log(mm);
        const output = try builder.add(exponent, logarithm);
        const func = try builder.finish(.{ .returns = &.{output} });

        const matched = MirageProvider.match_connected(func, 0) orelse
            return error.TestUnexpectedResult;
        try testing.expectEqual(@as(usize, 4), matched.len());
    }
}

test "Mirage recognizes a SiLU composite" {
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();
    var builder = try pr.FunctionBuilder.init(&program, "silu");
    defer builder.deinit();
    const input = try builder.param_tensor(.f32, &.{ 4, 8 });
    const activation = try builder.logistic(input);
    const output = try builder.multiply(input, activation);
    const func = try builder.finish(.{ .returns = &.{output} });

    const matched = match_silu(func, 0) orelse return error.TestUnexpectedResult;
    try testing.expect(matched.input == input);
    try testing.expect(matched.output == output);
}

test "Mirage recognizes a gated MLP branch and join" {
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();
    var builder = try pr.FunctionBuilder.init(&program, "gated_mlp");
    defer builder.deinit();
    const input = try builder.param_tensor(.f16, &.{ 8, 4096 });
    const gate_weight = try builder.param_tensor(.f16, &.{ 4096, 4096 });
    const up_weight = try builder.param_tensor(.f16, &.{ 4096, 4096 });
    const gate = try builder.mm(input, gate_weight);
    const up = try builder.mm(input, up_weight);
    const activation = try builder.logistic(gate);
    const silu = try builder.multiply(gate, activation);
    const output = try builder.multiply(silu, up);
    const func = try builder.finish(.{ .returns = &.{output} });

    const matched = match_gated_mlp(func, func.ops.len - 1) orelse
        return error.TestUnexpectedResult;
    try testing.expectEqual(@as(usize, 5), matched.len());

    var matches = std.ArrayList(kernel.Match).empty;
    defer matches.deinit(testing.allocator);
    try MirageProvider.discover_impl(undefined, func, &matches, testing.allocator);
    var found_full = false;
    var single_matmuls: usize = 0;
    for (matches.items) |candidate| {
        found_full = found_full or candidate.start == matched.start and candidate.end == matched.end;
        if (candidate.len() == 1) single_matmuls += 1;
    }
    try testing.expect(found_full);
    try testing.expectEqual(@as(usize, 2), single_matmuls);
}

test "Mirage gated MLP matching follows dataflow across branch order" {
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();
    var builder = try pr.FunctionBuilder.init(&program, "gated_mlp_swapped");
    defer builder.deinit();
    const input = try builder.param_tensor(.f16, &.{ 8, 4096 });
    const gate_weight = try builder.param_tensor(.f16, &.{ 4096, 4096 });
    const up_weight = try builder.param_tensor(.f16, &.{ 4096, 4096 });
    const up = try builder.mm(input, up_weight);
    const gate = try builder.mm(input, gate_weight);
    const activation = try builder.logistic(gate);
    const silu = try builder.multiply(activation, gate);
    const output = try builder.multiply(up, silu);
    const func = try builder.finish(.{ .returns = &.{output} });

    const matched = match_gated_mlp(func, func.ops.len - 1) orelse
        return error.TestUnexpectedResult;
    try testing.expectEqual(pattern.Range{ .start = 0, .end = 5 }, matched);
}
