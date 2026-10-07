//! Automatic differentiation transforms on PR functions.
//!
//! Linearization produces an augmented primal and a residualized linear
//!  function. JVP applies that linear function. VJP transposes it.
//!
//! JVP and VJP register the complete derived call graph in the supplied
//!  program and return the derived callable's identity.
//!
//! Generated functions are ordinary PR functions and may be differentiated
//!  again. For a scalar $f$ on a Euclidean space, holding the VJP seed at
//!  one gives the Hessian-vector product
//!
//! $$
//! \operatorname{jvp}_{x \mapsto \operatorname{vjp}_f(x, 1)}(x, v)
//! = H_f(x)v.
//! $$
//!
//! Higher-level traced transforms live in `transforms.zig`.
const std = @import("std");

const pr = @import("pr.zig");
const ops = @import("ops/ops.zig");

pub const AdError = ops.types.AdError || pr.FunctionRegistrationError || error{
    /// PR differentiation does not transform recursive call graphs.
    RecursiveDifferentiationUnsupported,
};

/// Options for `vjp`.
pub const VjpOpts = struct {
    /// Select source-function outputs whose cotangents seed the VJP.
    ///
    /// `null` selects every output whose dtype has a standard cotangent. A
    ///  provided slice must be nonempty and contain only such outputs. The
    ///  transformed function accepts seeds in this slice's order. Duplicate
    ///  indices contribute independent seeds to the same output cotangent.
    of: ?[]const usize = null,

    /// Select primal-input cotangents returned by the transformed function.
    ///
    /// `null` selects every input whose dtype has a standard cotangent. A
    ///  provided slice may contain only such inputs. The output order matches
    ///  this slice, including duplicate indices.
    wrt: ?[]const usize = null,

    /// Include source-function outputs before the selected input cotangents.
    ///
    /// When enabled, the transformed codomain is
    ///
    /// $$
    /// (Y_1 \times \cdots \times Y_m)
    /// \times (T^*_{x_{w_1}}X_{w_1} \times \cdots
    /// \times T^*_{x_{w_k}}X_{w_k}).
    /// $$
    include_primal_outputs: bool = false,
};

/// Options for `jvp`.
pub const JvpOpts = struct {
    /// Include source-function outputs before the differentiable output tangents.
    ///
    /// When enabled, the transformed codomain is
    ///
    /// $$
    /// (Y_1 \times \cdots \times Y_m)
    /// \times (T_{f_{e_1}(x)}Y_{e_1} \times \cdots
    /// \times T_{f_{e_l}(x)}Y_{e_l}),
    /// $$
    ///
    /// where \(E = (e_1, \ldots, e_l)\) contains the output indices whose
    /// dtypes have standard dual values.
    include_primal_outputs: bool = false,
};

/// Functions and output mapping produced by linearizing one PR function.
///
/// Function identities and the tangent-index map have the program's lifetime.
pub const Linearization = struct {
    /// Accepts source parameters and returns source results followed by residuals.
    augmented_primal: pr.FunctionId,
    /// Accepts differentiable source-parameter tangents followed by residuals.
    linear: pr.FunctionId,
    /// Number of augmented-primal results consumed as linear-function residuals.
    residual_count: usize,
    /// Maps each source parameter to its linear-function tangent parameter.
    ///
    /// `null` means the source parameter's dtype has no dual values.
    input_tangent_indices: []const ?usize,
    /// Maps each source result to its linear-function tangent result.
    ///
    /// `null` means the result has no standard dual or its tangent is
    ///  structurally zero. JVP materializes zeros only for differentiable
    ///  results.
    output_tangent_indices: []const ?usize,
};

const ResidualCandidate = struct {
    primal: *pr.Var,
    linear_param: *pr.Var,
};

/// State for replaying primal operations into one derived function.
const PrimalContext = struct {
    builder: *pr.FunctionBuilder,
    primals: []?*pr.Var,
    allocator: std.mem.Allocator,

    fn primal(self: PrimalContext, source: *const pr.Var) error{MissingPrimal}!*pr.Var {
        std.debug.assert(source.id < self.primals.len);
        return self.primals[source.id] orelse error.MissingPrimal;
    }

    fn set_primal(self: PrimalContext, source: *const pr.Var, value: *pr.Var) void {
        std.debug.assert(source.id < self.primals.len);
        self.primals[source.id] = value;
    }
};

const GeneratedLinearization = struct {
    source: pr.FunctionId,
    linearization: Linearization,
};

fn pop_active(active: *std.ArrayList(pr.FunctionId), expected: pr.FunctionId) void {
    const source = active.pop() orelse unreachable;
    std.debug.assert(source == expected);
}

const LinearizationTraversal = struct {
    allocator: std.mem.Allocator,
    program: *pr.Program,
    generated: std.ArrayList(GeneratedLinearization) = .empty,
    active: std.ArrayList(pr.FunctionId) = .empty,

    fn deinit(self: *LinearizationTraversal) void {
        self.generated.deinit(self.allocator);
        self.active.deinit(self.allocator);
        self.* = undefined;
    }

    fn get_or_create(self: *LinearizationTraversal, source: pr.FunctionId) AdError!Linearization {
        for (self.generated.items) |entry| {
            if (entry.source == source) return entry.linearization;
        }
        for (self.active.items) |active| {
            if (active == source) return error.RecursiveDifferentiationUnsupported;
        }

        // The active stack rejects direct and mutually recursive call graphs.
        try self.active.append(self.allocator, source);
        defer pop_active(&self.active, source);

        const func = self.program.get_function_by_id(source) orelse
            return error.CallUnresolvedCallee;
        const base_name = try self.allocator.print("{s}_linearized", .{func.name});
        defer self.allocator.free(base_name);
        const result = try linearize_impl(self, func, base_name);
        try self.generated.append(self.allocator, .{
            .source = source,
            .linearization = result,
        });
        return result;
    }
};

/// Linearize `func` into an augmented primal and a residualized linear map.
///
/// For $f: X \to Y$, the transform chooses a residual space $R$ and
///  constructs
///
/// $$
/// p: X \to Y \times R,
/// \qquad
/// p(x) = \left(f(x), r(x)\right),
/// $$
///
/// and
///
/// $$
/// \ell: T_xX \times R \to T_{f(x)}Y,
/// \qquad
/// \ell\left(v, r(x)\right) = \mathrm{d}f_x(v).
/// $$
///
/// `Linearization.augmented_primal` identifies $p$, and
///  `Linearization.linear` identifies $\ell$. Structurally zero output
///  tangents may be omitted from $\ell$; `output_tangent_indices` maps the
///  retained results back to the outputs of $f$.
///
/// The generated functions are registered in `program`. Their names are
///  derived from `name` and made unique within the program. Generated
///  functions for callees are shared within this traversal.
pub fn linearize(
    allocator: std.mem.Allocator,
    program: *pr.Program,
    source: pr.FunctionId,
    name: []const u8,
) AdError!Linearization {
    var traversal = LinearizationTraversal{
        .allocator = allocator,
        .program = program,
    };
    defer traversal.deinit();

    const func = program.get_function_by_id(source) orelse
        return error.CallUnresolvedCallee;
    try traversal.active.append(allocator, source);
    defer pop_active(&traversal.active, source);
    return try linearize_impl(&traversal, func, name);
}

fn linearize_impl(
    traversal: *LinearizationTraversal,
    func: pr.Function,
    name: []const u8,
) AdError!Linearization {
    const allocator = traversal.allocator;
    const program = traversal.program;
    const saved = program.checkpoint_appends();
    errdefer program.restore_appends(saved);

    const primal_name_base = try allocator.print("{s}_primal", .{name});
    defer allocator.free(primal_name_base);
    const primal_name = try program.reserve_unique_function_name(primal_name_base);
    const linear_name_base = try allocator.print("{s}_linear", .{name});
    defer allocator.free(linear_name_base);
    const linear_name = try program.reserve_unique_function_name(linear_name_base);

    var primal_builder = try pr.FunctionBuilder.init(program, primal_name);
    defer primal_builder.deinit();
    var linear_builder = try pr.FunctionBuilder.init(program, linear_name);
    defer linear_builder.deinit();

    // Each source value may have a primal value in the augmented function,
    //  a residual parameter in the linear function, and a tangent value.
    const primal_map = try allocator.alloc(?*pr.Var, func.var_count);
    defer allocator.free(primal_map);
    @memset(primal_map, null);
    const linear_primal_map = try allocator.alloc(?*pr.Var, func.var_count);
    defer allocator.free(linear_primal_map);
    @memset(linear_primal_map, null);
    const tangent_map = try allocator.alloc(?*pr.Var, func.var_count);
    defer allocator.free(tangent_map);
    @memset(tangent_map, null);

    var residuals: std.ArrayList(ResidualCandidate) = .empty;
    defer residuals.deinit(allocator);
    const tangent_params = try allocator.alloc(*pr.Var, func.params.len);
    defer allocator.free(tangent_params);
    const input_tangent_indices = try program.allocator().alloc(?usize, func.params.len);
    var tangent_param_count: usize = 0;

    for (func.params, 0..) |source_param, index| {
        const primal_param = try primal_builder.param_like(source_param.aval);
        primal_map[source_param.id] = primal_param;

        if (ops.types.is_differentiable(source_param.aval)) {
            const tangent_param = try linear_builder.param_like(source_param.aval);
            input_tangent_indices[index] = tangent_param_count;
            tangent_params[tangent_param_count] = tangent_param;
            tangent_param_count += 1;
            tangent_map[source_param.id] = tangent_param;
        } else {
            input_tangent_indices[index] = null;
        }

        const residual_param = try linear_builder.param_like(source_param.aval);
        linear_primal_map[source_param.id] = residual_param;
        try residuals.append(allocator, .{
            .primal = primal_param,
            .linear_param = residual_param,
        });
    }

    const primal_ctx = PrimalContext{
        .builder = &primal_builder,
        .primals = primal_map,
        .allocator = allocator,
    };
    const linear_ctx = ops.types.JvpContext{
        .builder = &linear_builder,
        .primals = linear_primal_map,
        .tangents = tangent_map,
        .allocator = allocator,
    };

    for (func.ops) |op| {
        switch (op.params) {
            .call => try linearize_call(
                traversal,
                primal_ctx,
                linear_ctx,
                &residuals,
                op,
            ),
            else => {
                try replay_primal(primal_ctx, op);
                for (op.outputs) |source_output| {
                    const primal = try primal_ctx.primal(source_output);
                    const residual_param = try linear_builder.param_like(source_output.aval);
                    linear_ctx.set_primal(source_output, residual_param);
                    try residuals.append(allocator, .{
                        .primal = primal,
                        .linear_param = residual_param,
                    });
                }
                try ops.jvp(linear_ctx, op);
            },
        }
    }

    // The linear function only returns tangents that have PR values. Keep the
    //  source-result mapping so wrappers can omit discrete results and
    //  materialize differentiable structural zeros.
    const output_tangent_indices = try program.allocator().alloc(?usize, func.returns.len);
    const linear_returns = try allocator.alloc(*pr.Var, func.returns.len);
    defer allocator.free(linear_returns);
    var linear_return_count: usize = 0;
    for (func.returns, output_tangent_indices) |source_return, *tangent_index| {
        if (linear_ctx.tangent(source_return)) |tangent| {
            tangent_index.* = linear_return_count;
            linear_returns[linear_return_count] = tangent;
            linear_return_count += 1;
        } else {
            tangent_index.* = null;
        }
    }

    // Residual parameters are provisional. Their use lists identify the
    //  primal values the completed linear function actually needs.
    var used_residual_count: usize = 0;
    for (residuals.items) |candidate| {
        if (candidate.linear_param.first_use != null) used_residual_count += 1;
    }
    const linear_params = try allocator.alloc(
        *pr.Var,
        tangent_param_count + used_residual_count,
    );
    defer allocator.free(linear_params);
    @memcpy(linear_params[0..tangent_param_count], tangent_params[0..tangent_param_count]);
    var residual_index: usize = 0;
    for (residuals.items) |candidate| {
        if (candidate.linear_param.first_use == null) continue;
        linear_params[tangent_param_count + residual_index] = candidate.linear_param;
        residual_index += 1;
    }
    std.debug.assert(residual_index == used_residual_count);

    const linear_func = try linear_builder.finish(.{
        .returns = linear_returns[0..linear_return_count],
        .parameters = .{ .selected = linear_params },
    });

    const primal_returns = try allocator.alloc(
        *pr.Var,
        func.returns.len + used_residual_count,
    );
    defer allocator.free(primal_returns);
    for (func.returns, primal_returns[0..func.returns.len]) |source_return, *output| {
        output.* = try primal_ctx.primal(source_return);
    }
    residual_index = 0;
    for (residuals.items) |candidate| {
        if (candidate.linear_param.first_use == null) continue;
        primal_returns[func.returns.len + residual_index] = candidate.primal;
        residual_index += 1;
    }
    std.debug.assert(residual_index == used_residual_count);

    const primal_func = try primal_builder.finish(.{ .returns = primal_returns });
    const primal_id = try program.add_function(primal_func);
    const linear_id = try program.add_function(linear_func);
    return .{
        .augmented_primal = primal_id,
        .linear = linear_id,
        .residual_count = used_residual_count,
        .input_tangent_indices = input_tangent_indices,
        .output_tangent_indices = output_tangent_indices,
    };
}

fn replay_primal(ctx: PrimalContext, op: *const pr.Op) AdError!void {
    const inputs = try ctx.allocator.alloc(*pr.Var, op.inputs.len);
    defer ctx.allocator.free(inputs);
    for (op.inputs, inputs) |operand, *input| {
        input.* = try ctx.primal(operand.value);
    }

    const replayed = try ctx.builder.replay_op(op, inputs);
    if (replayed.outputs.len != op.outputs.len) return error.InvalidLinearization;
    for (op.outputs, replayed.outputs) |source, output| ctx.set_primal(source, output);
}

/// Emit an augmented-primal call and validate its linearization contract.
///
/// The returned operation-output slice belongs to the builder's program.
fn call_augmented_primal(
    program: *pr.Program,
    builder: *pr.FunctionBuilder,
    /// Function from which `linearization` was derived.
    source: pr.Function,
    /// Linearization whose augmented primal is called.
    linearization: Linearization,
    /// Arguments to the source function in parameter order.
    inputs: []const *pr.Var,
) AdError![]*pr.Var {
    const augmented = program.get_function_by_id(linearization.augmented_primal) orelse
        return error.CallUnresolvedCallee;
    const outputs = (try builder.call(linearization.augmented_primal, inputs)).outputs;
    if (outputs.len != augmented.returns.len or
        outputs.len != source.returns.len + linearization.residual_count)
        return error.InvalidLinearization;
    return outputs;
}

fn linearize_call(
    traversal: *LinearizationTraversal,
    primal_ctx: PrimalContext,
    linear_ctx: ops.types.JvpContext,
    residuals: *std.ArrayList(ResidualCandidate),
    op: *const pr.Op,
) AdError!void {
    const callee_id = op.params.call.callee;
    const callee = traversal.program.get_function_by_id(callee_id) orelse
        return error.CallUnresolvedCallee;
    const callee_linearization = try traversal.get_or_create(callee_id);

    const primal_inputs = try traversal.allocator.alloc(*pr.Var, op.inputs.len);
    defer traversal.allocator.free(primal_inputs);
    for (op.inputs, primal_inputs) |operand, *input| {
        input.* = try primal_ctx.primal(operand.value);
    }
    const primal_outputs = try call_augmented_primal(
        traversal.program,
        primal_ctx.builder,
        callee,
        callee_linearization,
        primal_inputs,
    );
    for (op.outputs, primal_outputs[0..callee.returns.len]) |source_output, output| {
        primal_ctx.set_primal(source_output, output);

        // A caller operation may use the call result as a JVP coefficient.
        //  Unused result residuals are removed with the other candidates.
        const residual_param = try linear_ctx.builder.param_like(source_output.aval);
        linear_ctx.set_primal(source_output, residual_param);
        try residuals.append(traversal.allocator, .{
            .primal = output,
            .linear_param = residual_param,
        });
    }

    var has_active_input = false;
    for (op.inputs) |operand| {
        if (ops.types.is_differentiable(operand.value.aval) and
            linear_ctx.tangent(operand.value) != null)
        {
            has_active_input = true;
            break;
        }
    }
    var has_tangent_output = false;
    for (callee_linearization.output_tangent_indices) |index| {
        if (index != null) {
            has_tangent_output = true;
            break;
        }
    }
    if (!has_active_input or !has_tangent_output) return;

    if (callee_linearization.input_tangent_indices.len != op.inputs.len)
        return error.InvalidLinearization;
    var tangent_input_count: usize = 0;
    for (callee_linearization.input_tangent_indices) |index| {
        if (index != null) tangent_input_count += 1;
    }

    // The augmented callee returns ordinary results followed by residuals.
    //  Its linear callee consumes differentiable operand tangents followed by
    //  those residuals.
    const linear_inputs = try traversal.allocator.alloc(
        *pr.Var,
        tangent_input_count + callee_linearization.residual_count,
    );
    defer traversal.allocator.free(linear_inputs);
    for (op.inputs, callee_linearization.input_tangent_indices) |operand, tangent_index| {
        const index = tangent_index orelse continue;
        if (index >= tangent_input_count) return error.InvalidLinearization;
        linear_inputs[index] = try linear_ctx.tangent_or_zero(operand.value);
    }
    for (
        primal_outputs[callee.returns.len..],
        linear_inputs[tangent_input_count..],
    ) |primal_residual, *linear_input| {
        const residual_param = try linear_ctx.builder.param_like(primal_residual.aval);
        linear_input.* = residual_param;
        try residuals.append(traversal.allocator, .{
            .primal = primal_residual,
            .linear_param = residual_param,
        });
    }

    const tangent_outputs = (try linear_ctx.builder.call(
        callee_linearization.linear,
        linear_inputs,
    )).outputs;
    for (op.outputs, callee_linearization.output_tangent_indices) |source_output, tangent_index| {
        const index = tangent_index orelse continue;
        if (index >= tangent_outputs.len) return error.InvalidLinearization;
        linear_ctx.set_tangent(source_output, tangent_outputs[index]);
    }
}

const GeneratedTranspose = struct {
    callee: pr.FunctionId,
    selected_outputs: []const usize,
    function_id: pr.FunctionId,
};

const TransposeTraversal = struct {
    allocator: std.mem.Allocator,
    program: *pr.Program,
    generated: std.ArrayList(GeneratedTranspose) = .empty,
    active: std.ArrayList(pr.FunctionId) = .empty,

    fn deinit(self: *TransposeTraversal) void {
        for (self.generated.items) |entry| {
            self.allocator.free(entry.selected_outputs);
        }
        self.generated.deinit(self.allocator);
        self.active.deinit(self.allocator);
        self.* = undefined;
    }

    fn get_or_create(
        self: *TransposeTraversal,
        callee_id: pr.FunctionId,
        selected_outputs: []const usize,
    ) AdError!pr.FunctionId {
        for (self.generated.items) |existing| {
            if (existing.callee != callee_id) continue;
            if (!std.mem.eql(usize, existing.selected_outputs, selected_outputs)) continue;
            return existing.function_id;
        }
        for (self.active.items) |active| {
            if (active == callee_id) return error.RecursiveDifferentiationUnsupported;
        }

        // The active stack rejects direct and mutually recursive call graphs.
        try self.active.append(self.allocator, callee_id);
        defer pop_active(&self.active, callee_id);

        const callee = self.program.get_function_by_id(callee_id) orelse
            return error.CallUnresolvedCallee;
        const base_name = try self.allocator.print("{s}_transpose", .{callee.name});
        defer self.allocator.free(base_name);
        const transpose_name = try self.program.reserve_unique_function_name(base_name);
        const transpose = try transpose_impl(
            self,
            callee,
            transpose_name,
            null,
            selected_outputs,
        );
        const transpose_id = try self.program.add_function(transpose);

        const owned_outputs = try self.allocator.dupe(usize, selected_outputs);
        errdefer self.allocator.free(owned_outputs);
        try self.generated.append(self.allocator, .{
            .callee = callee_id,
            .selected_outputs = owned_outputs,
            .function_id = transpose_id,
        });
        return transpose_id;
    }
};

fn transpose_linear(
    allocator: std.mem.Allocator,
    program: *pr.Program,
    function_id: pr.FunctionId,
    name: []const u8,
    wrt: ?[]const usize,
    of: []const usize,
) AdError!pr.Function {
    var traversal = TransposeTraversal{ .allocator = allocator, .program = program };
    defer traversal.deinit();
    const func = program.get_function_by_id(function_id) orelse
        return error.CallUnresolvedCallee;
    try traversal.active.append(allocator, function_id);
    defer pop_active(&traversal.active, function_id);
    return try transpose_impl(&traversal, func, name, wrt, of);
}

/// Allocate indices for values with standard dual values.
///
/// A provided selection preserves its order and duplicates. `null` selects
///  every eligible value in source order. `reject_empty` applies only to a
///  provided slice. The caller owns the returned slice.
fn alloc_differentiable_indices(
    allocator: std.mem.Allocator,
    values: []const *pr.Var,
    requested: ?[]const usize,
    comptime out_of_range: AdError,
    comptime reject_empty: bool,
) AdError![]usize {
    if (requested) |indices| {
        if (reject_empty and indices.len == 0) return error.EmptyOutputSelection;
        for (indices) |index| {
            if (index >= values.len) return out_of_range;
            if (!ops.types.is_differentiable(values[index].aval))
                return error.NonDifferentiableSelection;
        }
        return try allocator.dupe(usize, indices);
    }

    var selected: std.ArrayList(usize) = .empty;
    defer selected.deinit(allocator);
    for (values, 0..) |value, index| {
        if (ops.types.is_differentiable(value.aval))
            try selected.append(allocator, index);
    }
    return try selected.toOwnedSlice(allocator);
}

/// Transpose a linear function by propagating output cotangents in reverse.
///
/// `of` restricts cotangent seeds. `wrt` restricts harvested input
///  cotangents; `null` harvests every input with a standard cotangent. The
///  transform replays primal operations required by the registered transpose
///  rules.
///
/// Missing rules on active differentiable paths return
///  `AdError.MissingDerivativeRule`.
/// An input outside the selected output's dependency path has a zero
///  cotangent.
/// TODO(ad): Make the built-in differentiable dtype set an explicit AD policy.
fn transpose_impl(
    traversal: *TransposeTraversal,
    func: pr.Function,
    name: []const u8,
    wrt: ?[]const usize,
    of: []const usize,
) AdError!pr.Function {
    const allocator = traversal.allocator;
    const program = traversal.program;
    const saved = program.checkpoint_appends();
    errdefer program.restore_appends(saved);

    var primal_map = try allocator.alloc(?*pr.Var, func.var_count);
    defer allocator.free(primal_map);
    @memset(primal_map, null);

    var cotangent_map = try allocator.alloc(?*pr.Var, func.var_count);
    defer allocator.free(cotangent_map);
    @memset(cotangent_map, null);

    var b = try pr.FunctionBuilder.init(program, name);
    defer b.deinit();

    for (func.params) |param_var| {
        const new_param = try b.param_like(param_var.aval);
        primal_map[param_var.id] = new_param;
    }

    if (of.len == 0) return AdError.EmptyOutputSelection;
    for (of) |index| {
        if (index >= func.returns.len) return AdError.OfIndexOutOfRange;
    }

    for (of) |output_index| {
        const v = func.returns[output_index];
        if (!ops.types.is_differentiable(v.aval))
            return error.NonDifferentiableSelection;
        const new_seed = try b.param_like(v.aval);
        if (cotangent_map[v.id]) |cotangent| {
            cotangent_map[v.id] = try b.add(cotangent, new_seed);
        } else {
            cotangent_map[v.id] = new_seed;
        }
    }

    const ad_ctx = ops.types.VjpContext{
        .builder = &b,
        .primals = primal_map,
        .cotangents = cotangent_map,
        .allocator = allocator,
    };
    const primal_ctx = PrimalContext{
        .builder = &b,
        .primals = primal_map,
        .allocator = allocator,
    };

    // Transpose rules may need primal values, so replay the linear function
    //  before propagating its selected output cotangents in reverse.
    for (func.ops) |op| try replay_primal(primal_ctx, op);
    var op_index: usize = func.ops.len;
    while (op_index > 0) {
        op_index -= 1;
        switch (func.ops[op_index].params) {
            .call => try transpose_call(
                traversal,
                ad_ctx,
                func.ops[op_index],
            ),
            else => try ops.vjp(ad_ctx, func.ops[op_index]),
        }
    }

    const selected_inputs = try alloc_differentiable_indices(
        allocator,
        func.params,
        wrt,
        error.WrtIndexOutOfRange,
        false,
    );
    defer allocator.free(selected_inputs);
    const returns = try allocator.alloc(*pr.Var, selected_inputs.len);
    defer allocator.free(returns);

    for (selected_inputs, returns) |param_index, *result| {
        const v = func.params[param_index];
        if (cotangent_map[v.id]) |cotangent| {
            result.* = cotangent;
        } else {
            const t = v.aval.as_tensor();
            result.* = try b.scalar_broadcast(t.dtype, t.shape.dims, 0.0);
        }
    }

    return try b.finish(.{ .returns = returns });
}

fn transpose_call(
    traversal: *TransposeTraversal,
    ctx: ops.types.VjpContext,
    op: *const pr.Op,
) AdError!void {
    const selected_outputs = try ctx.allocator.alloc(usize, op.outputs.len);
    defer ctx.allocator.free(selected_outputs);
    var selected_count: usize = 0;
    for (op.outputs, 0..) |output, index| {
        if (ctx.cotangent(output) == null) continue;
        selected_outputs[selected_count] = index;
        selected_count += 1;
    }
    if (selected_count == 0) return;

    const transpose_id = try traversal.get_or_create(
        op.params.call.callee,
        selected_outputs[0..selected_count],
    );

    const call_inputs = try ctx.allocator.alloc(*pr.Var, op.inputs.len + selected_count);
    defer ctx.allocator.free(call_inputs);
    for (op.inputs, call_inputs[0..op.inputs.len]) |operand, *input| {
        input.* = try ctx.primal(operand.value);
    }
    for (selected_outputs[0..selected_count], call_inputs[op.inputs.len..]) |index, *input| {
        input.* = ctx.cotangent(op.outputs[index]) orelse
            return error.InvalidLinearization;
    }

    const input_cotangents = (try ctx.builder.call(transpose_id, call_inputs)).outputs;
    var cotangent_index: usize = 0;
    for (op.inputs) |operand| {
        if (!ops.types.is_differentiable(operand.value.aval)) continue;
        if (cotangent_index >= input_cotangents.len)
            return error.InvalidLinearization;
        try ctx.add_cotangent(operand.value, input_cotangents[cotangent_index]);
        cotangent_index += 1;
    }
    if (cotangent_index != input_cotangents.len)
        return error.InvalidLinearization;
}

/// Reverse-mode AD applies the pullback of \(f: M \to N\):
///
/// $$
/// \operatorname{vjp}_f:
/// (x, v) \in M \times T^*_{f(x)}N
/// \mapsto \mathrm{d}f_x^*(v) \in T_x^*M.
/// $$
///
/// For a cotangent seed \(v\), it computes the transpose-Jacobian product
/// \(J^\mathsf{T}(x) \cdot v\), the pullback
/// \(f^*: T^*_{f(x)} N \to T^*_x M\) evaluated at \(x\).
///
/// For
///
/// $$
/// f: X_1 \times \cdots \times X_n \to Y_1 \times \cdots \times Y_m,
/// $$
///
/// let \(O = (o_1, \ldots, o_s)\) be `opts.of` and
/// \(W = (w_1, \ldots, w_k)\) be `opts.wrt`. The transformed function has
/// signature
///
/// $$
/// \operatorname{vjp}^{O,W}_f:
/// (X_1 \times \cdots \times X_n)
/// \times (T^*_{f_{o_1}(x)}Y_{o_1} \times \cdots
/// \times T^*_{f_{o_s}(x)}Y_{o_s})
/// \to T^*_{x_{w_1}}X_{w_1} \times \cdots
/// \times T^*_{x_{w_k}}X_{w_k}.
/// $$
///
/// `null` expands \(O\) or \(W\) to every corresponding index whose dtype has
///  a standard cotangent. In the Euclidean or Cartesian case (\(G = I\)),
///  these equal gradients. In general, they are covectors and must be raised
///  with \(G^{-1}\) to obtain gradient tangent vectors.
pub fn vjp(
    allocator: std.mem.Allocator,
    program: *pr.Program,
    source: pr.FunctionId,
    name: []const u8,
    /// See `VjpOpts` for seed and result selection. Default `.{}` seeds every
    ///  differentiable source output and returns every differentiable
    ///  parameter cotangent.
    opts: VjpOpts,
) AdError!pr.FunctionId {
    const saved = program.checkpoint_appends();
    errdefer program.restore_appends(saved);
    const func = program.get_function_by_id(source) orelse
        return error.CallUnresolvedCallee;
    const selected_outputs = try alloc_differentiable_indices(
        allocator,
        func.returns,
        opts.of,
        error.OfIndexOutOfRange,
        true,
    );
    defer allocator.free(selected_outputs);
    const selected_inputs = try alloc_differentiable_indices(
        allocator,
        func.params,
        opts.wrt,
        error.WrtIndexOutOfRange,
        false,
    );
    defer allocator.free(selected_inputs);

    const result = try linearize(allocator, program, source, name);
    const linear = program.get_function_by_id(result.linear) orelse
        return error.CallUnresolvedCallee;

    const selected_linear_outputs = try allocator.alloc(usize, selected_outputs.len);
    defer allocator.free(selected_linear_outputs);
    var selected_linear_count: usize = 0;
    for (selected_outputs) |source_index| {
        const linear_index = result.output_tangent_indices[source_index] orelse continue;
        selected_linear_outputs[selected_linear_count] = linear_index;
        selected_linear_count += 1;
    }

    var transpose_id: ?pr.FunctionId = null;
    if (selected_linear_count != 0) {
        const tangent_wrt = try allocator.alloc(usize, selected_inputs.len);
        defer allocator.free(tangent_wrt);
        for (selected_inputs, tangent_wrt) |source_index, *tangent_index| {
            tangent_index.* = result.input_tangent_indices[source_index] orelse
                return error.NonDifferentiableSelection;
        }

        const transpose_name_base = try allocator.print("{s}_transpose", .{name});
        defer allocator.free(transpose_name_base);
        const transpose_name = try program.reserve_unique_function_name(transpose_name_base);
        const transpose_func = try transpose_linear(
            allocator,
            program,
            result.linear,
            transpose_name,
            tangent_wrt,
            selected_linear_outputs[0..selected_linear_count],
        );
        transpose_id = try program.add_function(transpose_func);
    }

    var builder = try pr.FunctionBuilder.init(program, name);
    defer builder.deinit();
    const params = try allocator.alloc(*pr.Var, func.params.len);
    defer allocator.free(params);
    for (func.params, params) |source_param, *param| {
        param.* = try builder.param_like(source_param.aval);
    }
    const seeds = try allocator.alloc(*pr.Var, selected_outputs.len);
    defer allocator.free(seeds);
    for (selected_outputs, seeds) |source_index, *seed| {
        seed.* = try builder.param_like(func.returns[source_index].aval);
    }

    const primal_outputs = try call_augmented_primal(
        program,
        &builder,
        func,
        result,
        params,
    );

    const gradients = try allocator.alloc(*pr.Var, selected_inputs.len);
    defer allocator.free(gradients);
    if (transpose_id) |callee| {
        const transpose = program.get_function_by_id(callee) orelse
            return error.CallUnresolvedCallee;
        const transpose_inputs = try allocator.alloc(*pr.Var, transpose.params.len);
        defer allocator.free(transpose_inputs);
        if (transpose_inputs.len != linear.params.len + selected_linear_count)
            return error.InvalidLinearization;
        if (linear.params.len < result.residual_count)
            return error.InvalidLinearization;
        const tangent_param_count = linear.params.len - result.residual_count;

        // The first linear inputs are tangent variables. Their primal values
        //  are zero because only residual coefficients affect the transpose.
        for (func.params, result.input_tangent_indices) |source_param, tangent_index| {
            const index = tangent_index orelse continue;
            if (index >= tangent_param_count) return error.InvalidLinearization;
            const tensor = source_param.as_tensor();
            transpose_inputs[index] = try builder.scalar_broadcast(
                tensor.dtype,
                tensor.shape.dims,
                0.0,
            );
        }
        @memcpy(
            transpose_inputs[tangent_param_count..linear.params.len],
            primal_outputs[func.returns.len..],
        );
        var seed_index: usize = 0;
        for (selected_outputs, seeds) |source_index, seed| {
            if (result.output_tangent_indices[source_index] == null) continue;
            transpose_inputs[linear.params.len + seed_index] = seed;
            seed_index += 1;
        }
        std.debug.assert(seed_index == selected_linear_count);
        const transpose_outputs = (try builder.call(callee, transpose_inputs)).outputs;
        if (transpose_outputs.len != gradients.len)
            return error.InvalidLinearization;
        @memcpy(gradients, transpose_outputs);
    } else {
        for (selected_inputs, gradients) |source_index, *gradient| {
            const tensor = func.params[source_index].as_tensor();
            gradient.* = try builder.scalar_broadcast(
                tensor.dtype,
                tensor.shape.dims,
                0.0,
            );
        }
    }

    const primal_count: usize = if (opts.include_primal_outputs) func.returns.len else 0;
    const outputs = try allocator.alloc(*pr.Var, primal_count + gradients.len);
    defer allocator.free(outputs);
    if (opts.include_primal_outputs) {
        @memcpy(outputs[0..primal_count], primal_outputs[0..primal_count]);
    }
    @memcpy(outputs[primal_count..], gradients);
    return try program.add_function(try builder.finish(.{ .returns = outputs }));
}

/// Forward-mode AD applies the differential of \(f: M \to N\):
///
/// $$
/// \operatorname{jvp}_f:
/// (x, v) \in M \times T_xM
/// \mapsto \mathrm{d}f_x(v) \in T_{f(x)}N.
/// $$
///
/// For a tangent seed \(v\), it computes the Jacobian-vector product
/// \(J(x) \cdot v\), the differential
/// \(\mathrm{d}f_x: T_x M \to T_{f(x)} N\) applied to \(v\).
///
/// For
///
/// $$
/// f: X_1 \times \cdots \times X_n \to Y_1 \times \cdots \times Y_m,
/// $$
///
/// let \(D = (d_1, \ldots, d_k)\) and \(E = (e_1, \ldots, e_l)\) contain
/// the input and output indices whose dtypes have standard dual values. The
/// transformed function has signature
///
/// $$
/// \operatorname{jvp}_f:
/// (X_1 \times \cdots \times X_n)
/// \times (T_{x_{d_1}}X_{d_1} \times \cdots \times T_{x_{d_k}}X_{d_k})
/// \to T_{f_{e_1}(x)}Y_{e_1} \times \cdots
/// \times T_{f_{e_l}(x)}Y_{e_l}.
/// $$
pub fn jvp(
    allocator: std.mem.Allocator,
    program: *pr.Program,
    source: pr.FunctionId,
    name: []const u8,
    opts: JvpOpts,
) AdError!pr.FunctionId {
    const saved = program.checkpoint_appends();
    errdefer program.restore_appends(saved);
    const func = program.get_function_by_id(source) orelse
        return error.CallUnresolvedCallee;
    const result = try linearize(allocator, program, source, name);
    const linear = program.get_function_by_id(result.linear) orelse
        return error.CallUnresolvedCallee;

    var builder = try pr.FunctionBuilder.init(program, name);
    defer builder.deinit();

    const params = try allocator.alloc(*pr.Var, func.params.len);
    defer allocator.free(params);
    if (result.input_tangent_indices.len != func.params.len)
        return error.InvalidLinearization;
    var tangent_count: usize = 0;
    for (result.input_tangent_indices) |index| {
        if (index != null) tangent_count += 1;
    }
    const tangents = try allocator.alloc(*pr.Var, tangent_count);
    defer allocator.free(tangents);
    for (func.params, params) |source_param, *param| {
        param.* = try builder.param_like(source_param.aval);
    }
    for (func.params, result.input_tangent_indices) |source_param, tangent_index| {
        const index = tangent_index orelse continue;
        if (index >= tangents.len) return error.InvalidLinearization;
        tangents[index] = try builder.param_like(source_param.aval);
    }

    const primal_outputs = try call_augmented_primal(
        program,
        &builder,
        func,
        result,
        params,
    );

    const linear_inputs = try allocator.alloc(*pr.Var, linear.params.len);
    defer allocator.free(linear_inputs);
    if (linear_inputs.len != tangents.len + result.residual_count)
        return error.InvalidLinearization;
    @memcpy(linear_inputs[0..tangents.len], tangents);
    @memcpy(linear_inputs[tangents.len..], primal_outputs[func.returns.len..]);
    const linear_outputs = (try builder.call(result.linear, linear_inputs)).outputs;
    if (linear_outputs.len != linear.returns.len)
        return error.InvalidLinearization;

    const differentiable_outputs = try alloc_differentiable_indices(
        allocator,
        func.returns,
        null,
        error.OfIndexOutOfRange,
        false,
    );
    defer allocator.free(differentiable_outputs);
    const primal_count: usize = if (opts.include_primal_outputs) func.returns.len else 0;
    const output_count = primal_count + differentiable_outputs.len;
    const outputs = try allocator.alloc(*pr.Var, output_count);
    defer allocator.free(outputs);
    var output_index: usize = 0;
    if (opts.include_primal_outputs) {
        @memcpy(outputs[0..func.returns.len], primal_outputs[0..func.returns.len]);
        output_index = func.returns.len;
    }

    for (differentiable_outputs) |source_index| {
        const source_output = func.returns[source_index];
        const tangent_index = result.output_tangent_indices[source_index];
        outputs[output_index] = if (tangent_index) |index| tangent: {
            if (index >= linear_outputs.len) return error.InvalidLinearization;
            break :tangent linear_outputs[index];
        } else zero: {
            const tensor = source_output.as_tensor();
            break :zero try builder.scalar_broadcast(
                tensor.dtype,
                tensor.shape.dims,
                0.0,
            );
        };
        output_index += 1;
    }
    if (output_index != outputs.len) return error.InvalidLinearization;
    return try program.add_function(try builder.finish(.{ .returns = outputs }));
}

/// Emit a ones-like cotangent for VJP seeding.
pub fn emit_cotangent(builder: *pr.FunctionBuilder, tensor: pr.Tensor) pr.BuildError!*pr.Var {
    const s = try builder.scalar(tensor.dtype, 1.0);
    if (tensor.shape.rank() == 0) return s;
    return try builder.broadcast_in_dim(s, tensor.shape.dims, &.{});
}

test "linearize separates residuals from tangent inputs" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var builder = try pr.FunctionBuilder.init(&program, "square");
    defer builder.deinit();
    const x = try builder.param_tensor(.f32, &.{});
    const y = try builder.multiply(x, x);
    const source = try builder.finish(.{ .returns = &.{y} });
    const source_id = try program.add_function(source);

    const result = try linearize(std.testing.allocator, &program, source_id, "square_linearized");
    try std.testing.expectEqual(@as(usize, 1), result.residual_count);
    try std.testing.expectEqualSlices(?usize, &.{0}, result.output_tangent_indices);

    const augmented_primal = program.get_function_by_id(result.augmented_primal).?;
    const linear = program.get_function_by_id(result.linear).?;
    try std.testing.expectEqual(@as(usize, 1), augmented_primal.params.len);
    try std.testing.expectEqual(@as(usize, 2), augmented_primal.returns.len);
    try std.testing.expectEqual(@as(usize, 2), linear.params.len);
    try std.testing.expectEqual(@as(usize, 1), linear.returns.len);

    const pr_eval = @import("pr_eval");
    var primal = try pr_eval.HostTensor.init_with_data(std.testing.allocator, &.{}, &.{3.0});
    defer primal.deinit();
    const primal_results = try pr_eval.eval(
        std.testing.allocator,
        &program,
        augmented_primal,
        &.{primal},
    );
    defer {
        for (primal_results) |*value| value.deinit();
        std.testing.allocator.free(primal_results);
    }
    try std.testing.expectApproxEqAbs(@as(f32, 9.0), primal_results[0].data[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), primal_results[1].data[0], 1e-6);

    var tangent = try pr_eval.HostTensor.init_with_data(std.testing.allocator, &.{}, &.{2.0});
    defer tangent.deinit();
    const tangent_results = try pr_eval.eval(
        std.testing.allocator,
        &program,
        linear,
        &.{ tangent, primal_results[1] },
    );
    defer {
        for (tangent_results) |*value| value.deinit();
        std.testing.allocator.free(tangent_results);
    }
    try std.testing.expectApproxEqAbs(@as(f32, 12.0), tangent_results[0].data[0], 1e-6);
}

test "linearize carries structural zero through a predicate broadcast" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var builder = try pr.FunctionBuilder.init(&program, "masked_value");
    defer builder.deinit();
    const x = try builder.param_tensor(.f32, &.{2});
    const zero = try builder.scalar_broadcast(.f32, &.{2}, 0.0);
    const predicate = try builder.compare(x, zero, .{
        .direction = .GT,
        .compare_type = .FLOAT,
    });
    const predicate_row = try builder.broadcast_in_dim(predicate, &.{ 1, 2 }, &.{1});
    const x_row = try builder.broadcast_in_dim(x, &.{ 1, 2 }, &.{1});
    const zero_row = try builder.broadcast_in_dim(zero, &.{ 1, 2 }, &.{1});
    const selected = try builder.select(predicate_row, x_row, zero_row);
    const source = try program.add_function(try builder.finish(.{ .returns = &.{selected} }));

    const differentiated_id = try jvp(
        std.testing.allocator,
        &program,
        source,
        "masked_value_jvp",
        .{},
    );
    const differentiated = program.get_function_by_id(differentiated_id).?;

    const pr_eval = @import("pr_eval");
    var primal = try pr_eval.HostTensor.init_with_data(
        std.testing.allocator,
        &.{2},
        &.{ 2.0, -3.0 },
    );
    defer primal.deinit();
    var tangent = try pr_eval.HostTensor.init_with_data(
        std.testing.allocator,
        &.{2},
        &.{ 7.0, 11.0 },
    );
    defer tangent.deinit();
    const results = try pr_eval.eval(
        std.testing.allocator,
        &program,
        differentiated,
        &.{ primal, tangent },
    );
    defer {
        for (results) |*result| result.deinit();
        std.testing.allocator.free(results);
    }
    try std.testing.expectEqualSlices(f32, &.{ 7.0, 0.0 }, results[0].data);
}

test "jvp preserves structural zero across an inactive call" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var callee_builder = try pr.FunctionBuilder.init(&program, "square");
    defer callee_builder.deinit();
    const callee_input = try callee_builder.param_tensor(.f32, &.{});
    const callee_result = try callee_builder.multiply(callee_input, callee_input);
    const callee = try program.add_function(try callee_builder.finish(.{ .returns = &.{callee_result} }));

    var caller_builder = try pr.FunctionBuilder.init(&program, "constant_result");
    defer caller_builder.deinit();
    _ = try caller_builder.param_tensor(.f32, &.{});
    const constant = try caller_builder.scalar(.f32, 3.0);
    const call_outputs = (try caller_builder.call(callee, &.{constant})).outputs;
    const source = try program.add_function(try caller_builder.finish(.{ .returns = call_outputs }));

    const linearized = try linearize(
        std.testing.allocator,
        &program,
        source,
        "constant_result_linearized",
    );
    try std.testing.expectEqualSlices(?usize, &.{0}, linearized.input_tangent_indices);
    try std.testing.expectEqualSlices(?usize, &.{null}, linearized.output_tangent_indices);
    const linear = program.get_function_by_id(linearized.linear).?;
    try std.testing.expectEqual(@as(usize, 1), linear.params.len);
    try std.testing.expectEqual(@as(usize, 0), linear.ops.len);
    try std.testing.expectEqual(@as(usize, 0), linear.returns.len);

    const differentiated_id = try jvp(
        std.testing.allocator,
        &program,
        source,
        "constant_result_jvp",
        .{},
    );
    const differentiated = program.get_function_by_id(differentiated_id).?;
    try std.testing.expectEqual(@as(usize, 2), differentiated.params.len);
    try std.testing.expectEqual(@as(usize, 1), differentiated.returns.len);

    const pr_eval = @import("pr_eval");
    var primal = try pr_eval.HostTensor.init_with_data(
        std.testing.allocator,
        &.{},
        &.{9.0},
    );
    defer primal.deinit();
    var tangent = try pr_eval.HostTensor.init_with_data(
        std.testing.allocator,
        &.{},
        &.{4.0},
    );
    defer tangent.deinit();
    const results = try pr_eval.eval(
        std.testing.allocator,
        &program,
        differentiated,
        &.{ primal, tangent },
    );
    defer {
        for (results) |*result| result.deinit();
        std.testing.allocator.free(results);
    }
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), results[0].data[0], 1e-6);
}

fn build_mixed_select_call(program: *pr.Program) !pr.FunctionId {
    var callee_builder = try pr.FunctionBuilder.init(program, "select_values");
    defer callee_builder.deinit();
    const callee_condition = try callee_builder.param_tensor(.bool, &.{2});
    const callee_lhs = try callee_builder.param_tensor(.f32, &.{2});
    const callee_rhs = try callee_builder.param_tensor(.f32, &.{2});
    const callee_result = try callee_builder.select(
        callee_condition,
        callee_lhs,
        callee_rhs,
    );
    const callee = try program.add_function(
        try callee_builder.finish(.{ .returns = &.{callee_result} }),
    );

    var caller_builder = try pr.FunctionBuilder.init(program, "call_select_values");
    defer caller_builder.deinit();
    const condition = try caller_builder.param_tensor(.bool, &.{2});
    const lhs = try caller_builder.param_tensor(.f32, &.{2});
    const rhs = try caller_builder.param_tensor(.f32, &.{2});
    const selected = (try caller_builder.call(callee, &.{ condition, lhs, rhs })).outputs;
    return try program.add_function(
        try caller_builder.finish(.{ .returns = &.{ condition, selected[0] } }),
    );
}

test "AD omits dual values for discrete parameters and results across calls" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    const source = try build_mixed_select_call(&program);
    const linearized = try linearize(
        std.testing.allocator,
        &program,
        source,
        "call_select_values_linearized",
    );
    try std.testing.expectEqualSlices(
        ?usize,
        &.{ null, 0, 1 },
        linearized.input_tangent_indices,
    );
    try std.testing.expectEqualSlices(
        ?usize,
        &.{ null, 0 },
        linearized.output_tangent_indices,
    );

    const jvp_id = try jvp(
        std.testing.allocator,
        &program,
        source,
        "call_select_values_jvp",
        .{},
    );
    const jvp_func = program.get_function_by_id(jvp_id).?;
    try std.testing.expectEqual(@as(usize, 5), jvp_func.params.len);
    try std.testing.expectEqual(@as(usize, 1), jvp_func.returns.len);

    try std.testing.expectError(
        error.NonDifferentiableSelection,
        vjp(
            std.testing.allocator,
            &program,
            source,
            "call_select_values_bool_output_vjp",
            .{ .of = &.{0} },
        ),
    );
    try std.testing.expectError(
        error.NonDifferentiableSelection,
        vjp(
            std.testing.allocator,
            &program,
            source,
            "call_select_values_bool_input_vjp",
            .{ .wrt = &.{0} },
        ),
    );

    const vjp_id = try vjp(
        std.testing.allocator,
        &program,
        source,
        "call_select_values_vjp",
        .{},
    );
    const vjp_func = program.get_function_by_id(vjp_id).?;
    try std.testing.expectEqual(@as(usize, 4), vjp_func.params.len);
    try std.testing.expectEqual(@as(usize, 2), vjp_func.returns.len);

    const pr_eval = @import("pr_eval");
    var condition = try pr_eval.HostTensor.init_with_data(
        std.testing.allocator,
        &.{2},
        &.{ 1.0, 0.0 },
    );
    defer condition.deinit();
    var lhs = try pr_eval.HostTensor.init_with_data(
        std.testing.allocator,
        &.{2},
        &.{ 10.0, 20.0 },
    );
    defer lhs.deinit();
    var rhs = try pr_eval.HostTensor.init_with_data(
        std.testing.allocator,
        &.{2},
        &.{ 30.0, 40.0 },
    );
    defer rhs.deinit();
    var lhs_tangent = try pr_eval.HostTensor.init_with_data(
        std.testing.allocator,
        &.{2},
        &.{ 2.0, 3.0 },
    );
    defer lhs_tangent.deinit();
    var rhs_tangent = try pr_eval.HostTensor.init_with_data(
        std.testing.allocator,
        &.{2},
        &.{ 5.0, 7.0 },
    );
    defer rhs_tangent.deinit();

    const tangent_results = try pr_eval.eval(
        std.testing.allocator,
        &program,
        jvp_func,
        &.{ condition, lhs, rhs, lhs_tangent, rhs_tangent },
    );
    defer {
        for (tangent_results) |*result| result.deinit();
        std.testing.allocator.free(tangent_results);
    }
    try std.testing.expectEqualSlices(f32, &.{ 2.0, 7.0 }, tangent_results[0].data);

    var seed = try pr_eval.HostTensor.init_with_data(
        std.testing.allocator,
        &.{2},
        &.{ 11.0, 13.0 },
    );
    defer seed.deinit();
    const cotangent_results = try pr_eval.eval(
        std.testing.allocator,
        &program,
        vjp_func,
        &.{ condition, lhs, rhs, seed },
    );
    defer {
        for (cotangent_results) |*result| result.deinit();
        std.testing.allocator.free(cotangent_results);
    }
    try std.testing.expectEqualSlices(f32, &.{ 11.0, 0.0 }, cotangent_results[0].data);
    try std.testing.expectEqualSlices(f32, &.{ 0.0, 13.0 }, cotangent_results[1].data);
}

test "linearized programs serialize" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var builder = try pr.FunctionBuilder.init(&program, "square");
    defer builder.deinit();
    const x = try builder.param_tensor(.f32, &.{});
    const y = try builder.multiply(x, x);
    const source = try builder.finish(.{ .returns = &.{y} });
    const source_id = try program.add_function(source);
    _ = try linearize(std.testing.allocator, &program, source_id, "square_linearized");

    const serialize = @import("serialize.zig");
    var writer: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer writer.deinit();
    try serialize.emit(&program, &writer.writer);
    const bytes = try writer.toOwnedSlice();
    defer std.testing.allocator.free(bytes);

    var parsed = try serialize.parse(std.testing.allocator, bytes);
    defer parsed.deinit();
    try pr.validate_program(&parsed);
    try std.testing.expectEqual(program.functions().len, parsed.functions().len);
}

test "linearize rejects recursive call graphs" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var placeholder_builder = try pr.FunctionBuilder.init(&program, "recursive");
    defer placeholder_builder.deinit();
    const placeholder_param = try placeholder_builder.param_tensor(.f32, &.{});
    const function_id = try program.add_function(
        try placeholder_builder.finish(.{ .returns = &.{placeholder_param} }),
    );

    var recursive_builder = try pr.FunctionBuilder.init(&program, "recursive");
    defer recursive_builder.deinit();
    const recursive_param = try recursive_builder.param_tensor(.f32, &.{});
    const recursive_call = try recursive_builder.call(function_id, &.{recursive_param});
    const recursive = try recursive_builder.finish(.{ .returns = recursive_call.outputs });
    try program.replace_function(function_id, recursive);

    try std.testing.expectError(
        error.RecursiveDifferentiationUnsupported,
        linearize(std.testing.allocator, &program, function_id, "linearized"),
    );
    try std.testing.expectEqual(@as(usize, 1), program.functions().len);
}

test "vjp produces gradients matching input shapes" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const a_id = try b.param_tensor(.f32, &.{ 2, 3 });
    const b_id = try b.param_tensor(.f32, &.{ 3, 2 });
    const c_id = try b.param_tensor(.f32, &.{ 2, 2 });

    const mm_id = try b.mm(a_id, b_id);
    const add_id = try b.add(mm_id, c_id);
    const out_id = try b.multiply(add_id, c_id);

    const func = try b.finish(.{ .returns = &.{out_id} });
    const source_id = try program.add_function(func);

    const vjp_id = try vjp(std.testing.allocator, &program, source_id, "vjp", .{});
    const vjp_func = program.get_function_by_id(vjp_id).?;
    try pr.validate_function(vjp_func);

    try std.testing.expectEqual(@as(usize, func.params.len + func.returns.len), vjp_func.params.len);
    try std.testing.expectEqual(@as(usize, func.params.len), vjp_func.returns.len);

    for (func.params, 0..) |param_var, i| {
        const p_t = param_var.as_tensor();
        const g_var = vjp_func.returns[i];
        const g_t = g_var.as_tensor();
        try std.testing.expectEqual(p_t.dtype, g_t.dtype);
        try std.testing.expect(std.mem.eql(i64, p_t.shape.dims, g_t.shape.dims));
    }
}

test "vjp selects output cotangent seeds" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{});
    const square = try b.multiply(x, x);
    const double = try b.add(x, x);
    const func = try b.finish(.{ .returns = &.{ square, double } });
    const source_id = try program.add_function(func);

    const vjp_id = try vjp(std.testing.allocator, &program, source_id, "vjp", .{
        .of = &.{1},
    });
    const vjp_func = program.get_function_by_id(vjp_id).?;
    try pr.validate_function(vjp_func);

    try std.testing.expectEqual(@as(usize, 2), vjp_func.params.len);
    try std.testing.expectEqual(@as(usize, 1), vjp_func.returns.len);

    const pr_eval = @import("pr_eval");
    var primal = try pr_eval.HostTensor.init_with_data(std.testing.allocator, &.{}, &.{3.0});
    defer primal.deinit();
    var seed = try pr_eval.HostTensor.init_with_data(std.testing.allocator, &.{}, &.{1.0});
    defer seed.deinit();
    const results = try pr_eval.eval(
        std.testing.allocator,
        &program,
        vjp_func,
        &.{ primal, seed },
    );
    defer {
        for (results) |*result| result.deinit();
        std.testing.allocator.free(results);
    }
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), results[0].data[0], 1e-6);
}

test "vjp accumulates duplicate output seeds" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{});
    const y = try b.multiply(x, x);
    const func = try b.finish(.{ .returns = &.{y} });
    const source_id = try program.add_function(func);

    const vjp_id = try vjp(std.testing.allocator, &program, source_id, "vjp", .{
        .of = &.{ 0, 0 },
    });
    const vjp_func = program.get_function_by_id(vjp_id).?;
    try pr.validate_function(vjp_func);

    try std.testing.expectEqual(@as(usize, 3), vjp_func.params.len);
    try std.testing.expectEqual(@as(usize, 1), vjp_func.returns.len);
}

test "vjp rejects invalid output selection" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{});
    const y = try b.multiply(x, x);
    const func = try b.finish(.{ .returns = &.{y} });
    const source_id = try program.add_function(func);

    try std.testing.expectError(
        error.EmptyOutputSelection,
        vjp(std.testing.allocator, &program, source_id, "empty", .{ .of = &.{} }),
    );
    try std.testing.expectError(
        error.OfIndexOutOfRange,
        vjp(std.testing.allocator, &program, source_id, "out_of_range", .{ .of = &.{1} }),
    );
}

test "vjp can include primals before gradients" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{ 2, 2 });
    const y = try b.multiply(x, x);
    const func = try b.finish(.{ .returns = &.{y} });
    const source_id = try program.add_function(func);

    const vjp_id = try vjp(std.testing.allocator, &program, source_id, "vjp_with_primals", .{
        .include_primal_outputs = true,
    });
    const vjp_func = program.get_function_by_id(vjp_id).?;
    try pr.validate_function(vjp_func);

    try std.testing.expectEqual(@as(usize, func.returns.len + func.params.len), vjp_func.returns.len);
}

test "vjp distinguishes an unsupported derivative configuration" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "strided_slice");
    defer b.deinit();
    const input = try b.param_tensor(.f32, &.{6});
    const output = try b.slice(input, .{
        .start_indices = &.{0},
        .limit_indices = &.{6},
        .strides = &.{2},
    });
    const source = try program.add_function(try b.finish(.{ .returns = &.{output} }));

    try std.testing.expectError(
        error.UnsupportedDerivative,
        vjp(std.testing.allocator, &program, source, "strided_slice_vjp", .{}),
    );
}

test "dot_general vjp supports 2 batch dims" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const lhs = try b.param_tensor(.f32, &.{ 2, 3, 4, 5 }); // [B,H,M,K]
    const rhs = try b.param_tensor(.f32, &.{ 2, 3, 5, 6 }); // [B,H,K,N]
    const out = try b.dot_general(lhs, rhs, .{
        .lhs_batch_dims = &.{ 0, 1 },
        .rhs_batch_dims = &.{ 0, 1 },
        .lhs_contracting_dims = &.{3},
        .rhs_contracting_dims = &.{2},
    });

    const func = try b.finish(.{ .returns = &.{out} });
    const source_id = try program.add_function(func);

    const vjp_id = try vjp(std.testing.allocator, &program, source_id, "vjp", .{});
    const vjp_func = program.get_function_by_id(vjp_id).?;
    try pr.validate_function(vjp_func);

    try std.testing.expectEqual(@as(usize, func.params.len + func.returns.len), vjp_func.params.len);
    try std.testing.expectEqual(@as(usize, func.params.len), vjp_func.returns.len);

    for (func.params, 0..) |param_var, i| {
        const p_t = param_var.as_tensor();
        const g_var = vjp_func.returns[i];
        const g_t = g_var.as_tensor();
        try std.testing.expectEqual(p_t.dtype, g_t.dtype);
        try std.testing.expect(std.mem.eql(i64, p_t.shape.dims, g_t.shape.dims));
    }
}

test "dot_general vjp supports non-prefix batch dims" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    // lhs/rhs: [B,S,H,D]
    const lhs = try b.param_tensor(.f32, &.{ 2, 4, 3, 5 });
    const rhs = try b.param_tensor(.f32, &.{ 2, 4, 3, 5 });
    const out = try b.dot_general(lhs, rhs, .{
        .lhs_batch_dims = &.{ 0, 2 },
        .rhs_batch_dims = &.{ 0, 2 },
        .lhs_contracting_dims = &.{3},
        .rhs_contracting_dims = &.{3},
    });

    const func = try b.finish(.{ .returns = &.{out} });
    const source_id = try program.add_function(func);

    const vjp_id = try vjp(std.testing.allocator, &program, source_id, "vjp", .{});
    const vjp_func = program.get_function_by_id(vjp_id).?;
    try pr.validate_function(vjp_func);
}

test "dot_general vjp supports differing batch dim positions" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    // lhs: [B,H,S,S], rhs: [B,S,H,D] -> out: [B,H,S,D]
    const lhs = try b.param_tensor(.f32, &.{ 2, 3, 4, 4 });
    const rhs = try b.param_tensor(.f32, &.{ 2, 4, 3, 5 });
    const out = try b.dot_general(lhs, rhs, .{
        .lhs_batch_dims = &.{ 0, 1 },
        .rhs_batch_dims = &.{ 0, 2 },
        .lhs_contracting_dims = &.{3},
        .rhs_contracting_dims = &.{1},
    });

    const func = try b.finish(.{ .returns = &.{out} });
    const source_id = try program.add_function(func);

    const vjp_id = try vjp(std.testing.allocator, &program, source_id, "vjp", .{});
    const vjp_func = program.get_function_by_id(vjp_id).?;
    try pr.validate_function(vjp_func);
}

test "dot_general vjp supports multi-contract dims" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    // lhs/rhs: [B,S,H,D], contract over H and D -> out [B,S]
    const lhs = try b.param_tensor(.f32, &.{ 2, 3, 4, 5 });
    const rhs = try b.param_tensor(.f32, &.{ 2, 3, 4, 5 });
    const out = try b.dot_general(lhs, rhs, .{
        .lhs_batch_dims = &.{ 0, 1 },
        .rhs_batch_dims = &.{ 0, 1 },
        .lhs_contracting_dims = &.{ 2, 3 },
        .rhs_contracting_dims = &.{ 2, 3 },
    });

    const func = try b.finish(.{ .returns = &.{out} });
    const source_id = try program.add_function(func);

    const vjp_id = try vjp(std.testing.allocator, &program, source_id, "vjp", .{});
    const vjp_func = program.get_function_by_id(vjp_id).?;
    try pr.validate_function(vjp_func);
}

test "jvp produces tangent outputs matching function output shapes" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const a_id = try b.param_tensor(.f32, &.{ 2, 3 });
    const b_id = try b.param_tensor(.f32, &.{ 3, 2 });
    const c_id = try b.param_tensor(.f32, &.{ 2, 2 });

    const mm_id = try b.mm(a_id, b_id);
    const add_id = try b.add(mm_id, c_id);
    const out_id = try b.multiply(add_id, c_id);

    const func = try b.finish(.{ .returns = &.{out_id} });
    const source_id = try program.add_function(func);

    const jvp_id = try jvp(std.testing.allocator, &program, source_id, "jvp", .{});
    const jvp_func = program.get_function_by_id(jvp_id).?;
    try pr.validate_function(jvp_func);

    try std.testing.expectEqual(func.params.len * 2, jvp_func.params.len);
    try std.testing.expectEqual(func.returns.len, jvp_func.returns.len);

    for (func.returns, 0..) |ret_var, i| {
        const orig_t = ret_var.as_tensor();
        const jvp_var = jvp_func.returns[i];
        const jvp_t = jvp_var.as_tensor();
        try std.testing.expectEqual(orig_t.dtype, jvp_t.dtype);
        try std.testing.expect(std.mem.eql(i64, orig_t.shape.dims, jvp_t.shape.dims));
    }
}

fn build_repeated_square_calls(program: *pr.Program) !pr.FunctionId {
    var callee_builder = try pr.FunctionBuilder.init(program, "square");
    defer callee_builder.deinit();
    const callee_input = try callee_builder.param_tensor(.f32, &.{});
    const callee_output = try callee_builder.multiply(callee_input, callee_input);
    const callee_id = try program.add_function(try callee_builder.finish(.{ .returns = &.{callee_output} }));

    var caller_builder = try pr.FunctionBuilder.init(program, "caller");
    defer caller_builder.deinit();
    const caller_input = try caller_builder.param_tensor(.f32, &.{});
    const first_outputs = (try caller_builder.call(callee_id, &.{caller_input})).outputs;
    const second_outputs = (try caller_builder.call(callee_id, &.{caller_input})).outputs;
    const sum = try caller_builder.add(first_outputs[0], second_outputs[0]);
    return try program.add_function(try caller_builder.finish(.{ .returns = &.{sum} }));
}

test "linearize reuses a callee linearization across call sites" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    const caller_id = try build_repeated_square_calls(&program);
    const result = try linearize(std.testing.allocator, &program, caller_id, "caller");
    try pr.validate_program(&program);

    const augmented_primal = program.get_function_by_id(result.augmented_primal).?;
    const linear = program.get_function_by_id(result.linear).?;
    try std.testing.expectEqual(@as(usize, 3), augmented_primal.ops.len);
    try std.testing.expectEqual(@as(usize, 3), linear.ops.len);

    const first_primal_call = augmented_primal.ops[0].params.call.callee;
    const second_primal_call = augmented_primal.ops[1].params.call.callee;
    try std.testing.expectEqual(first_primal_call, second_primal_call);
    const first_linear_call = linear.ops[0].params.call.callee;
    const second_linear_call = linear.ops[1].params.call.callee;
    try std.testing.expectEqual(first_linear_call, second_linear_call);
    try std.testing.expect(first_primal_call != first_linear_call);
    try std.testing.expectEqual(@as(usize, 6), program.functions().len);
}

test "vjp composes through repeated calls" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    const caller_id = try build_repeated_square_calls(&program);
    const differentiated_id = try vjp(std.testing.allocator, &program, caller_id, "caller_vjp", .{});
    const differentiated = program.get_function_by_id(differentiated_id).?;
    try pr.validate_program(&program);

    try std.testing.expectEqual(@as(usize, 2), differentiated.params.len);
    try std.testing.expectEqual(@as(usize, 1), differentiated.returns.len);

    const pr_eval = @import("pr_eval");
    var primal = try pr_eval.HostTensor.init_with_data(std.testing.allocator, &.{}, &.{3.0});
    defer primal.deinit();
    var seed = try pr_eval.HostTensor.init_with_data(std.testing.allocator, &.{}, &.{1.0});
    defer seed.deinit();
    const results = try pr_eval.eval(
        std.testing.allocator,
        &program,
        differentiated,
        &.{ primal, seed },
    );
    defer {
        for (results) |*result| result.deinit();
        std.testing.allocator.free(results);
    }
    try std.testing.expectApproxEqAbs(@as(f32, 12.0), results[0].data[0], 1e-6);
}

test "AD carries call results into nonlinear consumers" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var callee_builder = try pr.FunctionBuilder.init(&program, "square");
    defer callee_builder.deinit();
    const callee_input = try callee_builder.param_tensor(.f32, &.{});
    const callee_output = try callee_builder.multiply(callee_input, callee_input);
    const callee_id = try program.add_function(
        try callee_builder.finish(.{ .returns = &.{callee_output} }),
    );

    var caller_builder = try pr.FunctionBuilder.init(&program, "fourth_power");
    defer caller_builder.deinit();
    const caller_input = try caller_builder.param_tensor(.f32, &.{});
    const call_outputs = (try caller_builder.call(callee_id, &.{caller_input})).outputs;
    const caller_output = try caller_builder.multiply(call_outputs[0], call_outputs[0]);
    const caller = try caller_builder.finish(.{ .returns = &.{caller_output} });
    const caller_id = try program.add_function(caller);

    const differentiated_id = try jvp(
        std.testing.allocator,
        &program,
        caller_id,
        "fourth_power_jvp",
        .{},
    );
    const differentiated = program.get_function_by_id(differentiated_id).?;
    try pr.validate_program(&program);

    const pr_eval = @import("pr_eval");
    var primal = try pr_eval.HostTensor.init_with_data(
        std.testing.allocator,
        &.{},
        &.{2.0},
    );
    defer primal.deinit();
    var tangent = try pr_eval.HostTensor.init_with_data(
        std.testing.allocator,
        &.{},
        &.{3.0},
    );
    defer tangent.deinit();
    const results = try pr_eval.eval(
        std.testing.allocator,
        &program,
        differentiated,
        &.{ primal, tangent },
    );
    defer {
        for (results) |*value| value.deinit();
        std.testing.allocator.free(results);
    }
    try std.testing.expectApproxEqAbs(@as(f32, 96.0), results[0].data[0], 1e-5);

    const reverse_id = try vjp(
        std.testing.allocator,
        &program,
        caller_id,
        "fourth_power_vjp",
        .{},
    );
    const reverse = program.get_function_by_id(reverse_id).?;
    var seed = try pr_eval.HostTensor.init_with_data(
        std.testing.allocator,
        &.{},
        &.{3.0},
    );
    defer seed.deinit();
    const cotangents = try pr_eval.eval(
        std.testing.allocator,
        &program,
        reverse,
        &.{ primal, seed },
    );
    defer {
        for (cotangents) |*value| value.deinit();
        std.testing.allocator.free(cotangents);
    }
    try std.testing.expectApproxEqAbs(@as(f32, 96.0), cotangents[0].data[0], 1e-5);
}

test "jvp can include primals before tangents" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{ 2, 2 });
    const y = try b.multiply(x, x);
    const func = try b.finish(.{ .returns = &.{y} });
    const source_id = try program.add_function(func);

    const jvp_id = try jvp(std.testing.allocator, &program, source_id, "jvp_with_primals", .{
        .include_primal_outputs = true,
    });
    const jvp_func = program.get_function_by_id(jvp_id).?;
    try pr.validate_function(jvp_func);

    try std.testing.expectEqual(func.returns.len * 2, jvp_func.returns.len);
    try std.testing.expectEqual(func.params.len * 2, jvp_func.params.len);
}

test "jvp routes maximum tangents through the selected operand" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "maximum");
    defer b.deinit();
    const lhs = try b.param_tensor(.f32, &.{2});
    const rhs = try b.param_tensor(.f32, &.{2});
    const source = try b.finish(.{ .returns = &.{try b.maximum(lhs, rhs)} });
    const source_id = try program.add_function(source);
    const differentiated_id = try jvp(
        std.testing.allocator,
        &program,
        source_id,
        "maximum_jvp",
        .{},
    );
    const differentiated = program.get_function_by_id(differentiated_id).?;

    const pr_eval = @import("pr_eval");
    var lhs_value = try pr_eval.HostTensor.init_with_data(
        std.testing.allocator,
        &.{2},
        &.{ 3.0, 1.0 },
    );
    defer lhs_value.deinit();
    var rhs_value = try pr_eval.HostTensor.init_with_data(
        std.testing.allocator,
        &.{2},
        &.{ 2.0, 4.0 },
    );
    defer rhs_value.deinit();
    var lhs_tangent = try pr_eval.HostTensor.init_with_data(
        std.testing.allocator,
        &.{2},
        &.{ 5.0, 6.0 },
    );
    defer lhs_tangent.deinit();
    var rhs_tangent = try pr_eval.HostTensor.init_with_data(
        std.testing.allocator,
        &.{2},
        &.{ 7.0, 8.0 },
    );
    defer rhs_tangent.deinit();

    const results = try pr_eval.eval(
        std.testing.allocator,
        &program,
        differentiated,
        &.{ lhs_value, rhs_value, lhs_tangent, rhs_tangent },
    );
    defer {
        for (results) |*result| result.deinit();
        std.testing.allocator.free(results);
    }
    try std.testing.expectEqualSlices(f32, &.{ 5.0, 8.0 }, results[0].data);
}

test "dot_general jvp with batch dims" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const lhs = try b.param_tensor(.f32, &.{ 2, 3, 4, 5 }); // [B,H,M,K]
    const rhs = try b.param_tensor(.f32, &.{ 2, 3, 5, 6 }); // [B,H,K,N]
    const out = try b.dot_general(lhs, rhs, .{
        .lhs_batch_dims = &.{ 0, 1 },
        .rhs_batch_dims = &.{ 0, 1 },
        .lhs_contracting_dims = &.{3},
        .rhs_contracting_dims = &.{2},
    });

    const func = try b.finish(.{ .returns = &.{out} });
    const source_id = try program.add_function(func);

    const jvp_id = try jvp(std.testing.allocator, &program, source_id, "jvp", .{});
    const jvp_func = program.get_function_by_id(jvp_id).?;
    try pr.validate_function(jvp_func);

    const orig_t = func.returns[0].as_tensor();
    const jvp_t = jvp_func.returns[0].as_tensor();
    try std.testing.expectEqual(orig_t.dtype, jvp_t.dtype);
    try std.testing.expect(std.mem.eql(i64, orig_t.shape.dims, jvp_t.shape.dims));
}
