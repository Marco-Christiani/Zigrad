//! Compile-time function transforms for Tensor-valued functions.
//!
//! Transforms augment a function's behavior *during tracing*. They are called
//!  inside a traced function body, not at the trace call site.
//!
//! The canonical use is `value_and_grad`, which:
//!  1. Builds a sub-function from the differentiated closure,
//!  2. Applies `pr.ad.vjp` to produce a VJP function that includes values,
//!  3. Emits a call to the VJP function in the *outer* builder,
//!  4. Returns every source output and a `Tree(Tensor)` of gradients.
//!
//! Because the transform runs inside the trace, the caller can compose it
//!  with arbitrary traced logic (optimizer updates, gradient clipping, etc.)
//!  in a single compiled program:
//!
//! ```zig
//! fn train_step(params: Params, batch: Batch) !TrainResult {
//!     var vg = try transforms.value_and_grad(loss_fn, .{ params, batch }, .{});
//!     defer vg.deinit();
//!     // Apply optimizer in the same trace:
//!     var updated = try params_tree.map2(..., sgd_update);
//!     return .{ .loss = vg.outputs, .updated = updated.extract(Params) };
//! }
//! // Trace and compile the whole step as one program:
//! var traced = try zg.trace(train_step, allocator, specs, .{ .name = "train_step" });
//! defer traced.deinit();
//! const exe = try pipeline.run(zg.Executor.LoadedProgram, &traced.program, &ctx);
//! ```
const std = @import("std");
const pr = @import("pr");
const ad = @import("pr").ad;
const ops = @import("pr").ops;
const utils = @import("utils.zig");
const meta = utils.meta;
const Tree = utils.Tree;

const Tensor = @import("tensor.zig");

/// Options for Tensor-level differentiation transforms.
pub const GradOpts = struct {
    /// Output leaf whose cotangent is seeded with one.
    of: OutputSelector = .{ .leafnum = 0 },

    /// Zero-based function arguments whose Tensor leaves receive gradients.
    /// One selection preserves that argument's structure. Multiple selections
    ///  return a tuple in this order. Repeated argument numbers remain repeated.
    wrt_argnums: []const usize = &.{0},
};

/// Select one Tensor leaf from a differentiated function's output tree.
pub const OutputSelector = union(enum) {
    /// Dot-separated struct-field and array-index path.
    path: []const u8,
    /// Zero-based DFS leaf position.
    leafnum: usize,
};

/// Result of a `value_and_grad` call during tracing.
///
/// Use `.grads.extract(SelectedType)` to recover the selected structure, or
/// `.grads.leaves` for bulk operations like `map2` with an optimizer. This
/// value owns the gradient tree's metadata and must be deinitialized.
pub fn ValueAndGrad(comptime OutputsType: type) type {
    return struct {
        /// Every output from the differentiated function.
        outputs: meta.RuntimeOf(OutputsType),
        /// Selected argument gradients.
        grads: Tree(Tensor),

        pub fn deinit(self: *@This()) void {
            self.grads.deinit();
        }
    };
}

/// Trace `func` and compute its outputs and selected argument gradients.
///
/// Must be called during tracing.
///
/// For \(f: X_1 \times \cdots \times X_n \to
/// Y_1 \times \cdots \times Y_m\), let `opts.of` select a scalar output
/// \(f_o\), and let \(W = (w_1, \ldots, w_k)\) be `opts.wrt_argnums`. This
/// returns
///
/// $$
/// \left(f(x),
/// \left(\nabla_{x_{w_1}}f_o(x), \ldots,
/// \nabla_{x_{w_k}}f_o(x)\right)\right).
/// $$
///
/// All Tensor arguments must be bound to a `FunctionBuilder`. Use
///  `trace` and explicit compilation operations.
///
/// ## Mechanism
///
///  1. Flattens `args` into a `Tree(Tensor)` and collects their VarIds.
///  2. Builds a sub-function by tracing `func` with fresh parameters matching
///      the input specs.
///  3. Applies `pr.ad.vjp` to produce a VJP function that returns every source
///      output followed by the selected gradients.
///  4. Emits a `call` to the VJP function in the *outer* builder with a
///      ones-like cotangent seed.
///  5. Reconstructs the source output tree and returns it with a
///      `Tree(Tensor)` of gradients for the selected arguments.
///
/// ## Constraints
///
///  - `opts.of` must select a scalar Tensor output.
///  - `args` must be a nonempty tuple. `opts.wrt_argnums` selects tuple
///      elements, and gradients are returned for their Tensor leaves.
///
/// ```zig
/// fn train_step(params: Params, batch: Batch) !struct { loss: Tensor, updated: Params } {
///     var vg = try transforms.value_and_grad(loss_fn, .{ params, batch }, .{});
///     defer vg.deinit();
///     var updated = try params_tree.map2(Tensor, &vg.grads, Tensor, lr, sgd_leaf);
///     defer updated.deinit();
///     return .{ .loss = vg.outputs, .updated = updated.extract(Params) };
/// }
/// ```
pub fn value_and_grad(
    comptime func: anytype,
    args: anytype,
    comptime opts: GradOpts,
) !ValueAndGrad(CallableOutputType(func)) {
    const ArgsType = @TypeOf(args);
    const OutputsType = CallableOutputType(func);
    const GradsType = SelectedGradsType(ArgsType, opts.wrt_argnums);
    const output_leaf_count = comptime Tree(Tensor).leaf_count(OutputsType);
    const grad_leaf_count = comptime Tree(Tensor).leaf_count(GradsType);
    const selected_output = comptime selected_output_leaf(OutputsType, opts.of);

    // Extract builder from first tensor leaf in args.
    const builder = extract_builder(args) orelse @panic("no Tensor found in args");
    const program = builder.program;
    const alloc = program.allocator();
    const saved = program.checkpoint_appends();
    errdefer program.restore_appends(saved);

    // Flatten arguments into a tree.
    //
    // The program arena releases intermediate trees with the program.
    var args_tree = try Tree(Tensor).from(alloc, args);
    defer args_tree.deinit();

    // Collect Var pointers from the input tensors.
    const input_vars = try alloc.alloc(*pr.Var, args_tree.leaves.len);
    defer alloc.free(input_vars);
    for (args_tree.leaves, 0..) |t, i| {
        input_vars[i] = try t.get_var();
    }

    // Build a sub-function: create traced params matching the
    // spec shapes, reconstruct the structured args, and call func.
    const source_name = try program.reserve_unique_function_name("vg_source");
    var source_builder = try pr.FunctionBuilder.init(program, source_name);
    defer source_builder.deinit();

    var sub_tree = try args_tree.map(Tensor, &source_builder, struct {
        fn f(b: *pr.FunctionBuilder, spec: Tensor) !Tensor {
            return try Tensor.param(b, spec.dtype, spec.shape.const_slice());
        }
    }.f);
    defer sub_tree.deinit();

    const output_result = invoke_callable(func, try sub_tree.extract(ArgsType));
    const outputs = switch (@typeInfo(@TypeOf(output_result))) {
        .error_union => try output_result,
        else => output_result,
    };

    var output_tree = try Tree(Tensor).from(alloc, outputs);
    defer output_tree.deinit();
    const output_vars = try alloc.alloc(*pr.Var, output_leaf_count);
    defer alloc.free(output_vars);
    for (output_tree.leaves, output_vars) |tensor, *output_var| {
        output_var.* = try tensor.get_var();
    }
    const selected_var = output_vars[selected_output];
    if (selected_var.as_tensor().shape.rank() != 0) return error.NonScalarOutput;
    const source_id = try program.add_function(try source_builder.finish(.{ .returns = output_vars }));

    // Request gradients only for the selected argument's leaves.
    //  `wrt` filters the VJP function's output list: cotangents for
    //  non-`wrt` inputs are omitted from the return
    //  signature, so the VJP function returns exactly `grad_leaf_count`
    //  gradients. Any intermediate cotangents that only fed omitted outputs
    //  become dead and are cleaned up by the backend's DCE.
    const wrt_indices = comptime selected_leaf_indices(ArgsType, opts.wrt_argnums);
    const vjp_name = try program.reserve_unique_function_name("vg_vjp");
    const vjp_id = try ad.vjp(alloc, program, source_id, vjp_name, .{
        .of = &.{selected_output},
        .wrt = &wrt_indices,
        .include_primal_outputs = true,
    });

    // Emit a ones-like cotangent for the selected scalar output in the outer builder.
    const cot = try ad.emit_cotangent(builder, selected_var.as_tensor());

    // Call VJP function from outer builder. The VJP takes every primal
    //  input plus the selected output's cotangent seed.
    const total_leaf_count = args_tree.leaves.len;
    const call_args = try alloc.alloc(*pr.Var, total_leaf_count + 1);
    defer alloc.free(call_args);
    @memcpy(call_args[0..total_leaf_count], input_vars[0..total_leaf_count]);
    call_args[total_leaf_count] = cot;

    const call_outputs = (try builder.call(vjp_id, call_args)).outputs;
    if (call_outputs.len != output_leaf_count + grad_leaf_count) return error.UnexpectedOutputs;

    const output_leaves = try alloc.alloc(Tensor, output_leaf_count);
    defer alloc.free(output_leaves);
    for (call_outputs[0..output_leaf_count], output_leaves) |output_var, *tensor| {
        tensor.* = Tensor.from_var(builder, output_var);
    }
    const output_paths = comptime meta.tree_paths(Tensor, OutputsType);
    var result_tree = try Tree(Tensor).from_slices(alloc, output_leaves, &output_paths);
    defer result_tree.deinit();

    // Extract selected gradients into a Tree.
    const grad_leaves = try alloc.alloc(Tensor, grad_leaf_count);
    defer alloc.free(grad_leaves);
    for (call_outputs[output_leaf_count..], 0..) |gv, i| {
        grad_leaves[i] = Tensor.from_var(builder, gv);
    }

    const comptime_paths = comptime meta.tree_paths(Tensor, GradsType);

    return .{
        .outputs = try result_tree.extract(OutputsType),
        .grads = try Tree(Tensor).from_slices(alloc, grad_leaves, &comptime_paths),
    };
}

// Comptime function generators
//
// These produce comptime callables from existing functions. The returned
//  values can be passed to `zg.trace()`.
//
// Unlike the trace-time `value_and_grad` above (called inside a traced
//  function body), these are called at comptime to *generate* a function
//  that will itself be traced.

/// Generate a function that computes gradients of `func` for selected arguments.
///
/// Returns a comptime callable that accepts `func`'s arguments and produces
/// gradients with the structure described by `GradOpts`. Pass the returned
/// callable to `zg.trace()`.
///
/// Using the notation from `value_and_grad`, the generated callable returns
///
/// $$
/// \left(\nabla_{x_{w_1}}f_o(x), \ldots,
/// \nabla_{x_{w_k}}f_o(x)\right).
/// $$
///
/// ```zig
/// const grad_fn = comptime zg.grad(loss_fn, .{});
/// var traced = try zg.trace(grad_fn, allocator, specs, .{ .name = "grad_step" });
/// defer traced.deinit();
/// ```
pub fn make_grad(comptime func: anytype, comptime opts: GradOpts) GeneratedCall(.grad, func, opts) {
    return .{};
}

/// Generate a function that computes both value and gradients of `func`.
///
/// Returns a comptime callable that accepts `func`'s arguments. Its result
/// contains the outputs and gradients shaped according to `GradOpts`.
///
/// ```zig
/// const vg_fn = comptime zg.value_and_grad(loss_fn, .{});
/// var traced = try zg.trace(vg_fn, allocator, specs, .{ .name = "vg_step" });
/// defer traced.deinit();
/// ```
pub fn make_value_and_grad(comptime func: anytype, comptime opts: GradOpts) GeneratedCall(.value_and_grad, func, opts) {
    return .{};
}

/// Return type for comptime `value_and_grad` generated functions.
///
/// Outputs and gradients are reconstructed into their source structures. The
/// result owns no tree metadata and requires no `deinit` call.
pub fn ValueAndGradResult(comptime OutputsType: type, comptime GradsType: type) type {
    return struct { outputs: meta.RuntimeOf(OutputsType), grads: GradsType };
}

const GeneratedTransform = enum { grad, value_and_grad };

fn GeneratedResult(comptime transform: GeneratedTransform, comptime func: anytype, comptime GradsType: type) type {
    return switch (transform) {
        .grad => GradsType,
        .value_and_grad => ValueAndGradResult(CallableOutputType(func), GradsType),
    };
}

fn GeneratedCall(comptime transform: GeneratedTransform, comptime func: anytype, comptime opts: GradOpts) type {
    const Args = CallableArgsType(func);
    const Grads = SelectedGradsType(Args, opts.wrt_argnums);
    return struct {
        pub const ArgsType = Args;
        pub const ResultType = GeneratedResult(transform, func, Grads);

        pub fn call(args: ArgsType) anyerror!ResultType {
            return try generated_impl(transform, func, Grads, args, opts);
        }
    };
}

fn generated_impl(
    comptime transform: GeneratedTransform,
    comptime func: anytype,
    comptime GradsType: type,
    args: anytype,
    comptime opts: GradOpts,
) !GeneratedResult(transform, func, GradsType) {
    var vg = try value_and_grad(func, args, opts);
    defer vg.deinit();
    return switch (transform) {
        .grad => try vg.grads.extract(GradsType),
        .value_and_grad => .{
            .outputs = vg.outputs,
            .grads = try vg.grads.extract(GradsType),
        },
    };
}

// Internal helpers

fn CallableOutputType(comptime func: anytype) type {
    const ReturnType = switch (@typeInfo(@TypeOf(func))) {
        .@"fn" => |info| info.return_type orelse
            @compileError("differentiated function must return a value"),
        .@"struct" => if (@hasDecl(@TypeOf(func), "ResultType"))
            @TypeOf(func).ResultType
        else
            @compileError("differentiated callable must declare ResultType"),
        else => @compileError("differentiated value is not callable"),
    };
    return switch (@typeInfo(ReturnType)) {
        .error_union => |info| info.payload,
        else => ReturnType,
    };
}

fn CallableArgsType(comptime func: anytype) type {
    return switch (@typeInfo(@TypeOf(func))) {
        .@"fn" => std.meta.ArgsTuple(@TypeOf(func)),
        .@"struct" => if (@hasDecl(@TypeOf(func), "ArgsType"))
            @TypeOf(func).ArgsType
        else
            @compileError("differentiated callable must declare ArgsType"),
        else => @compileError("differentiated value is not callable"),
    };
}

fn invoke_callable(comptime func: anytype, args: CallableArgsType(func)) anyerror!CallableOutputType(func) {
    const result = switch (@typeInfo(@TypeOf(func))) {
        .@"fn" => @call(.auto, func, args),
        .@"struct" => @TypeOf(func).call(args),
        else => unreachable,
    };
    return switch (@typeInfo(@TypeOf(result))) {
        .error_union => try result,
        else => result,
    };
}

fn selected_output_leaf(comptime OutputsType: type, comptime selector: OutputSelector) usize {
    const paths = comptime meta.tree_paths(Tensor, OutputsType);
    return switch (selector) {
        .leafnum => |leafnum| if (leafnum < paths.len)
            leafnum
        else
            @compileError("output leaf number is out of range"),
        .path => |selected_path| {
            for (paths, 0..) |path, index| {
                if (std.mem.eql(u8, path, selected_path)) return index;
            }
            @compileError("output path does not name a Tensor leaf");
        },
    };
}

fn SelectedGradsType(comptime ArgsType: type, comptime argnums: []const usize) type {
    const info = @typeInfo(ArgsType);
    if (info != .@"struct" or !info.@"struct".is_tuple or info.@"struct".fields.len == 0) {
        @compileError("args must be a tuple with at least one element");
    }
    if (argnums.len == 0) @compileError("wrt_argnums must not be empty");
    inline for (argnums) |argnum| {
        if (argnum >= info.@"struct".fields.len) @compileError("wrt_argnums contains an out-of-range argument");
    }
    if (argnums.len == 1) return info.@"struct".fields[argnums[0]].type;

    comptime var types: [argnums.len]type = undefined;
    inline for (argnums, 0..) |argnum, index| {
        types[index] = info.@"struct".fields[argnum].type;
    }
    return std.meta.Tuple(&types);
}

fn argument_leaf_offset(comptime ArgsType: type, comptime argument: usize) usize {
    const fields = @typeInfo(ArgsType).@"struct".fields;
    comptime var offset: usize = 0;
    inline for (fields[0..argument]) |field| {
        offset += Tree(Tensor).leaf_count(field.type);
    }
    return offset;
}

fn selected_leaf_indices(
    comptime ArgsType: type,
    comptime argnums: []const usize,
) [Tree(Tensor).leaf_count(SelectedGradsType(ArgsType, argnums))]usize {
    const fields = @typeInfo(ArgsType).@"struct".fields;
    var indices: [Tree(Tensor).leaf_count(SelectedGradsType(ArgsType, argnums))]usize = undefined;
    var cursor: usize = 0;
    inline for (argnums) |argnum| {
        const offset = comptime argument_leaf_offset(ArgsType, argnum);
        const count = comptime Tree(Tensor).leaf_count(fields[argnum].type);
        for (0..count) |index| {
            indices[cursor] = offset + index;
            cursor += 1;
        }
    }
    return indices;
}

/// Extract the FunctionBuilder pointer from the first Tensor leaf in a structured value.
fn extract_builder(val: anytype) ?*pr.FunctionBuilder {
    const T = @TypeOf(val);
    if (T == Tensor) return val.backing.traced.builder;
    switch (@typeInfo(T)) {
        .@"struct" => |info| {
            inline for (info.fields) |field| {
                if (extract_builder(@field(val, field.name))) |b| return b;
            }
        },
        .array => |info| {
            comptime var i: usize = 0;
            inline while (i < info.len) : (i += 1) {
                if (extract_builder(val[i])) |b| return b;
            }
        },
        else => {},
    }
    return null;
}

// Tests.

const trace = @import("trace.zig").trace;

const TestParams = struct { w: Tensor, b: Tensor };
const TestBatch = struct { x: Tensor };

fn test_loss(params: TestParams, batch: TestBatch) !Tensor {
    const z = try batch.x.mm(params.w);
    const b_broadcast = try params.b.broadcast_in_dim(&.{ 3, 2 }, &.{1});
    const pred = try z.add(b_broadcast);
    return try pred.reduce(.{ .axes = &.{ 0, 1 }, .operation = .sum });
}

const TestOutputs = struct {
    loss: Tensor,
    metrics: struct { doubled: Tensor },
};

fn test_outputs(params: TestParams, batch: TestBatch) !TestOutputs {
    const loss = try test_loss(params, batch);
    return .{
        .loss = loss,
        .metrics = .{ .doubled = try loss.add(loss) },
    };
}

fn five_arg_loss(a: Tensor, b: Tensor, c: Tensor, d: Tensor, e: Tensor) !Tensor {
    return try (try (try (try a.add(b)).add(c)).add(d)).add(e);
}

fn unsupported_loss(input: Tensor) !Tensor {
    const builder = switch (input.backing) {
        .traced => |traced| traced.builder,
        else => return error.UnsupportedAval,
    };
    const input_var = try input.get_var();
    const outputs = (try builder.custom_call(.{
        .target_name = "test.missing_vjp",
        .has_side_effect = false,
        .payload = &.{},
    }, &.{input_var}, &.{input_var.aval})).outputs;
    return Tensor.from_var(builder, outputs[0]);
}

test "value_and_grad restores registered functions after failure" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();
    var builder = try pr.FunctionBuilder.init(&program, "outer");
    defer builder.deinit();
    const input = try Tensor.param(&builder, .f32, &.{});

    try std.testing.expectError(error.MissingDerivativeRule, value_and_grad(unsupported_loss, .{input}, .{}));
    try std.testing.expectEqual(@as(usize, 0), program.functions().len);
    try std.testing.expectEqual(@as(usize, 0), program.checkpoint_appends().reservation_count);
    try std.testing.expectError(error.MissingDerivativeRule, value_and_grad(unsupported_loss, .{input}, .{}));
    try std.testing.expectEqual(@as(usize, 0), program.functions().len);
    try std.testing.expectEqual(@as(usize, 0), program.checkpoint_appends().reservation_count);
}

test make_grad {
    const grad_fn = comptime make_grad(test_loss, .{});

    // Verify return type is the params type (TestParams).
    const Callable = @TypeOf(grad_fn);
    try std.testing.expectEqual(@as(usize, 2), @typeInfo(Callable.ArgsType).@"struct".fields.len);
    try std.testing.expect(Callable.ResultType == TestParams);

    // Trace the generated function into a complete PR program.
    const specs = .{
        TestParams{
            .w = Tensor.abstract(.f32, &.{ 4, 2 }),
            .b = Tensor.abstract(.f32, &.{2}),
        },
        TestBatch{
            .x = Tensor.abstract(.f32, &.{ 3, 4 }),
        },
    };

    var traced = try trace(grad_fn, std.testing.allocator, specs, .{ .name = "grad_test" });
    defer traced.deinit();

    // grad returns only grads for first arg (2 leaves: w, b).
    const grad_test = traced.program.get_function_by_id(try traced.program.resolve_entry()).?;
    try std.testing.expectEqual(2, grad_test.returns.len);
    try std.testing.expectEqual(3, grad_test.params.len);
}

test "make_grad selects differentiated arguments in order" {
    const grad_fn = comptime make_grad(test_loss, .{ .wrt_argnums = &.{ 1, 0, 1 } });
    const return_type = @TypeOf(grad_fn).ResultType;
    const return_fields = @typeInfo(return_type).@"struct".fields;
    try std.testing.expectEqual(@as(usize, 3), return_fields.len);
    try std.testing.expect(return_fields[0].type == TestBatch);
    try std.testing.expect(return_fields[1].type == TestParams);
    try std.testing.expect(return_fields[2].type == TestBatch);

    const specs = .{
        TestParams{
            .w = Tensor.abstract(.f32, &.{ 4, 2 }),
            .b = Tensor.abstract(.f32, &.{2}),
        },
        TestBatch{
            .x = Tensor.abstract(.f32, &.{ 3, 4 }),
        },
    };
    var traced = try trace(grad_fn, std.testing.allocator, specs, .{ .name = "batch_grad_test" });
    defer traced.deinit();

    const function = traced.program.get_function_by_id(try traced.program.resolve_entry()).?;
    try std.testing.expectEqual(@as(usize, 4), function.returns.len);
    try std.testing.expectEqualSlices(i64, &.{ 3, 4 }, function.returns[0].as_tensor().shape.dims);
    try std.testing.expectEqualSlices(i64, &.{ 4, 2 }, function.returns[1].as_tensor().shape.dims);
    try std.testing.expectEqualSlices(i64, &.{2}, function.returns[2].as_tensor().shape.dims);
    try std.testing.expectEqualSlices(i64, &.{ 3, 4 }, function.returns[3].as_tensor().shape.dims);
}

test "make_grad has no positional arity ceiling" {
    const grad_fn = comptime make_grad(five_arg_loss, .{});
    const scalar = Tensor.abstract(.f32, &.{});
    const specs = .{ scalar, scalar, scalar, scalar, scalar };

    var traced = try trace(grad_fn, std.testing.allocator, specs, .{ .name = "five_arg_grad_test" });
    defer traced.deinit();

    const five_arg_grad_test = traced.program.get_function_by_id(try traced.program.resolve_entry()).?;
    try std.testing.expectEqual(@as(usize, 1), five_arg_grad_test.returns.len);
    try std.testing.expectEqual(@as(usize, 5), five_arg_grad_test.params.len);
}

test make_value_and_grad {
    const vg_fn = comptime make_value_and_grad(test_loss, .{});

    // Verify return type is ValueAndGradResult(Tensor, TestParams).
    const ReturnType = @TypeOf(vg_fn).ResultType;
    try std.testing.expect(@hasField(ReturnType, "outputs"));
    try std.testing.expect(@hasField(ReturnType, "grads"));

    const specs = .{
        TestParams{
            .w = Tensor.abstract(.f32, &.{ 4, 2 }),
            .b = Tensor.abstract(.f32, &.{2}),
        },
        TestBatch{
            .x = Tensor.abstract(.f32, &.{ 3, 4 }),
        },
    };

    var traced = try trace(vg_fn, std.testing.allocator, specs, .{ .name = "vg_test" });
    defer traced.deinit();

    // value_and_grad returns value (1 tensor) + grads for first arg (2 leaves).
    const vg_test = traced.program.get_function_by_id(try traced.program.resolve_entry()).?;
    try std.testing.expectEqual(3, vg_test.returns.len);
    try std.testing.expectEqual(3, vg_test.params.len);
}

test "make_value_and_grad preserves structured outputs and selects by path" {
    const vg_fn = comptime make_value_and_grad(test_outputs, .{
        .of = .{ .path = "metrics.doubled" },
    });

    const ReturnType = @TypeOf(vg_fn).ResultType;
    try std.testing.expect(@TypeOf(@as(ReturnType, undefined).outputs) == TestOutputs);

    const specs = .{
        TestParams{
            .w = Tensor.abstract(.f32, &.{ 4, 2 }),
            .b = Tensor.abstract(.f32, &.{2}),
        },
        TestBatch{
            .x = Tensor.abstract(.f32, &.{ 3, 4 }),
        },
    };

    var traced = try trace(vg_fn, std.testing.allocator, specs, .{ .name = "structured_vg_test" });
    defer traced.deinit();

    const structured_vg_test = traced.program.get_function_by_id(try traced.program.resolve_entry()).?;
    try std.testing.expectEqual(@as(usize, 4), structured_vg_test.returns.len);
    try std.testing.expectEqual(@as(usize, 3), structured_vg_test.params.len);
}
