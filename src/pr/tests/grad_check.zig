//! Finite-difference gradient checker for PR automatic differentiation.
//!
//! Compares analytic gradients (from VJP) against numeric gradients
//!  (central finite differences) to verify AD correctness.
const std = @import("std");
const pr = @import("../pr.zig");
const pr_eval = @import("pr_eval");
const ad = @import("../ad.zig");
const log = std.log.scoped(.@"zg/grad_check");

pub const HostTensor = pr_eval.HostTensor;

pub const GradCheckOpts = struct {
    epsilon: f32 = 1e-3,
    tolerance: f32 = 1e-2,
};

pub const GradCheckError = pr_eval.EvalError || ad.AdError || error{GradientMismatch};

/// Verify that analytic gradients (VJP) match numeric gradients (finite differences).
///
/// 1. Generates VJP function via `ad.vjp`.
/// 2. Evaluates VJP to get analytic gradients.
/// 3. For each input element, computes numeric gradient via central differences.
/// 4. Compares analytic vs numeric within tolerance.
pub fn check_gradients(
    allocator: std.mem.Allocator,
    program: *pr.Program,
    source: pr.FunctionId,
    inputs: []const HostTensor,
    opts: GradCheckOpts,
) GradCheckError!void {
    const func = program.get_function_by_id(source) orelse
        return error.CallUnresolvedCallee;

    // 1. Generate VJP function
    const vjp_id = try ad.vjp(allocator, program, source, "grad_check_vjp", .{});
    const vjp_func = program.get_function_by_id(vjp_id) orelse
        return error.CallUnresolvedCallee;

    // 2. Compute analytic gradients
    //    VJP params: [N primals, M cotangents], returns: [N grads]
    const vjp_inputs = try allocator.alloc(HostTensor, func.params.len + func.returns.len);
    defer allocator.free(vjp_inputs);

    // Primals = inputs
    for (inputs, 0..) |inp, i| {
        vjp_inputs[i] = inp;
    }

    // Cotangents = ones matching each output shape
    // First, eval forward to get output shapes
    const fwd_results = try pr_eval.eval(allocator, program, func, inputs);
    defer {
        for (fwd_results) |*r| {
            var tmp = r.*;
            tmp.deinit();
        }
        allocator.free(fwd_results);
    }

    // Create cotangent tensors (ones)
    var cot_tensors = try allocator.alloc(HostTensor, func.returns.len);
    defer {
        for (cot_tensors) |*ct| ct.deinit();
        allocator.free(cot_tensors);
    }
    for (fwd_results, 0..) |fwd_r, i| {
        cot_tensors[i] = try HostTensor.init(allocator, fwd_r.shape);
        cot_tensors[i].fill(1.0);
        vjp_inputs[func.params.len + i] = cot_tensors[i];
    }

    // Evaluate VJP
    const analytic_grads = try pr_eval.eval(allocator, program, vjp_func, vjp_inputs);
    defer {
        for (analytic_grads) |*g| {
            var tmp = g.*;
            tmp.deinit();
        }
        allocator.free(analytic_grads);
    }

    // 3. Compute numeric gradients via central finite differences
    // Clone inputs so we can perturb them
    var perturbed = try allocator.alloc(HostTensor, inputs.len);
    defer {
        for (perturbed) |*p| p.deinit();
        allocator.free(perturbed);
    }
    for (inputs, 0..) |inp, i| {
        perturbed[i] = try inp.clone();
    }

    for (inputs, 0..) |_, input_idx| {
        // Skip non-float inputs (e.g., integer indices for gather/scatter)
        const param_var = func.params[input_idx];
        const param_tensor = param_var.aval.as_tensor();
        switch (param_tensor.dtype) {
            .f16, .bf16, .f32, .f64 => {},
            else => continue,
        }

        const n_elem = perturbed[input_idx].data.len;
        for (0..n_elem) |elem_idx| {
            const original = perturbed[input_idx].data[elem_idx];

            // f(x + eps)
            perturbed[input_idx].data[elem_idx] = original + opts.epsilon;
            const f_plus = try eval_scalar_sum(allocator, program, func, perturbed);

            // f(x - eps)
            perturbed[input_idx].data[elem_idx] = original - opts.epsilon;
            const f_minus = try eval_scalar_sum(allocator, program, func, perturbed);

            // Restore
            perturbed[input_idx].data[elem_idx] = original;

            const numeric = (f_plus - f_minus) / (2.0 * opts.epsilon);
            const analytic = analytic_grads[input_idx].data[elem_idx];

            // Relative + absolute tolerance check
            const diff = @abs(analytic - numeric);
            const scale = @max(@abs(analytic), @abs(numeric));
            const rel_ok = if (scale > 1e-6) diff / scale < opts.tolerance else true;
            const abs_ok = diff < opts.tolerance;

            if (!rel_ok and !abs_ok) {
                log.err(
                    "gradient mismatch: input[{d}][{d}] analytic={e:.6} numeric={e:.6} diff={e:.6}",
                    .{ input_idx, elem_idx, analytic, numeric, diff },
                );
                return error.GradientMismatch;
            }
        }
    }
}

/// Evaluate func on inputs and return the sum of all output elements as a scalar.
fn eval_scalar_sum(
    allocator: std.mem.Allocator,
    program: *const pr.Program,
    func: pr.Function,
    inputs: []const HostTensor,
) pr_eval.EvalError!f32 {
    const results = try pr_eval.eval(allocator, program, func, inputs);
    defer {
        for (results) |*r| {
            var tmp = r.*;
            tmp.deinit();
        }
        allocator.free(results);
    }
    var total: f32 = 0;
    for (results) |r| {
        for (r.data) |v| total += v;
    }
    return total;
}

/// Generate deterministic test inputs in [0.5, 2.0] range.
///
/// Avoids zeros (divide VJP), negatives (log VJP), and values that cause
/// vanishing gradients.
pub fn make_test_inputs(
    allocator: std.mem.Allocator,
    func: pr.Function,
) ![]HostTensor {
    const inputs = try allocator.alloc(HostTensor, func.params.len);
    errdefer {
        for (inputs[0..]) |*inp| inp.deinit();
        allocator.free(inputs);
    }

    for (func.params, 0..) |param_var, i| {
        const tensor = param_var.as_tensor();
        inputs[i] = try HostTensor.init(allocator, tensor.shape.dims);
        const n = inputs[i].data.len;
        for (0..n) |j| {
            // Deterministic values in [0.5, 2.0]
            const seed: f32 = @floatFromInt((i * 137 + j * 31 + 17) % 256);
            inputs[i].data[j] = 0.5 + (seed / 256.0) * 1.5;
        }
    }
    return inputs;
}

// ============================================================================
// Helper to build simple functions for testing
// ============================================================================

fn build_func(program: *pr.Program, b: *pr.FunctionBuilder, returns: []const *pr.Var) !pr.FunctionId {
    return try program.add_function(try b.finish(.{ .returns = returns }));
}

// ============================================================================
// Priority 1 - Bug regression tests
// ============================================================================

test "grad: gather with 1D start_index_map" {
    // Gather extracts a contiguous slice from operand.
    // We test the gradient w.r.t. the operand (data), not the indices.
    // The VJP for gather produces a scatter-add backward.
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "gather_1d");
    defer b.deinit();
    const operand = try b.param_tensor(.f32, &.{5});
    const indices = try b.param_tensor(.i32, &.{ 3, 1 });
    const out = try b.gather(operand, indices, .{
        .slice_sizes = &.{1},
        .offset_dims = &.{},
        .collapsed_slice_dims = &.{0},
        .start_index_map = &.{0},
        .index_vector_dim = 1,
    });
    const func_id = try build_func(&program, &b, &.{out});

    // Make test inputs - operand gets [0.5, 2.0] values, indices get integer values
    var in_op = try HostTensor.init_with_data(std.testing.allocator, &.{5}, &.{ 1.0, 1.5, 0.8, 1.2, 0.6 });
    defer in_op.deinit();
    var in_idx = try HostTensor.init_with_data(std.testing.allocator, &.{ 3, 1 }, &.{ 0, 2, 4 });
    defer in_idx.deinit();

    try check_gradients(std.testing.allocator, &program, func_id, &.{ in_op, in_idx }, .{});
}

test "grad: dot_general batched [B,H,M,K] x [B,H,K,N]" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "dg_batched");
    defer b.deinit();
    const lhs = try b.param_tensor(.f32, &.{ 2, 2, 3, 4 });
    const rhs = try b.param_tensor(.f32, &.{ 2, 2, 4, 3 });
    const out = try b.dot_general(lhs, rhs, .{
        .lhs_batch_dims = &.{ 0, 1 },
        .rhs_batch_dims = &.{ 0, 1 },
        .lhs_contracting_dims = &.{3},
        .rhs_contracting_dims = &.{2},
    });
    const func_id = try build_func(&program, &b, &.{out});
    const func = program.get_function_by_id(func_id).?;

    const test_inputs = try make_test_inputs(std.testing.allocator, func);
    defer {
        for (test_inputs) |*t| t.deinit();
        std.testing.allocator.free(test_inputs);
    }

    try check_gradients(std.testing.allocator, &program, func_id, test_inputs, .{});
}

test "grad: dot_general non-prefix batch dims" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "dg_nonprefix");
    defer b.deinit();
    const lhs = try b.param_tensor(.f32, &.{ 2, 3, 2, 4 });
    const rhs = try b.param_tensor(.f32, &.{ 2, 3, 2, 4 });
    const out = try b.dot_general(lhs, rhs, .{
        .lhs_batch_dims = &.{ 0, 2 },
        .rhs_batch_dims = &.{ 0, 2 },
        .lhs_contracting_dims = &.{3},
        .rhs_contracting_dims = &.{3},
    });
    const func_id = try build_func(&program, &b, &.{out});
    const func = program.get_function_by_id(func_id).?;

    const test_inputs = try make_test_inputs(std.testing.allocator, func);
    defer {
        for (test_inputs) |*t| t.deinit();
        std.testing.allocator.free(test_inputs);
    }

    try check_gradients(std.testing.allocator, &program, func_id, test_inputs, .{});
}

test "grad: dot_general multi-contract dims" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "dg_multicontract");
    defer b.deinit();
    const lhs = try b.param_tensor(.f32, &.{ 2, 3, 4, 5 });
    const rhs = try b.param_tensor(.f32, &.{ 2, 3, 4, 5 });
    const out = try b.dot_general(lhs, rhs, .{
        .lhs_batch_dims = &.{ 0, 1 },
        .rhs_batch_dims = &.{ 0, 1 },
        .lhs_contracting_dims = &.{ 2, 3 },
        .rhs_contracting_dims = &.{ 2, 3 },
    });
    const func_id = try build_func(&program, &b, &.{out});
    const func = program.get_function_by_id(func_id).?;

    const test_inputs = try make_test_inputs(std.testing.allocator, func);
    defer {
        for (test_inputs) |*t| t.deinit();
        std.testing.allocator.free(test_inputs);
    }

    try check_gradients(std.testing.allocator, &program, func_id, test_inputs, .{});
}

// ============================================================================
// Priority 2 - Systematic per-op coverage
// ============================================================================

test "grad: add" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "add");
    defer b.deinit();
    const x = try b.param_tensor(.f32, &.{ 2, 3 });
    const y = try b.param_tensor(.f32, &.{ 2, 3 });
    const z = try b.add(x, y);
    const func_id = try build_func(&program, &b, &.{z});
    const func = program.get_function_by_id(func_id).?;

    const test_inputs = try make_test_inputs(std.testing.allocator, func);
    defer {
        for (test_inputs) |*t| t.deinit();
        std.testing.allocator.free(test_inputs);
    }

    try check_gradients(std.testing.allocator, &program, func_id, test_inputs, .{});
}

test "grad: subtract" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "sub");
    defer b.deinit();
    const x = try b.param_tensor(.f32, &.{ 2, 3 });
    const y = try b.param_tensor(.f32, &.{ 2, 3 });
    const z = try b.subtract(x, y);
    const func_id = try build_func(&program, &b, &.{z});
    const func = program.get_function_by_id(func_id).?;

    const test_inputs = try make_test_inputs(std.testing.allocator, func);
    defer {
        for (test_inputs) |*t| t.deinit();
        std.testing.allocator.free(test_inputs);
    }

    try check_gradients(std.testing.allocator, &program, func_id, test_inputs, .{});
}

test "grad: multiply" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "mul");
    defer b.deinit();
    const x = try b.param_tensor(.f32, &.{ 2, 3 });
    const y = try b.param_tensor(.f32, &.{ 2, 3 });
    const z = try b.multiply(x, y);
    const func_id = try build_func(&program, &b, &.{z});
    const func = program.get_function_by_id(func_id).?;

    const test_inputs = try make_test_inputs(std.testing.allocator, func);
    defer {
        for (test_inputs) |*t| t.deinit();
        std.testing.allocator.free(test_inputs);
    }

    try check_gradients(std.testing.allocator, &program, func_id, test_inputs, .{});
}

test "grad: divide" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "div");
    defer b.deinit();
    const x = try b.param_tensor(.f32, &.{ 2, 3 });
    const y = try b.param_tensor(.f32, &.{ 2, 3 });
    const z = try b.divide(x, y);
    const func_id = try build_func(&program, &b, &.{z});
    const func = program.get_function_by_id(func_id).?;

    const test_inputs = try make_test_inputs(std.testing.allocator, func);
    defer {
        for (test_inputs) |*t| t.deinit();
        std.testing.allocator.free(test_inputs);
    }

    try check_gradients(std.testing.allocator, &program, func_id, test_inputs, .{});
}

test "grad: exp" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "exp");
    defer b.deinit();
    const x = try b.param_tensor(.f32, &.{ 2, 3 });
    const z = try b.exp(x);
    const func_id = try build_func(&program, &b, &.{z});
    const func = program.get_function_by_id(func_id).?;

    const test_inputs = try make_test_inputs(std.testing.allocator, func);
    defer {
        for (test_inputs) |*t| t.deinit();
        std.testing.allocator.free(test_inputs);
    }

    try check_gradients(std.testing.allocator, &program, func_id, test_inputs, .{});
}

test "grad: log" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "log");
    defer b.deinit();
    const x = try b.param_tensor(.f32, &.{ 2, 3 });
    const z = try b.log(x);
    const func_id = try build_func(&program, &b, &.{z});
    const func = program.get_function_by_id(func_id).?;

    const test_inputs = try make_test_inputs(std.testing.allocator, func);
    defer {
        for (test_inputs) |*t| t.deinit();
        std.testing.allocator.free(test_inputs);
    }

    try check_gradients(std.testing.allocator, &program, func_id, test_inputs, .{});
}

test "grad: rsqrt" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "rsqrt");
    defer b.deinit();
    const x = try b.param_tensor(.f32, &.{ 2, 3 });
    const z = try b.rsqrt(x);
    const func_id = try build_func(&program, &b, &.{z});
    const func = program.get_function_by_id(func_id).?;

    const test_inputs = try make_test_inputs(std.testing.allocator, func);
    defer {
        for (test_inputs) |*t| t.deinit();
        std.testing.allocator.free(test_inputs);
    }

    try check_gradients(std.testing.allocator, &program, func_id, test_inputs, .{});
}

test "grad: logistic" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "logistic");
    defer b.deinit();
    const x = try b.param_tensor(.f32, &.{ 2, 3 });
    const z = try b.logistic(x);
    const func_id = try build_func(&program, &b, &.{z});
    const func = program.get_function_by_id(func_id).?;

    const test_inputs = try make_test_inputs(std.testing.allocator, func);
    defer {
        for (test_inputs) |*t| t.deinit();
        std.testing.allocator.free(test_inputs);
    }

    try check_gradients(std.testing.allocator, &program, func_id, test_inputs, .{});
}

test "grad: dot" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "dot");
    defer b.deinit();
    const x = try b.param_tensor(.f32, &.{3});
    const y = try b.param_tensor(.f32, &.{3});
    const z = try b.dot(x, y);
    const func_id = try build_func(&program, &b, &.{z});
    const func = program.get_function_by_id(func_id).?;

    const test_inputs = try make_test_inputs(std.testing.allocator, func);
    defer {
        for (test_inputs) |*t| t.deinit();
        std.testing.allocator.free(test_inputs);
    }

    try check_gradients(std.testing.allocator, &program, func_id, test_inputs, .{});
}

test "grad: mm" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "mm");
    defer b.deinit();
    const x = try b.param_tensor(.f32, &.{ 2, 3 });
    const y = try b.param_tensor(.f32, &.{ 3, 2 });
    const z = try b.mm(x, y);
    const func_id = try build_func(&program, &b, &.{z});
    const func = program.get_function_by_id(func_id).?;

    const test_inputs = try make_test_inputs(std.testing.allocator, func);
    defer {
        for (test_inputs) |*t| t.deinit();
        std.testing.allocator.free(test_inputs);
    }

    try check_gradients(std.testing.allocator, &program, func_id, test_inputs, .{});
}

test "grad: bmm" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "bmm");
    defer b.deinit();
    const x = try b.param_tensor(.f32, &.{ 2, 2, 3 });
    const y = try b.param_tensor(.f32, &.{ 2, 3, 2 });
    const z = try b.bmm(x, y);
    const func_id = try build_func(&program, &b, &.{z});
    const func = program.get_function_by_id(func_id).?;

    const test_inputs = try make_test_inputs(std.testing.allocator, func);
    defer {
        for (test_inputs) |*t| t.deinit();
        std.testing.allocator.free(test_inputs);
    }

    try check_gradients(std.testing.allocator, &program, func_id, test_inputs, .{});
}

test "grad: convolution NHWC HWIO" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "convolution");
    defer b.deinit();
    const x = try b.param_tensor(.f32, &.{ 1, 4, 4, 2 });
    const kernel = try b.param_tensor(.f32, &.{ 3, 3, 2, 3 });
    const y = try b.convolution(x, kernel, .{
        .window_strides = &.{ 1, 1 },
        .padding = &.{ 1, 1, 1, 1 },
        .lhs_dilation = &.{ 1, 1 },
        .rhs_dilation = &.{ 1, 1 },
        .window_reversal = &.{ false, false },
        .dimensions = .{
            .input_batch_dimension = 0,
            .input_feature_dimension = 3,
            .input_spatial_dimensions = &.{ 1, 2 },
            .kernel_input_feature_dimension = 2,
            .kernel_output_feature_dimension = 3,
            .kernel_spatial_dimensions = &.{ 0, 1 },
            .output_batch_dimension = 0,
            .output_feature_dimension = 3,
            .output_spatial_dimensions = &.{ 1, 2 },
        },
    });
    const func_id = try build_func(&program, &b, &.{y});
    const func = program.get_function_by_id(func_id).?;
    const test_inputs = try make_test_inputs(std.testing.allocator, func);
    defer {
        for (test_inputs) |*t| t.deinit();
        std.testing.allocator.free(test_inputs);
    }
    try check_gradients(std.testing.allocator, &program, func_id, test_inputs, .{
        .epsilon = 2e-3,
        .tolerance = 2e-2,
    });
}

test "grad: strided convolution NHWC HWIO" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "convolution_stride");
    defer b.deinit();
    const x = try b.param_tensor(.f32, &.{ 1, 5, 5, 1 });
    const kernel = try b.param_tensor(.f32, &.{ 3, 3, 1, 2 });
    const y = try b.convolution(x, kernel, .{
        .window_strides = &.{ 2, 2 },
        .padding = &.{ 1, 1, 1, 1 },
        .lhs_dilation = &.{ 1, 1 },
        .rhs_dilation = &.{ 1, 1 },
        .window_reversal = &.{ false, false },
        .dimensions = .{
            .input_batch_dimension = 0,
            .input_feature_dimension = 3,
            .input_spatial_dimensions = &.{ 1, 2 },
            .kernel_input_feature_dimension = 2,
            .kernel_output_feature_dimension = 3,
            .kernel_spatial_dimensions = &.{ 0, 1 },
            .output_batch_dimension = 0,
            .output_feature_dimension = 3,
            .output_spatial_dimensions = &.{ 1, 2 },
        },
    });
    const func_id = try build_func(&program, &b, &.{y});
    const func = program.get_function_by_id(func_id).?;
    const test_inputs = try make_test_inputs(std.testing.allocator, func);
    defer {
        for (test_inputs) |*t| t.deinit();
        std.testing.allocator.free(test_inputs);
    }
    try check_gradients(std.testing.allocator, &program, func_id, test_inputs, .{
        .epsilon = 2e-3,
        .tolerance = 2e-2,
    });
}

test "grad: reshape" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "reshape");
    defer b.deinit();
    const x = try b.param_tensor(.f32, &.{ 2, 3 });
    const y = try b.reshape(x, &.{ 3, 2 });
    const func_id = try build_func(&program, &b, &.{y});
    const func = program.get_function_by_id(func_id).?;

    const test_inputs = try make_test_inputs(std.testing.allocator, func);
    defer {
        for (test_inputs) |*t| t.deinit();
        std.testing.allocator.free(test_inputs);
    }

    try check_gradients(std.testing.allocator, &program, func_id, test_inputs, .{});
}

test "grad: transpose" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "transpose");
    defer b.deinit();
    const x = try b.param_tensor(.f32, &.{ 2, 3 });
    const y = try b.transpose(x, &.{ 1, 0 });
    const func_id = try build_func(&program, &b, &.{y});
    const func = program.get_function_by_id(func_id).?;

    const test_inputs = try make_test_inputs(std.testing.allocator, func);
    defer {
        for (test_inputs) |*t| t.deinit();
        std.testing.allocator.free(test_inputs);
    }

    try check_gradients(std.testing.allocator, &program, func_id, test_inputs, .{});
}

test "grad: broadcast_in_dim scalar to matrix" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "bcast_scalar");
    defer b.deinit();
    const x = try b.param_tensor(.f32, &.{});
    const y = try b.broadcast_in_dim(x, &.{ 2, 3 }, &.{});
    const func_id = try build_func(&program, &b, &.{y});
    const func = program.get_function_by_id(func_id).?;

    const test_inputs = try make_test_inputs(std.testing.allocator, func);
    defer {
        for (test_inputs) |*t| t.deinit();
        std.testing.allocator.free(test_inputs);
    }

    try check_gradients(std.testing.allocator, &program, func_id, test_inputs, .{});
}

test "grad: broadcast_in_dim with size-1 dims" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "bcast_size1");
    defer b.deinit();
    const x = try b.param_tensor(.f32, &.{ 1, 3 });
    const y = try b.broadcast_in_dim(x, &.{ 4, 3 }, &.{ 0, 1 });
    const func_id = try build_func(&program, &b, &.{y});
    const func = program.get_function_by_id(func_id).?;

    const test_inputs = try make_test_inputs(std.testing.allocator, func);
    defer {
        for (test_inputs) |*t| t.deinit();
        std.testing.allocator.free(test_inputs);
    }

    try check_gradients(std.testing.allocator, &program, func_id, test_inputs, .{});
}

test "grad: reduce sum single axis" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "rsum1");
    defer b.deinit();
    const x = try b.param_tensor(.f32, &.{ 2, 3 });
    const y = try b.reduce(x, .{ .axes = &.{1}, .operation = .sum });
    const func_id = try build_func(&program, &b, &.{y});
    const func = program.get_function_by_id(func_id).?;

    const test_inputs = try make_test_inputs(std.testing.allocator, func);
    defer {
        for (test_inputs) |*t| t.deinit();
        std.testing.allocator.free(test_inputs);
    }

    try check_gradients(std.testing.allocator, &program, func_id, test_inputs, .{});
}

test "grad: reduce sum all axes" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "rsum_all");
    defer b.deinit();
    const x = try b.param_tensor(.f32, &.{ 2, 3 });
    const y = try b.reduce(x, .{ .axes = &.{ 0, 1 }, .operation = .sum });
    const func_id = try build_func(&program, &b, &.{y});
    const func = program.get_function_by_id(func_id).?;

    const test_inputs = try make_test_inputs(std.testing.allocator, func);
    defer {
        for (test_inputs) |*t| t.deinit();
        std.testing.allocator.free(test_inputs);
    }

    try check_gradients(std.testing.allocator, &program, func_id, test_inputs, .{});
}

test "grad: slice" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "slc");
    defer b.deinit();
    const x = try b.param_tensor(.f32, &.{ 4, 4 });
    const y = try b.slice(x, .{
        .start_indices = &.{ 1, 1 },
        .limit_indices = &.{ 3, 3 },
        .strides = &.{ 1, 1 },
    });
    const func_id = try build_func(&program, &b, &.{y});
    const func = program.get_function_by_id(func_id).?;

    const test_inputs = try make_test_inputs(std.testing.allocator, func);
    defer {
        for (test_inputs) |*t| t.deinit();
        std.testing.allocator.free(test_inputs);
    }

    try check_gradients(std.testing.allocator, &program, func_id, test_inputs, .{});
}

test "grad: concatenate" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "cat");
    defer b.deinit();
    const x = try b.param_tensor(.f32, &.{ 2, 2 });
    const y = try b.param_tensor(.f32, &.{ 2, 3 });
    const z = try b.concatenate(&.{ x, y }, 1);
    const func_id = try build_func(&program, &b, &.{z});
    const func = program.get_function_by_id(func_id).?;

    const test_inputs = try make_test_inputs(std.testing.allocator, func);
    defer {
        for (test_inputs) |*t| t.deinit();
        std.testing.allocator.free(test_inputs);
    }

    try check_gradients(std.testing.allocator, &program, func_id, test_inputs, .{});
}

test "grad: reduce maximum" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "rmax");
    defer b.deinit();
    const x = try b.param_tensor(.f32, &.{ 3, 4 });
    const y = try b.reduce(x, .{ .axes = &.{1}, .operation = .maximum });
    const func_id = try build_func(&program, &b, &.{y});
    const func = program.get_function_by_id(func_id).?;

    const test_inputs = try make_test_inputs(std.testing.allocator, func);
    defer {
        for (test_inputs) |*t| t.deinit();
        std.testing.allocator.free(test_inputs);
    }

    try check_gradients(std.testing.allocator, &program, func_id, test_inputs, .{});
}

// ============================================================================
// Priority 3 - Composite patterns
// ============================================================================

test "grad: softmax pattern" {
    // softmax = exp(x) / sum(exp(x))
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "softmax");
    defer b.deinit();
    const x = try b.param_tensor(.f32, &.{ 2, 3 });
    const ex = try b.exp(x);
    const sum_ex = try b.reduce(ex, .{ .axes = &.{1}, .operation = .sum }); // [2]
    const sum_bcast = try b.broadcast_in_dim(try b.reshape(sum_ex, &.{ 2, 1 }), &.{ 2, 3 }, &.{ 0, 1 });
    const softmax = try b.divide(ex, sum_bcast);
    const func_id = try build_func(&program, &b, &.{softmax});
    const func = program.get_function_by_id(func_id).?;

    const test_inputs = try make_test_inputs(std.testing.allocator, func);
    defer {
        for (test_inputs) |*t| t.deinit();
        std.testing.allocator.free(test_inputs);
    }

    try check_gradients(std.testing.allocator, &program, func_id, test_inputs, .{ .tolerance = 2e-2 });
}

test "grad: matmul + bias + logistic chain" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "mlp_layer");
    defer b.deinit();
    const x = try b.param_tensor(.f32, &.{ 2, 3 });
    const w = try b.param_tensor(.f32, &.{ 3, 4 });
    const bias = try b.param_tensor(.f32, &.{ 1, 4 });

    const matmul = try b.mm(x, w); // [2,4]
    const bias_bcast = try b.broadcast_in_dim(bias, &.{ 2, 4 }, &.{ 0, 1 });
    const with_bias = try b.add(matmul, bias_bcast);
    const activated = try b.logistic(with_bias);
    const func_id = try build_func(&program, &b, &.{activated});
    const func = program.get_function_by_id(func_id).?;

    const test_inputs = try make_test_inputs(std.testing.allocator, func);
    defer {
        for (test_inputs) |*t| t.deinit();
        std.testing.allocator.free(test_inputs);
    }

    try check_gradients(std.testing.allocator, &program, func_id, test_inputs, .{});
}

test "grad: cross-entropy loss" {
    // loss = -sum(labels * log(probs))
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "ce_loss");
    defer b.deinit();
    const probs = try b.param_tensor(.f32, &.{ 2, 3 }); // predicted probabilities
    const labels = try b.param_tensor(.f32, &.{ 2, 3 }); // one-hot labels

    const log_probs = try b.log(probs);
    const weighted = try b.multiply(labels, log_probs);
    const sum_per_sample = try b.reduce(weighted, .{ .axes = &.{1}, .operation = .sum }); // [2]
    const neg_one = try b.literal_scalar(.{ .f32 = -1.0 });
    const neg_bcast = try b.broadcast_in_dim(neg_one, &.{2}, &.{});
    const neg_sum = try b.multiply(neg_bcast, sum_per_sample);
    const func_id = try build_func(&program, &b, &.{neg_sum});
    const func = program.get_function_by_id(func_id).?;

    const test_inputs = try make_test_inputs(std.testing.allocator, func);
    defer {
        for (test_inputs) |*t| t.deinit();
        std.testing.allocator.free(test_inputs);
    }

    try check_gradients(std.testing.allocator, &program, func_id, test_inputs, .{});
}
