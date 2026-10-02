const std = @import("std");

const pr = @import("pr");
const ad = @import("pr").ad;
const pr_eval = @import("pr_eval");
const Tensor = @import("../tensor.zig");
const trace = @import("../trace.zig").trace;
const transforms = @import("../transforms.zig");

fn square(x: Tensor) !Tensor {
    return try x.mul(x);
}

fn cube(x: Tensor) !Tensor {
    return try (try x.mul(x)).mul(x);
}

fn scaled_square(x: Tensor, scale: Tensor) !Tensor {
    return try (try x.mul(x)).mul(scale);
}

fn expect_scalar_result(
    program: *const pr.Program,
    function_id: pr.FunctionId,
    input_values: []const f32,
    expected: f32,
) !void {
    const allocator = std.testing.allocator;
    const inputs = try allocator.alloc(pr_eval.HostTensor, input_values.len);
    var initialized_inputs: usize = 0;
    defer {
        for (inputs[0..initialized_inputs]) |*input| input.deinit();
        allocator.free(inputs);
    }
    for (input_values, inputs) |value, *input| {
        input.* = try pr_eval.HostTensor.init_with_data(allocator, &.{}, &.{value});
        initialized_inputs += 1;
    }

    const function = program.get_function_by_id(function_id) orelse
        return error.TestUnexpectedResult;
    const results = try pr_eval.eval(allocator, program, function, inputs);
    defer {
        for (results) |*result| result.deinit();
        allocator.free(results);
    }
    try std.testing.expectEqual(@as(usize, 1), results.len);
    try std.testing.expectApproxEqAbs(expected, results[0].data[0], 1e-5);
}

test "second derivative executes through generated calls" {
    const first = comptime transforms.make_grad(square, .{});
    const second = comptime transforms.make_grad(first, .{});
    const specs = .{Tensor.abstract(.f32, &.{})};

    var traced = try trace(second, std.testing.allocator, specs, .{ .name = "second_derivative" });
    defer traced.deinit();
    try pr.validate_program(&traced.program);
    try expect_scalar_result(&traced.program, try traced.program.resolve_entry(), &.{3.0}, 2.0);
}

test "generated gradients compose recursively" {
    const first = comptime transforms.make_grad(cube, .{});
    const second = comptime transforms.make_grad(first, .{});
    const third = comptime transforms.make_grad(second, .{});
    const specs = .{Tensor.abstract(.f32, &.{})};

    var traced = try trace(third, std.testing.allocator, specs, .{ .name = "third_derivative" });
    defer traced.deinit();
    try pr.validate_program(&traced.program);
    try expect_scalar_result(&traced.program, try traced.program.resolve_entry(), &.{3.0}, 6.0);
}

test "mixed partial selects a different argument at each order" {
    const dx = comptime transforms.make_grad(scaled_square, .{ .wrt_argnums = &.{0} });
    const dx_dscale = comptime transforms.make_grad(dx, .{ .wrt_argnums = &.{1} });
    const specs = .{
        Tensor.abstract(.f32, &.{}),
        Tensor.abstract(.f32, &.{}),
    };

    var traced = try trace(dx_dscale, std.testing.allocator, specs, .{ .name = "mixed_partial" });
    defer traced.deinit();
    try pr.validate_program(&traced.program);
    try expect_scalar_result(&traced.program, try traced.program.resolve_entry(), &.{ 3.0, 4.0 }, 6.0);
}

test "JVP composes over a generated JVP program" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var builder = try pr.FunctionBuilder.init(&program, "cube");
    defer builder.deinit();
    const x = try builder.param_tensor(.f32, &.{});
    const x_squared = try builder.multiply(x, x);
    const x_cubed = try builder.multiply(x_squared, x);
    const source = try program.add_function(try builder.finish(.{ .returns = &.{x_cubed} }));

    const first = try ad.jvp(std.testing.allocator, &program, source, "cube_jvp", .{});
    const second = try ad.jvp(std.testing.allocator, &program, first, "cube_jvp_jvp", .{});
    try pr.validate_program(&program);

    // d(3 x^2 dx)[delta_x, delta_dx]
    //   = 6 x dx delta_x + 3 x^2 delta_dx
    try expect_scalar_result(&program, second, &.{ 2.0, 3.0, 5.0, 7.0 }, 264.0);
}

test "JVP of VJP computes a Hessian-vector product" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var builder = try pr.FunctionBuilder.init(&program, "sum_of_cubes");
    defer builder.deinit();
    const x = try builder.param_tensor(.f32, &.{2});
    const x_squared = try builder.multiply(x, x);
    const x_cubed = try builder.multiply(x_squared, x);
    const loss = try builder.reduce(x_cubed, .{ .axes = &.{0}, .operation = .sum });
    const source = try program.add_function(try builder.finish(.{ .returns = &.{loss} }));

    const gradient = try ad.vjp(std.testing.allocator, &program, source, "sum_of_cubes_vjp", .{});
    const hvp = try ad.jvp(std.testing.allocator, &program, gradient, "sum_of_cubes_hvp", .{});
    try pr.validate_program(&program);

    var primal = try pr_eval.HostTensor.init_with_data(std.testing.allocator, &.{2}, &.{ 2.0, 3.0 });
    defer primal.deinit();
    var cotangent = try pr_eval.HostTensor.init_with_data(std.testing.allocator, &.{}, &.{1.0});
    defer cotangent.deinit();
    var direction = try pr_eval.HostTensor.init_with_data(std.testing.allocator, &.{2}, &.{ 5.0, 7.0 });
    defer direction.deinit();
    var zero_seed = try pr_eval.HostTensor.init_with_data(std.testing.allocator, &.{}, &.{0.0});
    defer zero_seed.deinit();

    const function = program.get_function_by_id(hvp).?;
    const results = try pr_eval.eval(
        std.testing.allocator,
        &program,
        function,
        &.{ primal, cotangent, direction, zero_seed },
    );
    defer {
        for (results) |*result| result.deinit();
        std.testing.allocator.free(results);
    }
    try std.testing.expectEqual(@as(usize, 1), results.len);
    try std.testing.expectEqualSlices(f32, &.{ 60.0, 126.0 }, results[0].data);
}
