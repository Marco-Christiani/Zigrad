//! Observable side-effect analysis for PR call graphs.

const pr = @import("../pr.zig");

/// Return whether a function or one of its callees may have observable side effects.
///
/// Unknown callees and recursive call cycles are treated as effectful. PR primitives
///  other than `custom_call` and `call` are pure.
/// TODO: under-developed.
pub fn function_may_have_side_effects(program: *const pr.Program, func: pr.Function) bool {
    return function_may_have_side_effects_impl(program, func, 0);
}

fn function_may_have_side_effects_impl(
    program: *const pr.Program,
    func: pr.Function,
    call_depth: usize,
) bool {
    for (func.ops) |op| switch (op.params) {
        .custom_call => |params| {
            if (params.has_side_effect) return true;
        },
        .call => |params| {
            if (call_depth >= program.functions().len) return true;
            const callee = program.get_function_by_id(params.callee) orelse return true;
            if (function_may_have_side_effects_impl(program, callee, call_depth + 1)) {
                return true;
            }
        },
        else => {},
    };
    return false;
}

test function_may_have_side_effects {
    const testing = @import("std").testing;
    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var pure_builder = try pr.FunctionBuilder.init(&program, "pure");
    defer pure_builder.deinit();
    const pure_input = try pure_builder.param_tensor(.f32, &.{2});
    _ = try program.add_function(try pure_builder.finish(.{ .returns = &.{pure_input} }));

    var effectful_builder = try pr.FunctionBuilder.init(&program, "effectful");
    defer effectful_builder.deinit();
    const effectful_input = try effectful_builder.param_tensor(.f32, &.{2});
    const effectful_outputs = (try effectful_builder.custom_call(.{
        .target_name = "test.effectful",
        .has_side_effect = true,
    }, &.{effectful_input}, &.{effectful_input.aval})).outputs;
    const effectful_id = try program.add_function(try effectful_builder.finish(.{ .returns = effectful_outputs }));

    var caller_builder = try pr.FunctionBuilder.init(&program, "caller");
    defer caller_builder.deinit();
    const caller_input = try caller_builder.param_tensor(.f32, &.{2});
    const caller_outputs = (try caller_builder.call(effectful_id, &.{caller_input})).outputs;
    const caller_id = try program.add_function(try caller_builder.finish(.{ .returns = caller_outputs }));

    try testing.expect(!function_may_have_side_effects(&program, program.functions()[0]));
    try testing.expect(function_may_have_side_effects(&program, program.functions()[1]));
    try testing.expect(function_may_have_side_effects(&program, program.functions()[2]));

    program.functions()[2].ops[0].params.call.callee = caller_id;
    try testing.expect(function_may_have_side_effects(&program, program.functions()[2]));

    program.functions()[2].ops[0].params.call.callee = @enumFromInt(99);
    try testing.expect(function_may_have_side_effects(&program, program.functions()[2]));
}
