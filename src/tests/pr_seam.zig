//! Seam check: a root that imports both `zigrad` and `pr` sees one program
//!  representation, because every `pr` file belongs to the `pr` module alone.

const std = @import("std");
const zg = @import("zigrad");
const pr = @import("pr");

comptime {
    std.debug.assert(zg.pr == pr);
    for (.{
        "Program", "Function",   "Var",        "Shape",           "BoundedShape", "Aval",
        "DType",   "FunctionId", "Annotation", "AnnotationValue",
    }) |name| std.debug.assert(@field(zg.pr, name) == @field(pr, name));
    std.debug.assert(zg.DType == pr.DType);
}

test "pr_seam program crosses zigrad pipeline through zg.Validate" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var builder = try pr.FunctionBuilder.init(&program, "main");
    defer builder.deinit();
    const input = try builder.param_tensor(.f32, &.{2});
    const sum = try builder.add(input, input);
    const function = try builder.finish(.{ .returns = &.{sum} });
    const function_id = try program.add_function(function);
    try program.set_entry(function_id);
    try std.testing.expectEqual(@as(usize, 1), program.functions().len);
    try std.testing.expectEqual(@as(usize, 1), program.functions()[0].ops.len);

    var pipeline = zg.Pipeline.init(std.testing.allocator);
    defer pipeline.deinit();
    try pipeline.add(zg.Validate{});

    var ctx: zg.CompilationCtx = .{
        .allocator = std.testing.allocator,
        .io = std.testing.io,
    };
    const out = try pipeline.run(*pr.Program, &program, &ctx);
    try std.testing.expect(out == &program);
}
