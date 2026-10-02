//! Seam check: a root that imports both `zigrad` and `pr` sees one program
//! representation, because every `pr` file belongs to the `pr` module alone.

const std = @import("std");
const zg = @import("zigrad");
const pr = @import("pr");

comptime {
    std.debug.assert(zg.pr == pr);
    std.debug.assert(zg.pr.Program == pr.Program);
}

test "pr_seam program crosses zigrad pipeline" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var pipeline = zg.Pipeline.init(std.testing.allocator);
    defer pipeline.deinit();
    try pipeline.add(pr.Validate{});

    var ctx: zg.CompilationCtx = .{
        .allocator = std.testing.allocator,
        .io = std.testing.io,
    };
    const out = try pipeline.run(*pr.Program, &program, &ctx);
    try std.testing.expect(out == &program);
}
