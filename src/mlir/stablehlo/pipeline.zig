//! Default validated PR to StableHLO pipeline segment.

const compilation = @import("../../compilation.zig");
const zg = @import("../../zigrad.zig");
const outline = @import("../../outline.zig");
const stablehlo = @import("../../stablehlo.zig");
const lower = @import("lower.zig");

/// Operations and lowering policy for the default StableHLO segment.
pub const Options = struct {
    /// StableHLO serialization passed to the next operation.
    encoding: stablehlo.Encoding = .binary,
};

/// Append the default validated PR to StableHLO segment.
pub fn add(
    pipeline: *compilation.Pipeline,
    options: Options,
) compilation.Pipeline.AddError!void {
    try pipeline.add(zg.Validate{});
    try pipeline.add(outline.Pass{});
    try pipeline.add(lower.Lower{
        .config = .{
            .encoding = options.encoding,
        },
    });
}
