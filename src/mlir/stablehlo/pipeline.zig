//! Default validated PR to StableHLO pipeline segment.

const compilation = @import("compilation");
const pr = @import("pr");
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
    try pipeline.add(pr.Validate{});
    try pipeline.add(pr.transform.outline.Pass{});
    try pipeline.add(lower.Lower{
        .config = .{
            .encoding = options.encoding,
        },
    });
}
