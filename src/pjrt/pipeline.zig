//! Default PR to PJRT pipeline composition.

const std = @import("std");
const compilation = @import("../compilation.zig");
const stablehlo_pipeline = @import("../mlir/stablehlo/pipeline.zig");
const Backend = @import("backend.zig").Backend;

/// Operations and lowering policy for the default PJRT pipeline.
pub const Options = struct {
    /// Validated PR to StableHLO segment.
    stablehlo: stablehlo_pipeline.Options = .{},
};

/// Create the default validated PR to loaded PJRT program pipeline.
///
/// The returned queue owns copies of its operations. `backend_instance` and any
///  state referenced by optional operations must outlive the queue.
pub fn create(
    allocator: std.mem.Allocator,
    backend_instance: *Backend,
    options: Options,
) compilation.Pipeline.AddError!compilation.Pipeline {
    var result = compilation.Pipeline.init(allocator);
    errdefer result.deinit();

    try stablehlo_pipeline.add(&result, options.stablehlo);
    try result.add(&backend_instance.interface);
    return result;
}
