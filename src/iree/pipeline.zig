//! Default PR to IREE pipeline composition.

const std = @import("std");
const compilation = @import("../compilation.zig");
const stablehlo_pipeline = @import("../mlir/stablehlo/pipeline.zig");
const Backend = @import("backend.zig").Backend;
const Compiler = @import("compiler.zig").Compiler;
const Artifact = @import("compiler.zig").Artifact;

/// Terminal operation selected for the default IREE pipeline.
pub const Terminal = union(enum) {
    /// Compile and load the program for execution.
    loaded: *Backend,

    /// Compile the program into IREE VM bytecode.
    vmfb: *Compiler,
};

/// Operations and lowering policy for the default IREE pipeline.
pub const Options = struct {
    /// Validated PR to StableHLO segment.
    stablehlo: stablehlo_pipeline.Options = .{},
};

/// Create a default PR to IREE pipeline for the selected terminal operation.
pub fn create(
    allocator: std.mem.Allocator,
    terminal: Terminal,
    options: Options,
) compilation.Pipeline.AddError!compilation.Pipeline {
    var result = compilation.Pipeline.init(allocator);
    errdefer result.deinit();

    try stablehlo_pipeline.add(&result, options.stablehlo);
    switch (terminal) {
        .loaded => |backend| try result.add(&backend.interface),
        .vmfb => |compiler| try result.add(&compiler.interface),
    }
    return result;
}
