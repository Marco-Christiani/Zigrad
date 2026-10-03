//! Public surface for the optional TVM integration.
//!
//! Translated declarations, DLPack ABI values, packed-call values, and TVM
//!  object wrappers remain implementation details.

const config = @import("tvm/config.zig");
const matmul = @import("tvm/matmul.zig");

pub const runtime = @import("tvm/runtime.zig");
pub const TargetKind = config.TargetKind;
pub const BuildIdentity = config.BuildIdentity;
pub const CompileConfig = config.CompileConfig;
pub const TimeEvaluatorOptions = @import("c/tvm/runtime.zig").TimeEvaluatorOptions;

pub const ElementType = matmul.ElementType;
pub const MatmulShape = matmul.Shape;
pub const TuneOptions = matmul.TuneOptions;
pub const TuneResult = matmul.TuneResult;
pub const tune_matmul = matmul.tune;
pub const CachedMatmul = matmul.CachedMatmul;

pub const Provider = @import("tvm/provider.zig").TvmProvider;
pub const DispatchState = @import("tvm/dispatch.zig").TvmDispatchState;

test {
    @import("std").testing.refAllDecls(@This());
}

test "public TVM surface excludes raw integration namespaces" {
    const testing = @import("std").testing;
    try testing.expect(!@hasDecl(@This(), "ffi"));
    try testing.expect(!@hasDecl(@This(), "dlpack"));
    try testing.expect(!@hasDecl(@This(), "tir"));
    try testing.expect(!@hasDecl(@This(), "meta_schedule"));
    try testing.expect(!@hasDecl(@This(), "compile"));
    try testing.expect(!@hasDecl(@This(), "artifact"));

    switch (@typeInfo(CachedMatmul)) {
        .@"opaque" => {},
        else => return error.CachedMatmulMustRemainOpaque,
    }
}
