//! PR program representation and operations.

const core = @import("pr/pr.zig");
const compilation = @import("compilation");

pub const DType = core.DType;
pub const Shape = core.Shape;
pub const BoundedShape = core.BoundedShape;
pub const Aval = core.Aval;
pub const Tensor = core.Tensor;
pub const max_rank = core.max_rank;

pub const Literal = core.Literal;
pub const Prim = core.Prim;

// Param structs
pub const GatherParams = core.GatherParams;
pub const ScatterParams = core.ScatterParams;
pub const CompareParams = core.CompareParams;
pub const CompareDirection = core.CompareDirection;
pub const CompareType = core.CompareType;
pub const ScatterReduction = core.ScatterReduction;
pub const SliceParams = core.SliceParams;
pub const DotGeneralParams = core.DotGeneralParams;
pub const ConvolutionParams = core.ConvolutionParams;
pub const ConvolutionDimensionNumbers = core.ConvolutionDimensionNumbers;
pub const ReshapeParams = core.ReshapeParams;
pub const IotaParams = core.IotaParams;
pub const BroadcastInDimParams = core.BroadcastInDimParams;
pub const TransposeParams = core.TransposeParams;
pub const ConcatenateParams = core.ConcatenateParams;
pub const ReduceParams = core.ReduceParams;
pub const Reduction = core.Reduction;
pub const CallParams = core.CallParams;
pub const FunctionId = core.FunctionId;
pub const FunctionRegistrationError = core.FunctionRegistrationError;
pub const CustomCallParams = core.CustomCallParams;

// Core IR types
pub const Var = core.Var;
pub const Operand = core.Operand;
pub const Op = core.Op;
pub const Params = core.Params;

pub const AnnotationValue = core.AnnotationValue;
pub const Annotation = core.Annotation;
pub const dupe_annotations = core.dupe_annotations;
pub const Region = core.Region;
pub const Function = core.Function;
pub const RegionIterator = core.RegionIterator;
pub const Program = core.Program;

pub const ValidationError = core.ValidationError;
pub const validate_function = core.validate_function;
pub const validate_program = core.validate_program;

/// Validates a borrowed PR program and returns it unchanged.
pub const Validate = struct {
    pub const Input = *Program;
    pub const Output = *Program;

    pub fn run(_: Validate, program: Input, _: *compilation.Context) !Output {
        try validate_program(program);
        return program;
    }
};

pub const BuildError = core.BuildError;
pub const FunctionBuilder = core.FunctionBuilder;

pub const ad = @import("pr/ad.zig");
pub const json = @import("pr/json.zig");
pub const serialize = @import("pr/serialize.zig");
pub const tool = @import("pr/tool.zig");
pub const zxpr = @import("pr/zxpr.zig");
pub const dump = @import("pr/dump.zig");
pub const ops = @import("pr/ops/ops.zig");
pub const analysis = @import("pr/analysis.zig");

pub const transform = struct {
    pub const annotate = @import("pr/transform/annotate.zig");
    pub const kernelize = @import("pr/transform/kernelize.zig");
    pub const outline = @import("pr/transform/outline.zig");
};

test {
    @import("std").testing.refAllDecls(@This());
    _ = transform.kernelize;
}
