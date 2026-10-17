#include "zigrad/mirage/MirageKernelSelectPass.h"

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "zigrad/ZigradDialect.h"
#include "zigrad/ZigradKernelUtils.h"

namespace mlir::zigrad::mirage {
namespace {

constexpr StringLiteral kProvider("mirage");

/// Per-pass kernel key counter, reset at each pass invocation. Patterns hold a
/// pointer to the counter owned by MirageKernelSelectPass::runOnOperation so
/// keys start from mk_0 for every compilation unit.
struct KeyCounter {
  unsigned value = 0;
  std::string next() { return "mk_" + std::to_string(value++); }
};

// mul(add(dot, x), x) -> dot_add_mul

struct DotAddMulPattern final : RewritePattern {
  KeyCounter *counter;

  explicit DotAddMulPattern(MLIRContext *ctx, KeyCounter *counter)
      : RewritePattern("stablehlo.multiply", 3, ctx), counter(counter) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 2) return failure();

    const Value mul_lhs = op->getOperand(0);
    const Value mul_rhs = op->getOperand(1);
    Operation *mul_lhs_def = mul_lhs.getDefiningOp();
    Operation *mul_rhs_def = mul_rhs.getDefiningOp();

    Operation *add_op = nullptr;
    Value mul_passthrough;
    if (kernel_utils::is_named_op(mul_lhs_def, "stablehlo.add")) {
      add_op = mul_lhs_def;
      mul_passthrough = mul_rhs;
    } else if (kernel_utils::is_named_op(mul_rhs_def, "stablehlo.add")) {
      add_op = mul_rhs_def;
      mul_passthrough = mul_lhs;
    } else {
      return failure();
    }

    if (add_op->getNumOperands() != 2) return failure();

    const Value add_lhs = add_op->getOperand(0);
    const Value add_rhs = add_op->getOperand(1);
    Operation *add_lhs_def = add_lhs.getDefiningOp();
    Operation *add_rhs_def = add_rhs.getDefiningOp();

    Operation *dot_op = nullptr;
    if (kernel_utils::is_dot_op(add_lhs_def) && add_rhs == mul_passthrough) {
      dot_op = add_lhs_def;
    } else if (kernel_utils::is_dot_op(add_rhs_def) && add_lhs == mul_passthrough) {
      dot_op = add_rhs_def;
    } else {
      return failure();
    }

    if (dot_op->getNumResults() != 1 || !dot_op->getResult(0).hasOneUse())
      return failure();
    if (dot_op->getNumOperands() != 2) return failure();
    if (!kernel_utils::has_matmul_dims(dot_op,
                                       kernel_utils::RhsLayout::standard))
      return failure();

    SmallVector<Value, 3> call_operands = {
        dot_op->getOperand(0),
        dot_op->getOperand(1),
        mul_passthrough,
    };

    std::string key = counter->next();
    Operation *replacement = kernel_utils::create_kernel_call(
        op, call_operands, op->getResultTypes(),
        kProvider, key, "dot_add_mul", rewriter);
    rewriter.replaceOp(op, replacement->getResults());

    if (add_op->use_empty()) rewriter.eraseOp(add_op);
    if (dot_op->use_empty()) rewriter.eraseOp(dot_op);

    return success();
  }
};

// add(dot, x) -> dot_add

struct DotAddPattern final : RewritePattern {
  KeyCounter *counter;

  explicit DotAddPattern(MLIRContext *ctx, KeyCounter *counter)
      : RewritePattern("stablehlo.add", 2, ctx), counter(counter) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 2) return failure();

    const Value lhs = op->getOperand(0);
    const Value rhs = op->getOperand(1);
    Operation *lhs_def = lhs.getDefiningOp();
    Operation *rhs_def = rhs.getDefiningOp();

    Operation *dot_op = nullptr;
    Value passthrough;
    if (kernel_utils::is_dot_op(lhs_def)) {
      dot_op = lhs_def;
      passthrough = rhs;
    } else if (kernel_utils::is_dot_op(rhs_def)) {
      dot_op = rhs_def;
      passthrough = lhs;
    } else {
      return failure();
    }

    if (dot_op->getNumResults() != 1 || !dot_op->getResult(0).hasOneUse())
      return failure();
    if (dot_op->getNumOperands() != 2) return failure();
    if (!kernel_utils::has_matmul_dims(dot_op,
                                       kernel_utils::RhsLayout::standard))
      return failure();

    SmallVector<Value, 3> call_operands = {
        dot_op->getOperand(0),
        dot_op->getOperand(1),
        passthrough,
    };

    std::string key = counter->next();
    Operation *replacement = kernel_utils::create_kernel_call(
        op, call_operands, op->getResultTypes(),
        kProvider, key, "dot_add", rewriter);
    rewriter.replaceOp(op, replacement->getResults());

    if (dot_op->use_empty()) rewriter.eraseOp(dot_op);

    return success();
  }
};

// exp(dot) -> dot_exp and log(dot) -> dot_log

struct DotUnaryPattern final : RewritePattern {
  KeyCounter *counter;

  explicit DotUnaryPattern(MLIRContext *ctx, StringRef op_name,
                           KeyCounter *counter)
      : RewritePattern(op_name, 2, ctx), counter(counter) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1) return failure();

    Operation *dot_op = op->getOperand(0).getDefiningOp();
    if (!kernel_utils::is_dot_op(dot_op)) return failure();

    if (dot_op->getNumResults() != 1 || !dot_op->getResult(0).hasOneUse())
      return failure();
    if (dot_op->getNumOperands() != 2) return failure();
    if (!kernel_utils::has_matmul_dims(dot_op,
                                       kernel_utils::RhsLayout::standard))
      return failure();

    SmallVector<Value, 2> call_operands = {
        dot_op->getOperand(0),
        dot_op->getOperand(1),
    };

    const StringRef unary_name = op->getName().getStringRef();
    const StringRef pattern =
        unary_name == "stablehlo.log" ? "dot_log" : "dot_exp";

    std::string key = counter->next();
    Operation *replacement = kernel_utils::create_kernel_call(
        op, call_operands, op->getResultTypes(),
        kProvider, key, pattern, rewriter);
    rewriter.replaceOp(op, replacement->getResults());

    if (dot_op->use_empty()) rewriter.eraseOp(dot_op);

    return success();
  }
};

// rms_norm(X) @ W -> rms_norm_matmul
//
// Anchored on stablehlo.dot_general, walks backward through:
//   dot_general(lhs, W)
//     lhs = [reshape] -> [convert f32->bf16] -> multiply(normed, gamma_bc) -> ...
//     normed = multiply(x, broadcast(rsqrt(add(multiply(reduce_sum(multiply(x,x)), scale), eps))))
//
// Pre-multiplies gamma into W (rms_norm(X)*gamma)@W = rms_norm(X)@(gamma*W)) to avoid
// passing a rank-1 gamma through Mirage (which can't broadcast rank-1 vs rank-2).

struct RmsNormMatmulPattern final : RewritePattern {
  KeyCounter *counter;

  explicit RmsNormMatmulPattern(MLIRContext *ctx, KeyCounter *counter)
      : RewritePattern("stablehlo.dot_general", 5, ctx), counter(counter) {}

  LogicalResult matchAndRewrite(Operation *dot_op,
                                PatternRewriter &rewriter) const override {
    if (dot_op->getNumOperands() != 2 || dot_op->getNumResults() != 1)
      return failure();
    if (!kernel_utils::has_matmul_dims(dot_op,
                                       kernel_utils::RhsLayout::standard))
      return failure();

    Value dot_lhs = dot_op->getOperand(0);
    Value w = dot_op->getOperand(1);

    // Walk backward through optional reshape and convert.
    Value pre_gamma = dot_lhs;
    Operation *reshape_op = nullptr;
    Operation *convert_to_bf16_op = nullptr;

    // Optional: reshape (e.g. [1,4,2048] -> [4,2048])
    {
      Operation *def = pre_gamma.getDefiningOp();
      if (kernel_utils::is_named_op(def, "stablehlo.reshape")) {
        reshape_op = def;
        if (reshape_op->getNumOperands() != 1) return failure();
        pre_gamma = reshape_op->getOperand(0);
      }
    }

    // Optional: convert f32 -> bf16 (precision cast after rms_norm)
    {
      Operation *def = pre_gamma.getDefiningOp();
      if (kernel_utils::is_named_op(def, "stablehlo.convert")) {
        convert_to_bf16_op = def;
        if (convert_to_bf16_op->getNumOperands() != 1) return failure();
        pre_gamma = convert_to_bf16_op->getOperand(0);
      }
    }

    // gamma_mul = multiply(normed, gamma_bc)
    Operation *gamma_mul_op = pre_gamma.getDefiningOp();
    if (!kernel_utils::is_named_op(gamma_mul_op, "stablehlo.multiply"))
      return failure();
    if (gamma_mul_op->getNumOperands() != 2) return failure();

    // Identify which operand is the gamma broadcast and which is normed.
    Operation *gamma_bc_op = nullptr;
    Value normed_val;
    for (int i = 0; i < 2; ++i) {
      Operation *def = gamma_mul_op->getOperand(i).getDefiningOp();
      if (kernel_utils::is_named_op(def, "stablehlo.broadcast_in_dim")) {
        gamma_bc_op = def;
        normed_val = gamma_mul_op->getOperand(1 - i);
        break;
      }
    }
    if (!gamma_bc_op) return failure();
    if (gamma_bc_op->getNumOperands() != 1) return failure();

    // gamma_val: the 1D gamma tensor (possibly through a convert bf16->f32).
    Value gamma_val = gamma_bc_op->getOperand(0);
    Operation *gamma_convert_op = nullptr;
    {
      Operation *def = gamma_val.getDefiningOp();
      if (def && kernel_utils::is_named_op(def, "stablehlo.convert")) {
        gamma_convert_op = def;
        if (gamma_convert_op->getNumOperands() != 1) return failure();
        gamma_val = gamma_convert_op->getOperand(0);
      }
    }
    // gamma_val is the original gamma (bf16 or f32).
    Value gamma_original = gamma_val;

    // normed = multiply(x, inv_bc)
    Operation *normed_mul_op = normed_val.getDefiningOp();
    if (!kernel_utils::is_named_op(normed_mul_op, "stablehlo.multiply"))
      return failure();
    if (normed_mul_op->getNumOperands() != 2) return failure();

    // Find inv_bc (broadcast_in_dim) and x among normed_mul operands.
    Operation *inv_bc_op = nullptr;
    Value x;
    for (int i = 0; i < 2; ++i) {
      Operation *def = normed_mul_op->getOperand(i).getDefiningOp();
      if (kernel_utils::is_named_op(def, "stablehlo.broadcast_in_dim")) {
        inv_bc_op = def;
        x = normed_mul_op->getOperand(1 - i);
        break;
      }
    }
    if (!inv_bc_op) return failure();
    if (inv_bc_op->getNumOperands() != 1) return failure();

    // inv = rsqrt(denom)
    Operation *rsqrt_op = inv_bc_op->getOperand(0).getDefiningOp();
    if (!kernel_utils::is_named_op(rsqrt_op, "stablehlo.rsqrt"))
      return failure();
    if (rsqrt_op->getNumOperands() != 1) return failure();

    // denom = add(mean, eps)
    Operation *denom_add_op = rsqrt_op->getOperand(0).getDefiningOp();
    if (!kernel_utils::is_named_op(denom_add_op, "stablehlo.add"))
      return failure();
    if (denom_add_op->getNumOperands() != 2) return failure();

    // mean = multiply(reduce_sum, scale)
    Operation *mean_mul_op = nullptr;
    for (int i = 0; i < 2; ++i) {
      Operation *def = denom_add_op->getOperand(i).getDefiningOp();
      if (kernel_utils::is_named_op(def, "stablehlo.multiply")) {
        mean_mul_op = def;
        break;
      }
    }
    if (!mean_mul_op || mean_mul_op->getNumOperands() != 2) return failure();

    // reduce_sum(sq)
    Operation *reduce_op = nullptr;
    for (int i = 0; i < 2; ++i) {
      Operation *def = mean_mul_op->getOperand(i).getDefiningOp();
      if (kernel_utils::is_named_op(def, "stablehlo.reduce")) {
        reduce_op = def;
        break;
      }
    }
    if (!reduce_op) return failure();

    // Verify reduce body is add.
    if (reduce_op->getNumRegions() != 1) return failure();
    Region &body = reduce_op->getRegion(0);
    if (body.empty() || !body.hasOneBlock()) return failure();
    {
      Operation *combiner = nullptr;
      for (Operation &inner : body.front().without_terminator())
        combiner = &inner;
      if (!combiner || !kernel_utils::is_named_op(combiner, "stablehlo.add"))
        return failure();
    }

    // sq = multiply(x, x)
    if (reduce_op->getNumOperands() < 1) return failure();
    Operation *sq_op = reduce_op->getOperand(0).getDefiningOp();
    if (!kernel_utils::is_named_op(sq_op, "stablehlo.multiply"))
      return failure();
    if (sq_op->getNumOperands() != 2) return failure();
    if (sq_op->getOperand(0) != sq_op->getOperand(1))
      return failure();

    // Verify x is the same Value used in sq_op.
    if (sq_op->getOperand(0) != x) return failure();

    // Require one use for each result checked below
    if (!denom_add_op->getResult(0).hasOneUse()) return failure();
    if (!mean_mul_op->getResult(0).hasOneUse()) return failure();
    if (!reduce_op->getResult(0).hasOneUse()) return failure();
    if (!sq_op->getResult(0).hasOneUse()) return failure();
    if (!rsqrt_op->getResult(0).hasOneUse()) return failure();
    if (!inv_bc_op->getResult(0).hasOneUse()) return failure();
    if (!normed_mul_op->getResult(0).hasOneUse()) return failure();
    if (!gamma_bc_op->getResult(0).hasOneUse()) return failure();
    // NOTE: gamma_mul_op, convert_to_bf16_op, and reshape_op may have >1 use
    //       when gate and up matmuls share the same rms_norm chain
    // The first match replaces one dot_general, the second replaces the other,
    // then the shared chain becomes dead and is erased.

    // Look through bf16->f32 convert on x to find the original bf16 input.
    Value x_original = x;
    Operation *x_convert_op = nullptr;
    {
      Operation *def = x.getDefiningOp();
      if (def && kernel_utils::is_named_op(def, "stablehlo.convert")) {
        x_convert_op = def;
        x_original = def->getOperand(0);
      }
    }

    // Extract normalized_size from the last dim of x.
    auto x_type = dyn_cast<RankedTensorType>(x.getType());
    if (!x_type || x_type.getRank() == 0) return failure();
    int64_t normalized_size = x_type.getDimSize(x_type.getRank() - 1);
    if (normalized_size <= 0) return failure();

    // === Emit pre-multiply: W' = broadcast(gamma) * W ===
    rewriter.setInsertionPoint(dot_op);

    // Determine X_2D: if x_original is rank > 2, reshape to 2D.
    Value x_2d = x_original;
    auto x_orig_type = dyn_cast<RankedTensorType>(x_original.getType());
    if (!x_orig_type) return failure();
    if (x_orig_type.getRank() > 2) {
      // Collapse leading dims: [d0, d1, ..., dn-1, last] -> [d0*d1*...*dn-1, last]
      int64_t leading = 1;
      for (int64_t i = 0; i < x_orig_type.getRank() - 1; ++i)
        leading *= x_orig_type.getDimSize(i);
      int64_t last = x_orig_type.getDimSize(x_orig_type.getRank() - 1);
      auto reshaped_type = RankedTensorType::get({leading, last}, x_orig_type.getElementType());
      OperationState rs(dot_op->getLoc(), "stablehlo.reshape");
      rs.addOperands(x_original);
      rs.addTypes(reshaped_type);
      Operation *rs_op = rewriter.create(rs);
      x_2d = rs_op->getResult(0);
    }

    // Match gamma dtype to W dtype for pre-multiply.
    auto w_type = dyn_cast<RankedTensorType>(w.getType());
    if (!w_type || w_type.getRank() != 2) return failure();
    auto gamma_orig_type = dyn_cast<RankedTensorType>(gamma_original.getType());
    if (!gamma_orig_type) return failure();

    Value gamma_matched = gamma_original;
    if (gamma_orig_type.getElementType() != w_type.getElementType()) {
      auto cvt_type = RankedTensorType::get(gamma_orig_type.getShape(), w_type.getElementType());
      OperationState cvt(dot_op->getLoc(), "stablehlo.convert");
      cvt.addOperands(gamma_original);
      cvt.addTypes(cvt_type);
      Operation *cvt_op = rewriter.create(cvt);
      gamma_matched = cvt_op->getResult(0);
    }

    // broadcast gamma to W shape along dim 0.
    OperationState gbc(dot_op->getLoc(), "stablehlo.broadcast_in_dim");
    gbc.addOperands(gamma_matched);
    gbc.addTypes(w_type);
    gbc.addAttribute("broadcast_dimensions", rewriter.getDenseI64ArrayAttr({0}));
    Operation *gamma_bc_w = rewriter.create(gbc);

    // W' = multiply(gamma_bc_w, W)
    OperationState wmul(dot_op->getLoc(), "stablehlo.multiply");
    wmul.addOperands({gamma_bc_w->getResult(0), w});
    wmul.addTypes(w_type);
    Operation *w_prime = rewriter.create(wmul);

    // Emit kernel_call(X_2D, W').
    SmallVector<Value, 2> call_operands = {x_2d, w_prime->getResult(0)};
    SmallVector<NamedAttribute, 1> extra_config;
    extra_config.push_back(rewriter.getNamedAttr(
        "zigrad.normalized_size",
        rewriter.getI32IntegerAttr(static_cast<int32_t>(normalized_size))));

    std::string key = counter->next();
    Operation *replacement = kernel_utils::create_kernel_call(
        dot_op, call_operands, dot_op->getResultTypes(),
        kProvider, key, "rms_norm_matmul", extra_config, rewriter);
    rewriter.replaceOp(dot_op, replacement->getResults());

    // Clean up dead ops in reverse order (only erase if no remaining uses).
    if (reshape_op && reshape_op->use_empty()) rewriter.eraseOp(reshape_op);
    if (convert_to_bf16_op && convert_to_bf16_op->use_empty())
      rewriter.eraseOp(convert_to_bf16_op);
    if (gamma_mul_op->use_empty()) rewriter.eraseOp(gamma_mul_op);
    if (gamma_bc_op->use_empty()) rewriter.eraseOp(gamma_bc_op);
    if (gamma_convert_op && gamma_convert_op->use_empty())
      rewriter.eraseOp(gamma_convert_op);
    if (normed_mul_op->use_empty()) rewriter.eraseOp(normed_mul_op);
    if (inv_bc_op->use_empty()) rewriter.eraseOp(inv_bc_op);
    if (rsqrt_op->use_empty()) rewriter.eraseOp(rsqrt_op);
    if (denom_add_op->use_empty()) rewriter.eraseOp(denom_add_op);
    if (mean_mul_op->use_empty()) rewriter.eraseOp(mean_mul_op);
    if (reduce_op->use_empty()) rewriter.eraseOp(reduce_op);
    if (sq_op->use_empty()) rewriter.eraseOp(sq_op);
    if (x_convert_op && x_convert_op->use_empty())
      rewriter.eraseOp(x_convert_op);

    return success();
  }
};

// softmax(scores) @ V -> softmax_matmul
//
// Matches: dot_general(div(exp(scores), broadcast(reduce_sum(exp(scores)))), V)

struct SoftmaxMatmulPattern final : RewritePattern {
  KeyCounter *counter;

  explicit SoftmaxMatmulPattern(MLIRContext *ctx, KeyCounter *counter)
      : RewritePattern("stablehlo.dot_general", 5, ctx), counter(counter) {}

  LogicalResult matchAndRewrite(Operation *dot_op,
                                PatternRewriter &rewriter) const override {
    if (dot_op->getNumOperands() != 2 || dot_op->getNumResults() != 1)
      return failure();
    if (!kernel_utils::has_matmul_dims(dot_op,
                                       kernel_utils::RhsLayout::standard))
      return failure();

    // Step 1: dot LHS -> divide (attn_probs = exp / sum)
    // Allow an optional stablehlo.convert between divide and dot_general
    // (bf16 models compute softmax in f32, then convert back for the matmul).
    Value dot_lhs = dot_op->getOperand(0);
    Operation *convert_op = nullptr;
    Operation *div_op = dot_lhs.getDefiningOp();
    if (kernel_utils::is_named_op(div_op, "stablehlo.convert")) {
      convert_op = div_op;
      if (convert_op->getNumOperands() != 1) return failure();
      div_op = convert_op->getOperand(0).getDefiningOp();
    }
    if (!kernel_utils::is_named_op(div_op, "stablehlo.divide"))
      return failure();
    if (div_op->getNumOperands() != 2) return failure();

    // Step 2: div operand 0 -> exponential
    Value div_lhs = div_op->getOperand(0);
    Operation *exp_op = div_lhs.getDefiningOp();
    if (!kernel_utils::is_named_op(exp_op, "stablehlo.exponential"))
      return failure();
    if (exp_op->getNumOperands() != 1) return failure();

    // Step 3: div operand 1 -> broadcast_in_dim -> reduce(exp_result)
    Value div_rhs = div_op->getOperand(1);
    Operation *sum_broadcast = div_rhs.getDefiningOp();
    if (!kernel_utils::is_named_op(sum_broadcast, "stablehlo.broadcast_in_dim"))
      return failure();
    if (sum_broadcast->getNumOperands() != 1) return failure();

    Operation *reduce_op = sum_broadcast->getOperand(0).getDefiningOp();
    if (!kernel_utils::is_named_op(reduce_op, "stablehlo.reduce"))
      return failure();

    // Step 4: verify reduce input == exp result (same SSA Value)
    if (reduce_op->getNumOperands() < 1) return failure();
    if (reduce_op->getOperand(0) != exp_op->getResult(0))
      return failure();

    // Verify reduce body is add (reduce_sum).
    if (reduce_op->getNumRegions() != 1) return failure();
    Region &body = reduce_op->getRegion(0);
    if (body.empty() || !body.hasOneBlock()) return failure();
    Block &block = body.front();
    Operation *combiner = nullptr;
    for (Operation &inner_op : block.without_terminator()) {
      combiner = &inner_op;
    }
    if (!combiner || !kernel_utils::is_named_op(combiner, "stablehlo.add"))
      return failure();

    // Step 5: dot RHS = V
    Value v = dot_op->getOperand(1);

    // scores = exp's input
    Value scores = exp_op->getOperand(0);

    // Verify single-use chain.
    // exp feeds both div and reduce, so it should have exactly 2 uses.
    {
      unsigned exp_uses = 0;
      for ([[maybe_unused]] auto &use : exp_op->getResult(0).getUses())
        ++exp_uses;
      if (exp_uses != 2) return failure();
    }
    if (!div_op->getResult(0).hasOneUse()) return failure();
    if (convert_op && !convert_op->getResult(0).hasOneUse()) return failure();
    if (!reduce_op->getResult(0).hasOneUse()) return failure();
    if (!sum_broadcast->getResult(0).hasOneUse()) return failure();

    // Extract reduction_dim and reduction_factor from reduce's dimensions attr.
    // StableHLO reduce uses DenseI64ArrayAttr (not DenseIntElementsAttr).
    auto dims_attr = reduce_op->getAttrOfType<DenseI64ArrayAttr>("dimensions");
    if (!dims_attr || dims_attr.size() != 1)
      return failure();
    int64_t reduction_dim = dims_attr[0];

    // Get reduction_factor from exp_result's tensor type at that dimension.
    auto exp_type = dyn_cast<RankedTensorType>(exp_op->getResult(0).getType());
    if (!exp_type || reduction_dim < 0 || reduction_dim >= exp_type.getRank())
      return failure();
    int64_t reduction_factor = exp_type.getDimSize(reduction_dim);
    if (reduction_factor <= 0) return failure();

    // Store the V dot_general's dimension numbers for faithful expand round-trip.
    Attribute value_dot_dims = dot_op->getAttr("dot_dimension_numbers");
    if (!value_dot_dims) return failure();

    // Emit kernel_call.
    SmallVector<Value, 2> call_operands = {scores, v};
    SmallVector<NamedAttribute, 3> extra_config;
    extra_config.push_back(rewriter.getNamedAttr(
        "zigrad.reduction_dim",
        rewriter.getI32IntegerAttr(static_cast<int32_t>(reduction_dim))));
    extra_config.push_back(rewriter.getNamedAttr(
        "zigrad.reduction_factor",
        rewriter.getI32IntegerAttr(static_cast<int32_t>(reduction_factor))));
    extra_config.push_back(rewriter.getNamedAttr(
        "zigrad.value_dot_dims", value_dot_dims));

    std::string key = counter->next();
    Operation *replacement = kernel_utils::create_kernel_call(
        dot_op, call_operands, dot_op->getResultTypes(),
        kProvider, key, "softmax_matmul", extra_config, rewriter);
    rewriter.replaceOp(dot_op, replacement->getResults());

    // Clean up dead ops.
    if (convert_op && convert_op->use_empty()) rewriter.eraseOp(convert_op);
    if (div_op->use_empty()) rewriter.eraseOp(div_op);
    if (sum_broadcast->use_empty()) rewriter.eraseOp(sum_broadcast);
    if (reduce_op->use_empty()) rewriter.eraseOp(reduce_op);
    if (exp_op->use_empty()) rewriter.eraseOp(exp_op);

    return success();
  }
};

// Q@K -> scale -> stable_softmax -> @V -> attention
//
// Matches the full unmasked attention subgraph:
//   raw_scores = dot_general(Q, K)
//   scaled = multiply(raw_scores, broadcast(constant(scale)))
//   max = reduce(scaled, -inf, maximum, dim)
//   shifted = subtract(scaled, broadcast(max))
//   exp = exponential(shifted)
//   sum = reduce(exp, 0, add, dim)
//   probs = divide(exp, broadcast(sum))
//   [optional convert]
//   result = dot_general(probs, V)

struct AttentionPattern final : RewritePattern {
  KeyCounter *counter;

  explicit AttentionPattern(MLIRContext *ctx, KeyCounter *counter)
      : RewritePattern("stablehlo.dot_general", 6, ctx), counter(counter) {}

  LogicalResult matchAndRewrite(Operation *v_dot_op,
                                PatternRewriter &rewriter) const override {
    if (v_dot_op->getNumOperands() != 2 || v_dot_op->getNumResults() != 1)
      return failure();
    if (!kernel_utils::has_matmul_dims(v_dot_op,
                                       kernel_utils::RhsLayout::standard))
      return failure();

    // Step 1: V dot LHS -> optional convert -> divide (probs = exp / sum)
    Value dot_lhs = v_dot_op->getOperand(0);
    Operation *convert_op = nullptr;
    Operation *div_op = dot_lhs.getDefiningOp();
    if (kernel_utils::is_named_op(div_op, "stablehlo.convert")) {
      convert_op = div_op;
      if (convert_op->getNumOperands() != 1) return failure();
      div_op = convert_op->getOperand(0).getDefiningOp();
    }
    if (!kernel_utils::is_named_op(div_op, "stablehlo.divide"))
      return failure();
    if (div_op->getNumOperands() != 2) return failure();

    // Step 2: div LHS -> exponential
    Operation *exp_op = div_op->getOperand(0).getDefiningOp();
    if (!kernel_utils::is_named_op(exp_op, "stablehlo.exponential"))
      return failure();
    if (exp_op->getNumOperands() != 1) return failure();

    // Step 3: div RHS -> broadcast_in_dim -> reduce(exp, 0, add) = reduce_sum
    Operation *sum_broadcast = div_op->getOperand(1).getDefiningOp();
    if (!kernel_utils::is_named_op(sum_broadcast, "stablehlo.broadcast_in_dim"))
      return failure();
    if (sum_broadcast->getNumOperands() != 1) return failure();

    Operation *sum_reduce = sum_broadcast->getOperand(0).getDefiningOp();
    if (!kernel_utils::is_named_op(sum_reduce, "stablehlo.reduce"))
      return failure();
    if (sum_reduce->getNumOperands() < 1) return failure();
    if (sum_reduce->getOperand(0) != exp_op->getResult(0))
      return failure();

    // Verify reduce_sum body is add.
    {
      if (sum_reduce->getNumRegions() != 1) return failure();
      Region &body = sum_reduce->getRegion(0);
      if (body.empty() || !body.hasOneBlock()) return failure();
      Operation *combiner = nullptr;
      for (Operation &inner : body.front().without_terminator())
        combiner = &inner;
      if (!combiner || !kernel_utils::is_named_op(combiner, "stablehlo.add"))
        return failure();
    }

    // Step 4: exp input -> subtract (shifted = scaled - max_broadcast)
    Operation *sub_op = exp_op->getOperand(0).getDefiningOp();
    if (!kernel_utils::is_named_op(sub_op, "stablehlo.subtract"))
      return failure();
    if (sub_op->getNumOperands() != 2) return failure();

    // Step 5: sub LHS -> scaled = multiply(raw_scores, scale_broadcast)
    //         sub RHS -> broadcast(reduce_max(scaled))
    Value scaled_val = sub_op->getOperand(0);
    Operation *scaled_op = scaled_val.getDefiningOp();
    if (!kernel_utils::is_named_op(scaled_op, "stablehlo.multiply"))
      return failure();
    if (scaled_op->getNumOperands() != 2) return failure();

    // sub RHS -> broadcast_in_dim -> reduce(scaled, -inf, maximum) = reduce_max
    Operation *max_broadcast = sub_op->getOperand(1).getDefiningOp();
    if (!kernel_utils::is_named_op(max_broadcast, "stablehlo.broadcast_in_dim"))
      return failure();
    if (max_broadcast->getNumOperands() != 1) return failure();

    Operation *max_reduce = max_broadcast->getOperand(0).getDefiningOp();
    if (!kernel_utils::is_named_op(max_reduce, "stablehlo.reduce"))
      return failure();
    if (max_reduce->getNumOperands() < 1) return failure();
    if (max_reduce->getOperand(0) != scaled_val)
      return failure();

    // Verify reduce_max body is maximum.
    {
      if (max_reduce->getNumRegions() != 1) return failure();
      Region &body = max_reduce->getRegion(0);
      if (body.empty() || !body.hasOneBlock()) return failure();
      Operation *combiner = nullptr;
      for (Operation &inner : body.front().without_terminator())
        combiner = &inner;
      if (!combiner || !kernel_utils::is_named_op(combiner, "stablehlo.maximum"))
        return failure();
    }

    // Step 6: scaled = multiply(raw_scores, scale_broadcast)
    // One side is dot_general (raw_scores), the other is broadcast(constant(scale))
    Operation *score_dot = nullptr;
    Operation *scale_broadcast = nullptr;
    for (int i = 0; i < 2; ++i) {
      Operation *def = scaled_op->getOperand(i).getDefiningOp();
      if (kernel_utils::is_dot_op(def)) {
        score_dot = def;
      } else if (kernel_utils::is_named_op(def, "stablehlo.broadcast_in_dim")) {
        scale_broadcast = def;
      }
    }
    if (!score_dot || !scale_broadcast) return failure();
    if (score_dot->getNumOperands() != 2) return failure();
    if (!kernel_utils::has_matmul_dims(score_dot,
                                       kernel_utils::RhsLayout::transposed))
      return failure();

    // Extract scale constant value.
    if (scale_broadcast->getNumOperands() != 1) return failure();
    Operation *scale_const = scale_broadcast->getOperand(0).getDefiningOp();
    if (!kernel_utils::is_named_op(scale_const, "stablehlo.constant"))
      return failure();
    auto scale_dense = scale_const->getAttrOfType<DenseElementsAttr>("value");
    if (!scale_dense || !scale_dense.isSplat()) return failure();
    auto scale_type = dyn_cast<FloatType>(scale_dense.getElementType());
    if (!scale_type) return failure();
    float scale_value = scale_dense.getSplatValue<APFloat>().convertToFloat();

    // Step 7: score_dot operands are Q and K.
    Value q = score_dot->getOperand(0);
    Value v = v_dot_op->getOperand(1);
    Value k = score_dot->getOperand(1);

    // Use-count checks.
    // scaled has 2 uses: sub_op and max_reduce.
    {
      unsigned uses = 0;
      for ([[maybe_unused]] auto &u : scaled_op->getResult(0).getUses()) ++uses;
      if (uses != 2) return failure();
    }
    // exp has 2 uses: div_op and sum_reduce.
    {
      unsigned uses = 0;
      for ([[maybe_unused]] auto &u : exp_op->getResult(0).getUses()) ++uses;
      if (uses != 2) return failure();
    }
    if (!div_op->getResult(0).hasOneUse()) return failure();
    if (convert_op && !convert_op->getResult(0).hasOneUse()) return failure();
    if (!sum_reduce->getResult(0).hasOneUse()) return failure();
    if (!sum_broadcast->getResult(0).hasOneUse()) return failure();
    if (!sub_op->getResult(0).hasOneUse()) return failure();
    if (!max_reduce->getResult(0).hasOneUse()) return failure();
    if (!max_broadcast->getResult(0).hasOneUse()) return failure();
    if (!score_dot->getResult(0).hasOneUse()) return failure();
    if (!scale_broadcast->getResult(0).hasOneUse()) return failure();
    if (!scale_const->getResult(0).hasOneUse()) return failure();

    // Extract reduction_dim and reduction_factor from sum_reduce.
    auto sum_dims_attr = sum_reduce->getAttrOfType<DenseI64ArrayAttr>("dimensions");
    if (!sum_dims_attr || sum_dims_attr.size() != 1) return failure();
    int64_t reduction_dim = sum_dims_attr[0];

    auto exp_type = dyn_cast<RankedTensorType>(exp_op->getResult(0).getType());
    if (!exp_type || reduction_dim < 0 || reduction_dim >= exp_type.getRank())
      return failure();
    int64_t reduction_factor = exp_type.getDimSize(reduction_dim);
    if (reduction_factor <= 0) return failure();

    // Extract dot dimension attrs for round-trip in expand pass.
    Attribute score_dot_dims = score_dot->getAttr("dot_dimension_numbers");
    Attribute value_dot_dims = v_dot_op->getAttr("dot_dimension_numbers");
    if (!score_dot_dims || !value_dot_dims) return failure();

    // Scores shape (intermediate tensor shape for expand pass).
    auto scores_type = dyn_cast<RankedTensorType>(score_dot->getResult(0).getType());
    if (!scores_type) return failure();
    SmallVector<int64_t> scores_shape(scores_type.getShape());

    // Emit kernel_call.
    SmallVector<Value, 3> call_operands = {q, k, v};
    SmallVector<NamedAttribute> extra_config;
    extra_config.push_back(rewriter.getNamedAttr(
        "zigrad.scale",
        rewriter.getF32FloatAttr(scale_value)));
    extra_config.push_back(rewriter.getNamedAttr(
        "zigrad.reduction_dim",
        rewriter.getI32IntegerAttr(static_cast<int32_t>(reduction_dim))));
    extra_config.push_back(rewriter.getNamedAttr(
        "zigrad.reduction_factor",
        rewriter.getI32IntegerAttr(static_cast<int32_t>(reduction_factor))));
    extra_config.push_back(rewriter.getNamedAttr(
        "zigrad.score_dot_dims", score_dot_dims));
    extra_config.push_back(rewriter.getNamedAttr(
        "zigrad.value_dot_dims", value_dot_dims));
    extra_config.push_back(rewriter.getNamedAttr(
        "zigrad.scores_shape",
        rewriter.getDenseI64ArrayAttr(scores_shape)));

    std::string key = counter->next();
    Operation *replacement = kernel_utils::create_kernel_call(
        v_dot_op, call_operands, v_dot_op->getResultTypes(),
        kProvider, key, "attention", extra_config, rewriter);
    rewriter.replaceOp(v_dot_op, replacement->getResults());

    // Clean up dead ops (backward order).
    if (convert_op && convert_op->use_empty()) rewriter.eraseOp(convert_op);
    if (div_op->use_empty()) rewriter.eraseOp(div_op);
    if (sum_broadcast->use_empty()) rewriter.eraseOp(sum_broadcast);
    if (sum_reduce->use_empty()) rewriter.eraseOp(sum_reduce);
    if (exp_op->use_empty()) rewriter.eraseOp(exp_op);
    if (sub_op->use_empty()) rewriter.eraseOp(sub_op);
    if (max_broadcast->use_empty()) rewriter.eraseOp(max_broadcast);
    if (max_reduce->use_empty()) rewriter.eraseOp(max_reduce);
    if (scaled_op->use_empty()) rewriter.eraseOp(scaled_op);
    if (scale_broadcast->use_empty()) rewriter.eraseOp(scale_broadcast);
    if (scale_const->use_empty()) rewriter.eraseOp(scale_const);
    if (score_dot->use_empty()) rewriter.eraseOp(score_dot);

    return success();
  }
};

struct MirageKernelSelectPass final
    : PassWrapper<MirageKernelSelectPass, OperationPass<func::FuncOp>> {
  StringRef getArgument() const final { return "zg-mirage-kernel-select"; }

  StringRef getDescription() const final {
    return "Select fuseable StableHLO patterns for the Mirage kernel provider.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::zigrad::ZigradDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    KeyCounter counter;

    RewritePatternSet patterns(&getContext());
    patterns.add<AttentionPattern>(&getContext(), &counter);
    patterns.add<SoftmaxMatmulPattern>(&getContext(), &counter);
    patterns.add<RmsNormMatmulPattern>(&getContext(), &counter);
    // TODO(mirage): Select standalone RMSNorm when Mirage can materialize it.
    patterns.add<DotAddMulPattern>(&getContext(), &counter);
    patterns.add<DotAddPattern>(&getContext(), &counter);
    patterns.add<DotUnaryPattern>(&getContext(), "stablehlo.exponential",
                                  &counter);
    patterns.add<DotUnaryPattern>(&getContext(), "stablehlo.log", &counter);

    if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

void registerMirageKernelSelectPass() {
  static PassRegistration<MirageKernelSelectPass> registration;
  (void)registration;
}

} // namespace mlir::zigrad::mirage
