//! GEMM implementations for benchmark comparison.
//!
//! Provides a naive triple-loop baseline and an optional MKL BLAS binding.
//! Both are comptime-generic over the element type.
const std = @import("std");
const build_options = @import("build_options");

/// Naive strawman triple-loop matmul: C += A @ B.
///
/// i-k-j loop order for marginally better cache behavior on A.
/// Caller must zero C before calling if accumulation is not desired.
///
/// **Would want to exclude this for anything beyond trivial sizes.**
/// *Everything should absolutely demolish this*
pub fn gemm(
    comptime T: type,
    m: i64,
    n: i64,
    k: i64,
    a: []const T,
    lda: usize,
    b: []const T,
    ldb: usize,
    c: []T,
    ldc: usize,
) void {
    const _m: usize = @intCast(m);
    const _n: usize = @intCast(n);
    const _k: usize = @intCast(k);

    for (0.._m) |i| {
        for (0.._k) |k_idx| {
            const a_val = a[i * lda + k_idx];
            for (0.._n) |j| {
                c[i * ldc + j] += a_val * b[k_idx * ldb + j];
            }
        }
    }
}

const blas = if (build_options.has_mkl) @import("c-mkl") else struct {};

pub const has_blas = build_options.has_mkl;

/// MKL BLAS sgemm. Only supports f32 and errors for other types.
pub fn blas_gemm(
    comptime T: type,
    m: i64,
    n: i64,
    k: i64,
    a: []const T,
    lda: usize,
    b: []const T,
    ldb: usize,
    c: []T,
    ldc: usize,
) error{ MklUnavailable, UnsupportedDtype }!void {
    if (!has_blas) return error.MklUnavailable;
    if (comptime T != f32) return error.UnsupportedDtype;
    // Safety: T == f32 verified above; cast through erased pointer to satisfy
    // the generic signature while calling the f32-only C function.
    const a_f32: []const f32 = @ptrCast(a);
    const b_f32: []const f32 = @ptrCast(b);
    const c_f32: []f32 = @ptrCast(c);
    blas.cblas_sgemm(
        blas.CblasRowMajor,
        blas.CblasNoTrans,
        blas.CblasNoTrans,
        @intCast(m),
        @intCast(n),
        @intCast(k),
        1.0,
        a_f32.ptr,
        @intCast(lda),
        b_f32.ptr,
        @intCast(ldb),
        0,
        c_f32.ptr,
        @intCast(ldc),
    );
}

test gemm {
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const b = [_]f32{ 5.0, 6.0, 7.0, 8.0 };
    var c: [4]f32 = @splat(0.0);
    const expected = [_]f32{ 19.0, 22.0, 43.0, 50.0 };

    gemm(f32, 2, 2, 2, &a, 2, &b, 2, &c, 2);

    for (expected, c) |exp, act| {
        try std.testing.expectApproxEqAbs(exp, act, 1e-5);
    }
}

test "gemm f16" {
    const a = [_]f16{ 1.0, 2.0, 3.0, 4.0 };
    const b = [_]f16{ 5.0, 6.0, 7.0, 8.0 };
    var c: [4]f16 = @splat(0.0);
    const expected = [_]f16{ 19.0, 22.0, 43.0, 50.0 };

    gemm(f16, 2, 2, 2, &a, 2, &b, 2, &c, 2);

    for (expected, c) |exp, act| {
        try std.testing.expectApproxEqAbs(exp, act, 0.1);
    }
}
