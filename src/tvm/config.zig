//! Pure Zig configuration for the optional TVM integration.

const std = @import("std");
const builtin = @import("builtin");
const cuda_nvrtc = @import("../cuda/nvrtc.zig");
const device_mod = @import("device");
const runtime = @import("../runtime.zig");
const linker = @import("../toolchain/linker.zig");

const log = std.log.scoped(.@"zg/tvm_config");

pub const ffi_path_env = "ZG_TVM_FFI_PATH";
pub const runtime_path_env = "ZG_TVM_RUNTIME_PATH";
pub const compiler_path_env = "ZG_TVM_COMPILER_PATH";
pub const cuda_intrinsics_path_env = "ZG_TVM_CUDA_INTRINSICS_PATH";
pub const compiler_fingerprint_env = "ZG_TVM_COMPILER_FINGERPRINT";
pub const default_ffi_path = "libtvm_ffi.so";
pub const default_runtime_path = "libtvm_runtime.so";
pub const default_compiler_path = "libtvm.so";
pub const CompileConfigError = cuda_nvrtc.ConfigError || linker.ConfigError || error{
    /// Compiler fingerprint used by artifact cache keys was not configured.
    MissingCompilerFingerprint,
    /// CUDA TensorIntrin definitions were not configured.
    MissingCudaIntrinsics,
};

/// TVM compilation target.
pub const TargetKind = enum {
    cpu,
    cuda,

    /// Return whether this compiler target can consume the selected device.
    pub fn accepts(self: TargetKind, device: device_mod.Device) bool {
        return switch (self) {
            .cpu => device.platform.eql(.cpu),
            .cuda => device.platform.eql(.cuda),
        };
    }
};

/// Maximum TVM capability available to one process.
pub const RuntimeSurface = enum {
    ffi,
    runtime,
    compiler,

    /// Return whether this surface satisfies an operation's requirement.
    pub fn satisfies(self: RuntimeSurface, requirement: RuntimeSurface) bool {
        return switch (self) {
            .ffi => requirement == .ffi,
            .runtime => switch (requirement) {
                .ffi, .runtime => true,
                .compiler => false,
            },
            .compiler => true,
        };
    }
};

/// Runtime-loaded TVM libraries.
pub const RuntimeConfig = struct {
    /// Maximum TVM capability selected before the first library load.
    surface: RuntimeSurface,

    /// TVM FFI library that provides the packed-call ABI.
    ffi: runtime.RuntimeLibrary = .{ .path = default_ffi_path },

    /// TVM deployment library that executes generated modules.
    runtime: runtime.RuntimeLibrary = .{ .path = default_runtime_path },

    /// TVM compiler library that registers compilation functions.
    compiler: runtime.RuntimeLibrary = .{ .path = default_compiler_path },

    /// Resolve optional library overrides from an environment map.
    pub fn from_environ(
        environ: *const std.process.Environ.Map,
        surface: RuntimeSurface,
    ) RuntimeConfig {
        return .{
            .surface = surface,
            .ffi = .from_environ(environ, ffi_path_env, default_ffi_path),
            .runtime = .from_environ(
                environ,
                runtime_path_env,
                default_runtime_path,
            ),
            .compiler = .from_environ(
                environ,
                compiler_path_env,
                default_compiler_path,
            ),
        };
    }
};

/// Borrowed compilation inputs. The environment outlives compilation and providers.
pub const TargetInputs = union(TargetKind) {
    cpu,
    cuda: CudaInputs,

    pub fn kind(self: TargetInputs) TargetKind {
        return std.meta.activeTag(self);
    }
};

/// Required CUDA compilation inputs borrowed from the configuration environment.
pub const CudaInputs = struct {
    nvrtc: cuda_nvrtc.Config,
    intrinsics_path: []const u8,
};

/// Opaque cache identity combining this lowering build with external compilation inputs.
pub const BuildIdentity = struct {
    digest: [64]u8,

    pub fn from_external(external: []const u8) BuildIdentity {
        var hash = std.crypto.hash.Blake3.init(.{});
        hash.update("zigrad.tvm.build.v1\x00");
        hash.update(@import("build_options").tvm_lowering_identity);
        hash.update(external);
        var digest: [32]u8 = undefined;
        hash.final(&digest);
        return .{ .digest = std.fmt.bytesToHex(digest, .lower) };
    }
};

/// Target-specific inputs used while TVM compiles candidates.
pub const CompileConfig = struct {
    /// External TVM, LLVM, CUDA and intrinsic identity borrowed from the environment.
    external_identity: []const u8,

    /// Compilation target.
    target: TargetInputs,

    /// Linker used to produce candidate shared libraries.
    linker: linker.Linker,

    /// Resolve target-specific inputs from an environment map.
    pub fn from_environ(
        environ: *const std.process.Environ.Map,
        target: TargetKind,
    ) CompileConfigError!CompileConfig {
        return .{
            .external_identity = try resolve_compiler_fingerprint(environ),
            .linker = try .from_environ(environ),
            .target = switch (target) {
                .cpu => .cpu,
                .cuda => blk: {
                    const nvrtc = try cuda_nvrtc.Config.from_environ(environ);
                    const intrinsics = environ.get(cuda_intrinsics_path_env) orelse {
                        if (!builtin.is_test) log.err("CUDA compilation requires {s}", .{cuda_intrinsics_path_env});
                        return error.MissingCudaIntrinsics;
                    };
                    if (intrinsics.len == 0) {
                        if (!builtin.is_test) log.err("CUDA compilation input {s} is empty", .{cuda_intrinsics_path_env});
                        return error.MissingCudaIntrinsics;
                    }
                    break :blk .{ .cuda = .{ .nvrtc = nvrtc, .intrinsics_path = intrinsics } };
                },
            },
        };
    }

    /// Combine the current lowering build with this configuration's external inputs.
    pub fn build_identity(self: CompileConfig) BuildIdentity {
        return .from_external(self.external_identity);
    }

    /// Resolve cache identity for loading artifacts without compilation configuration.
    pub fn resolve_build_identity(environ: *const std.process.Environ.Map) error{MissingCompilerFingerprint}!BuildIdentity {
        return .from_external(try resolve_compiler_fingerprint(environ));
    }

    /// Resolve the compiler fingerprint from an environment map.
    pub fn resolve_compiler_fingerprint(
        environ: *const std.process.Environ.Map,
    ) error{MissingCompilerFingerprint}![]const u8 {
        const fingerprint = environ.get(compiler_fingerprint_env) orelse {
            if (!builtin.is_test) log.err("TVM compilation requires {s}", .{compiler_fingerprint_env});
            return error.MissingCompilerFingerprint;
        };
        if (fingerprint.len == 0) {
            if (!builtin.is_test) log.err("TVM compilation input {s} is empty", .{compiler_fingerprint_env});
            return error.MissingCompilerFingerprint;
        }
        return fingerprint;
    }
};

test "RuntimeConfig resolves TVM library paths" {
    var environ: std.process.Environ.Map = .init(std.testing.allocator);
    defer environ.deinit();

    const defaults = RuntimeConfig.from_environ(&environ, .runtime);
    try std.testing.expectEqual(RuntimeSurface.runtime, defaults.surface);
    try std.testing.expectEqualStrings(default_ffi_path, defaults.ffi.path);
    try std.testing.expectEqualStrings(default_runtime_path, defaults.runtime.path);
    try std.testing.expectEqualStrings(default_compiler_path, defaults.compiler.path);

    try environ.put(ffi_path_env, "/runtime/libtvm_ffi.so");
    try environ.put(runtime_path_env, "/runtime/libtvm_runtime.so");
    try environ.put(compiler_path_env, "/runtime/libtvm.so");
    const configured = RuntimeConfig.from_environ(&environ, .compiler);
    try std.testing.expectEqual(RuntimeSurface.compiler, configured.surface);
    try std.testing.expectEqualStrings("/runtime/libtvm_ffi.so", configured.ffi.path);
    try std.testing.expectEqualStrings("/runtime/libtvm_runtime.so", configured.runtime.path);
    try std.testing.expectEqualStrings("/runtime/libtvm.so", configured.compiler.path);
}

test "RuntimeSurface satisfies weaker operation requirements" {
    try std.testing.expect(RuntimeSurface.compiler.satisfies(.runtime));
    try std.testing.expect(RuntimeSurface.runtime.satisfies(.ffi));
    try std.testing.expect(!RuntimeSurface.runtime.satisfies(.compiler));
    try std.testing.expect(!RuntimeSurface.ffi.satisfies(.runtime));
}

test "CompileConfig resolves NVRTC host inputs only for CUDA" {
    var environ: std.process.Environ.Map = .init(std.testing.allocator);
    defer environ.deinit();

    try std.testing.expectError(
        error.MissingCompilerFingerprint,
        CompileConfig.from_environ(&environ, .cpu),
    );
    try environ.put(compiler_fingerprint_env, "");
    try std.testing.expectError(error.MissingCompilerFingerprint, CompileConfig.from_environ(&environ, .cpu));
    try environ.put(compiler_fingerprint_env, "tvm-test-compiler");
    try std.testing.expectError(
        error.MissingLinker,
        CompileConfig.from_environ(&environ, .cpu),
    );
    try environ.put(linker.path_env, "/toolchain/bin/ld.lld");
    const cpu = try CompileConfig.from_environ(&environ, .cpu);
    try std.testing.expectEqualStrings("tvm-test-compiler", cpu.external_identity);
    try std.testing.expectEqual(TargetKind.cpu, cpu.target.kind());
    try std.testing.expectEqualStrings("/toolchain/bin/ld.lld", cpu.linker.executable);

    try std.testing.expectError(
        error.MissingCudaToolkitRoot,
        CompileConfig.from_environ(&environ, .cuda),
    );
    try environ.put(cuda_nvrtc.cuda_home_env, "/cuda");
    try std.testing.expectError(
        error.MissingCudaIntrinsics,
        CompileConfig.from_environ(&environ, .cuda),
    );
    try environ.put(cuda_intrinsics_path_env, "");
    try std.testing.expectError(error.MissingCudaIntrinsics, CompileConfig.from_environ(&environ, .cuda));
    try environ.put(cuda_intrinsics_path_env, "/compiler/cuda_tensor_intrinsics.json");
    const cuda = try CompileConfig.from_environ(&environ, .cuda);
    try std.testing.expectEqual(TargetKind.cuda, cuda.target.kind());
    try std.testing.expectEqualStrings(
        "/compiler/cuda_tensor_intrinsics.json",
        cuda.target.cuda.intrinsics_path,
    );
}

test "TargetKind requires device agreement" {
    const cpu: device_mod.Device = .{ .platform = .cpu };
    const cuda: device_mod.Device = .{ .platform = .cuda };
    try std.testing.expect(TargetKind.cpu.accepts(cpu));
    try std.testing.expect(TargetKind.cuda.accepts(cuda));
    try std.testing.expect(!TargetKind.cpu.accepts(cuda));
    try std.testing.expect(!TargetKind.cuda.accepts(cpu));
}
