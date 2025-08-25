const std = @import("std");

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const deps = try setupDependencies(b, target, optimize);

    // Create the mlxzig module for external consumption
    const mlxzig = b.addModule("mlxzig", .{
        .root_source_file = b.path("src/mlx.zig"),
        .target = target,
        .optimize = optimize,
    });

    const lib = b.addLibrary(.{
        .linkage = .static,
        .name = "mlxzig",
        .root_module = mlxzig,
    });
    configureExecutable(lib, b, deps);
    b.installArtifact(lib);

    const llm_options = try LlmOptions.fromOptions(b);
    const llm_module = b.createModule(.{
        .root_source_file = b.path("src/llm.zig"),
        .target = target,
        .optimize = optimize,
    });
    llm_module.addImport("build_options", llm_options.createModule(b));
    const llm_exe = b.addExecutable(.{
        .name = "llm",
        .root_module = llm_module,
    });
    configureExecutable(llm_exe, b, deps);
    b.installArtifact(llm_exe);

    const whisper_module = b.createModule(.{
        .root_source_file = b.path("src/whisper.zig"),
        .target = target,
        .optimize = optimize,
    });
    whisper_module.addImport("build_options", llm_options.createModule(b));
    const whisper_exe = b.addExecutable(.{
        .name = "whisper",
        .root_module = whisper_module,
    });
    configureExecutable(whisper_exe, b, deps);
    b.installArtifact(whisper_exe);

    const llm_run = b.addRunArtifact(llm_exe);
    if (b.args) |args| llm_run.addArgs(args);
    const run_llm = b.step("run-llm", "Run LLM app");
    run_llm.dependOn(&llm_run.step);
    const whisper_run = b.addRunArtifact(whisper_exe);
    if (b.args) |args| whisper_run.addArgs(args);
    const run_whisper = b.step("run-whisper", "Run TTS app");
    run_whisper.dependOn(&whisper_run.step);
    const run_step = b.step("run", "Run default app"); // : run=run-llm for now
    run_step.dependOn(&llm_run.step);
    const test_step = b.step("test", "Run all tests");
    const main_test_module = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    const main_tests = b.addTest(.{
        .root_module = main_test_module,
    });

    configureExecutable(main_tests, b, deps);
    const run_main_tests = b.addRunArtifact(main_tests);
    test_step.dependOn(&run_main_tests.step);
}

const LlmOptions = struct {
    config: ?[]const u8,
    format: ?[]const u8,
    model_type: ?[]const u8,
    model_name: ?[]const u8,
    max: ?usize,

    fn fromOptions(b: *std.Build) !LlmOptions {
        return LlmOptions{
            .config = b.option([]const u8, "config", "Config: phi, llama, qwen, olympic"),
            .model_type = b.option([]const u8, "model-type", "Model-type: phi, llama, qwen"),
            .model_name = b.option([]const u8, "model-name", "Model-name"),
            .format = b.option([]const u8, "format", "Chat format"),
            .max = b.option(usize, "max", "Maximum number of tokens to generate"),
        };
    }

    fn createModule(self: LlmOptions, b: *std.Build) *std.Build.Module {
        const options_pkg = b.addOptions();
        options_pkg.addOption(?[]const u8, "config", self.config);
        options_pkg.addOption(?[]const u8, "format", self.format);
        options_pkg.addOption(?[]const u8, "model_type", self.model_type);
        options_pkg.addOption(?[]const u8, "model_name", self.model_name);
        options_pkg.addOption(?usize, "max", self.max);

        return options_pkg.createModule();
    }
};

const Dependencies = struct {
    mlx_c_path: []const u8,
    mlx_c_build_path: []const u8,
    mlx_c_lib_path: []const u8,
    install_step: *std.Build.Step,
    pcre2_dep: *std.Build.Dependency,
};

fn setupDependencies(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) !Dependencies {
    const mlx_c_path = b.pathJoin(&.{ b.cache_root.path.?, "mlx-c" });
    const mlx_c_build_path = b.pathJoin(&.{ mlx_c_path, "build" });
    const mlx_c_lib_path = b.pathJoin(&.{ mlx_c_build_path, "libmlxc.a" });
    const install_step = b.step("install-mlx-c", "Install MLX-C if needed");
    const needs_install = !doesFileExist(mlx_c_lib_path);
    if (needs_install) {
        const clone_cmd = b.addSystemCommand(&[_][]const u8{ "sh", "-c", b.fmt("if [ ! -d {s} ]; then mkdir -p $(dirname {s}) && curl -L https://github.com/ml-explore/mlx-c/archive/refs/tags/v0.2.0.tar.gz | tar xz -C $(dirname {s}) && mv $(dirname {s})/mlx-c-0.2.0 {s}; fi", .{ mlx_c_path, mlx_c_path, mlx_c_path, mlx_c_path, mlx_c_path }) });
        const mkdir_cmd = b.addSystemCommand(&[_][]const u8{ "mkdir", "-p", mlx_c_build_path });
        mkdir_cmd.step.dependOn(&clone_cmd.step);
        const cmake_cmd = b.addSystemCommand(&[_][]const u8{ "cmake", "..", "-DCMAKE_BUILD_TYPE=Release" });
        cmake_cmd.setCwd(.{ .cwd_relative = mlx_c_build_path });
        cmake_cmd.step.dependOn(&mkdir_cmd.step);
        const make_cmd = b.addSystemCommand(&[_][]const u8{ "make", "-j" });
        make_cmd.setCwd(.{ .cwd_relative = mlx_c_build_path });
        make_cmd.step.dependOn(&cmake_cmd.step);
        install_step.dependOn(&make_cmd.step);
    }
    if (doesFileExist(b.pathJoin(&.{ mlx_c_build_path, "_deps/mlx-build/mlx.metallib" }))) {
        const dest_dir = b.pathJoin(&.{ b.install_path, "lib", "metal" });
        const mkdir_cmd = b.addSystemCommand(&.{ "mkdir", "-p", dest_dir });
        const copy_cmd = b.addSystemCommand(&.{
            "cp",
            b.pathJoin(&.{ mlx_c_build_path, "_deps/mlx-build/mlx.metallib" }),
            b.pathJoin(&.{ dest_dir, "mlx.metallib" }),
        });
        copy_cmd.step.dependOn(&mkdir_cmd.step);
        b.getInstallStep().dependOn(&copy_cmd.step);
    }
    const pcre2_dep = b.dependency("pcre2", .{
        .target = target,
        .optimize = optimize,
    });
    return Dependencies{
        .mlx_c_path = mlx_c_path,
        .mlx_c_build_path = mlx_c_build_path,
        .mlx_c_lib_path = mlx_c_lib_path,
        .install_step = install_step,
        .pcre2_dep = pcre2_dep,
    };
}

fn configureExecutable(
    exe: *std.Build.Step.Compile,
    b: *std.Build,
    deps: Dependencies,
) void {
    exe.step.dependOn(deps.install_step);
    exe.addIncludePath(.{ .cwd_relative = deps.mlx_c_path });
    exe.addObjectFile(.{ .cwd_relative = b.pathJoin(&.{ deps.mlx_c_build_path, "libmlxc.a" }) });
    exe.addObjectFile(.{ .cwd_relative = b.pathJoin(&.{ deps.mlx_c_build_path, "_deps/mlx-build/libmlx.a" }) });
    exe.linkFramework("Metal");
    exe.linkFramework("Foundation");
    exe.linkFramework("QuartzCore");
    exe.linkFramework("Accelerate");
    exe.linkLibCpp();
    exe.linkLibrary(deps.pcre2_dep.artifact("pcre2-8"));
}

fn doesFileExist(path: []const u8) bool {
    var file = std.fs.cwd().openFile(path, .{}) catch |err| {
        if (err == error.FileNotFound) return false;
        return false;
    };
    file.close();
    return true;
}
