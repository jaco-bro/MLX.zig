# MLX.zig

A [Zig](https://ziglang.org/) binding for [MLX](https://github.com/ml-explore/mlx), Apple's array framework for machine learning on Apple Silicon.

## Prerequisites

- Apple Silicon Mac
- Zig v0.14.0
- CMake

## Using in your own project

After importing the dependency with `zig fetch --save clone https://github.com/jaco-bro/MLX.zig.git` add something like this to your `build.zig` file:

```fish
    const mlxzig = b.dependency("mlxzig", .{
        .target = target,
        .optimize = optimize,
    });
    exe_mod.linkLibrary(mlxzig.artifact("mlxzig"));
    exe_mod.addImport("mlxzig", mlxzig.module("mlxzig"));
```

this will allow you to import the library in your code like this:

```zig
const mlx = @import("mlxzig");
```

## Getting Started

```fish
git clone https://github.com/jaco-bro/MLX.zig.git
cd MLX.zig
zig build
```

This creates executables in `zig-out/bin/`:
- `llm` - Unified interface for LLM models (Llama-3.2, Phi-4, Qwen-2.5, ...)
- `whisper` - Speech-to-text using Whisper-Turbo-Large-v3

## Whisper Speech-to-Text

```fish
zig build run-whisper [-- audio_file.mp3]
# or
zig-out/bin/whisper audio_file.mp3
```

## LLM Interface

```fish
zig build run-llm [-- options]
# or
zig-out/bin/llm [options] [input]
```

### Options

```
--config=CONFIG         Config: llama, phi, qwen, olympic (default: qwen)
--format=FORMAT         Custom chat format template (defaults based on config)
--model-type=TYPE       Model type: llama, phi, qwen, ... (defaults based on config)
--model-name=NAME       Model name (defaults based on config)
--max=N                 Maximum tokens to generate (default: 30)
--help                  Show this help
```

### Examples

```fish
zig build -Dconfig=phi -Dformat={s}
zig build run-llm -Dmax=100 -- "Write a python function to check if a number is prime"
zig-out/bin/llm --config=qwen --format={s} "Write a python function to check if a number is prime"
```

The library supports several model configurations including QwQ, R1-Distill-Qwen, Qwen-2.5-Coder, Llama-3.2, and Phi-4.

## Acknowledgements

Inspired by Erik Kaunismäki's [zig-build-mlx](https://github.com/ErikKaum/zig-build-mlx).

## License

[Apache License 2.0](LICENSE)
