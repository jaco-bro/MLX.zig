//! main.zig - Default entry point
//!
//! Copyright 2025 Joe

const std = @import("std");
pub fn main() !void {
    std.debug.print(
        \\
        \\MLX Zig ML Models
        \\================
        \\
        \\Usage:
        \\
        \\  zig build [options]                    
        \\  zig build run-llm [comptime_options] -- [runtime_options]
        \\  zig build run-whisper -- [audio_file]
        \\
    , .{});
}
test "Redirects" {
    std.debug.print("\n=== MAIN.ZIG ===\n\n", .{});
    _ = @import("llm.zig");
    _ = @import("whisper.zig");
    std.testing.refAllDecls(@import("mlx.zig"));
}
