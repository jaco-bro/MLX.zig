//! phi.zig - Phi-4
//!
//! Copyright 2025

const std = @import("std");
const mlx = @import("mlx.zig");
const utils = @import("utils.zig");

pub const ModelConfig = mlx.ModelConfig;
pub const Model = mlx.Model(TransformerBlock, ModelConfig);
pub const Transformer = mlx.Transformer(Model, ModelConfig);

pub const MLP = struct {
    const Self = @This();
    base: mlx.Module = undefined,
    gate_up_proj: *mlx.Linear = undefined,
    down_proj: *mlx.Linear = undefined,

    pub fn init(key: []const u8, model_config: ModelConfig, mlx_config: *mlx.MLXConfig) !*Self {
        const self = try mlx_config.allocator.create(Self);
        self.* = .{ .base = mlx.Module.init(mlx_config.allocator, mlx_config.stream) };
        self.gate_up_proj = try mlx.Linear.init(try self.base.allocJoin(key, "gate_up_proj"), false, model_config.quantization, mlx_config);
        self.down_proj = try mlx.Linear.init(try self.base.allocJoin(key, "down_proj"), false, model_config.quantization, mlx_config);
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.gate_up_proj.deinit();
        self.down_proj.deinit();
        self.base.deinit();
        self.base.allocator.destroy(self);
    }

    pub fn forward(self: *Self, result: *mlx.Array, x: mlx.Array) !void {
        var gate_up = mlx.arrayNew();
        var gate = mlx.arrayNew();
        var up = mlx.arrayNew();
        defer {
            mlx.arrayFree(gate_up);
            mlx.arrayFree(gate);
            mlx.arrayFree(up);
        }
        try self.gate_up_proj.forward(&gate_up, x);
        try mlx.splitEqualParts(&.{ &gate, &up }, gate_up, 2, 2, self.base.stream);
        try mlx.silu(&gate, gate, self.base.stream);
        try mlx.multiply(&up, gate, up, self.base.stream);
        try self.down_proj.forward(result, up);
    }
};

pub const Attention = struct {
    const Self = @This();
    base: mlx.Module = undefined,
    n_heads: c_int = undefined,
    n_kv_heads: c_int = undefined,
    head_dim: c_int = undefined,
    scale: f32 = undefined,
    q_pos: c_int = undefined,
    k_pos: c_int = undefined,
    qkv_proj: *mlx.Linear = undefined,
    o_proj: *mlx.Linear = undefined,
    rope: *mlx.RoPE = undefined,

    pub fn init(key: []const u8, model_config: ModelConfig, mlx_config: *mlx.MLXConfig) !*Self {
        const self = try mlx_config.allocator.create(Self);
        self.* = .{ .base = mlx.Module.init(mlx_config.allocator, mlx_config.stream) };
        self.n_heads = model_config.num_attention_heads;
        self.n_kv_heads = model_config.num_key_value_heads orelse model_config.num_attention_heads;
        self.head_dim = @divExact(model_config.hidden_size, model_config.num_attention_heads);
        self.qkv_proj = try mlx.Linear.init(try self.base.allocJoin(key, "qkv_proj"), false, model_config.quantization, mlx_config);
        self.o_proj = try mlx.Linear.init(try self.base.allocJoin(key, "o_proj"), false, model_config.quantization, mlx_config);
        self.rope = try mlx.RoPE.init(self.head_dim, false, model_config.rope_theta, 1.0, mlx_config);
        self.scale = 1.0 / @sqrt(@as(f32, @floatFromInt(self.head_dim)));
        self.q_pos = self.n_heads * self.head_dim;
        self.k_pos = self.q_pos + self.n_kv_heads * self.head_dim;
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.qkv_proj.deinit();
        self.o_proj.deinit();
        self.rope.deinit();
        self.base.deinit();
        self.base.allocator.destroy(self);
    }

    pub fn forward(self: *Self, result: *mlx.Array, x: mlx.Array, mask: ?mlx.Array, cache: ?*mlx.KVCache, offset: c_int) !void {
        var qkv = mlx.arrayNew();
        var q = mlx.arrayNew();
        var k = mlx.arrayNew();
        var v = mlx.arrayNew();
        defer {
            mlx.arrayFree(qkv);
            mlx.arrayFree(q);
            mlx.arrayFree(k);
            mlx.arrayFree(v);
        }
        try self.qkv_proj.forward(&qkv, x);
        try mlx.split_sections(&.{ &q, &k, &v }, qkv, &[_]c_int{ self.q_pos, self.k_pos }, 2, self.base.stream);
        try mlx.rEshap(&q, q, "b l (h d) -> b h l d", .{ .h = self.n_heads, .d = self.head_dim }, self.base.stream);
        try mlx.rEshap(&k, k, "b l (h d) -> b h l d", .{ .h = self.n_kv_heads, .d = self.head_dim }, self.base.stream);
        try mlx.rEshap(&v, v, "b l (h d) -> b h l d", .{ .h = self.n_kv_heads, .d = self.head_dim }, self.base.stream);
        try self.rope.forward(&q, q, offset);
        try self.rope.forward(&k, k, offset);
        if (cache) |c| try c.update(&k, &v, null, self.base.stream);
        try mlx.fastScaledDotProductAttention(result, q, k, v, self.scale, mask, self.base.stream);
        try mlx.rEshap(result, result.*, "b h l d -> b l (h d)", .{}, self.base.stream);
        try self.o_proj.forward(result, result.*);
    }
};

pub const TransformerBlock = struct {
    const Self = @This();
    base: mlx.Module = undefined,
    self_attn: *Attention = undefined,
    mlp: *MLP = undefined,
    input_layernorm: *mlx.RMSNorm = undefined,
    post_attention_layernorm: *mlx.RMSNorm = undefined,

    pub fn init(key: []const u8, model_config: ModelConfig, mlx_config: *mlx.MLXConfig) !*Self {
        const self = try mlx_config.allocator.create(Self);
        self.* = .{ .base = mlx.Module.init(mlx_config.allocator, mlx_config.stream) };
        self.self_attn = try Attention.init(try self.base.allocJoin(key, "self_attn"), model_config, mlx_config);
        self.mlp = try MLP.init(try self.base.allocJoin(key, "mlp"), model_config, mlx_config);
        self.input_layernorm = try mlx.RMSNorm.init(try self.base.allocJoin(key, "input_layernorm"), model_config.rms_norm_eps, mlx_config);
        self.post_attention_layernorm = try mlx.RMSNorm.init(try self.base.allocJoin(key, "post_attention_layernorm"), model_config.rms_norm_eps, mlx_config);
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.self_attn.deinit();
        self.mlp.deinit();
        self.input_layernorm.deinit();
        self.post_attention_layernorm.deinit();
        self.base.deinit();
        self.base.allocator.destroy(self);
    }

    pub fn forward(self: *Self, result: *mlx.Array, x: mlx.Array, mask: ?mlx.Array, cache: ?*mlx.KVCache, offset: c_int) !void {
        var attn = mlx.arrayNew();
        var mlp = mlx.arrayNew();
        defer {
            mlx.arrayFree(attn);
            mlx.arrayFree(mlp);
        }
        try self.input_layernorm.forward(&attn, x);
        try self.self_attn.forward(&attn, attn, mask, cache, offset);
        try mlx.add(&attn, x, attn, self.base.stream);
        try self.post_attention_layernorm.forward(&mlp, attn);
        try self.mlp.forward(&mlp, mlp);
        try mlx.add(result, attn, mlp, self.base.stream);
    }
};

test "phi.zig" {
    std.debug.print("\n=== PHI.ZIG ===\n\n", .{});
    const allocator = std.testing.allocator;
    const initial_tokens = [_]u32{ 100264, 9125, 100266 };
    const num_tokens_to_generate = 10;
    var transformer = try Transformer.init(allocator, "phi-4-2bit");
    defer transformer.deinit();
    const generated_tokens = try transformer.generate(&initial_tokens, num_tokens_to_generate);
    defer allocator.free(generated_tokens);
    std.debug.print("\nGenerated sequence: ", .{});
    for (generated_tokens) |token| {
        std.debug.print("{d} ", .{token});
    }
    std.debug.print("\n", .{});
}
