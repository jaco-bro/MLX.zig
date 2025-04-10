//! llama.zig - Llama-3.2-Instruct
//!
//! Copyright 2025 Joe

const std = @import("std");
const mlx = @import("mlx.zig");
const loadJson = @import("utils.zig").loadJson;
const allocJoin = @import("utils.zig").allocJoin;

pub const MLP = struct {
    const Self = @This();
    base: mlx.Module,
    gate_weight: *mlx.Linear,
    up_weight: *mlx.Linear,
    down_weight: *mlx.Linear,

    pub fn init(mlx_config: mlx.MLXConfig, key: []const u8, quant_config: ?mlx.QuantConfig, weights_hash: *std.StringHashMap(*mlx.Array)) !*Self {
        const self = try mlx_config.allocator.create(Self);
        self.* = .{
            .base = mlx.Module.init(mlx_config.allocator, mlx_config.stream),
            .gate_weight = undefined,
            .up_weight = undefined,
            .down_weight = undefined,
        };
        const gate_key = try self.base.allocJoin(key, "gate_proj");
        self.gate_weight = try mlx.Linear.init(mlx_config, gate_key, false, quant_config, weights_hash);
        const up_key = try self.base.allocJoin(key, "up_proj");
        self.up_weight = try mlx.Linear.init(mlx_config, up_key, false, quant_config, weights_hash);
        const down_key = try self.base.allocJoin(key, "down_proj");
        self.down_weight = try mlx.Linear.init(mlx_config, down_key, false, quant_config, weights_hash);
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.gate_weight.deinit();
        self.up_weight.deinit();
        self.down_weight.deinit();
        self.base.deinit();
        self.base.allocator.destroy(self);
    }

    pub fn forward(self: *Self, result: *mlx.Array, x: mlx.Array) !void {
        var gate = mlx.arrayNew();
        var sigmoid = mlx.arrayNew();
        var up = mlx.arrayNew();
        defer {
            mlx.arrayFree(gate);
            mlx.arrayFree(sigmoid);
            mlx.arrayFree(up);
        }
        try self.gate_weight.forward(&gate, x);
        try mlx.sigmoid(&sigmoid, gate, self.base.stream);
        try mlx.multiply(&gate, gate, sigmoid, self.base.stream);
        try self.up_weight.forward(&up, x);
        try mlx.multiply(&up, gate, up, self.base.stream);
        try self.down_weight.forward(result, up);
    }
};

pub const Attention = struct {
    const Self = @This();
    base: mlx.Module,
    n_heads: c_int,
    n_kv_heads: c_int,
    head_dim: c_int,
    n_repeat: c_int,
    scale: mlx.Array,
    q_weight: *mlx.Linear,
    k_weight: *mlx.Linear,
    v_weight: *mlx.Linear,
    o_weight: *mlx.Linear,
    rope: *Llama3RoPE,

    pub fn init(mlx_config: mlx.MLXConfig, key: []const u8, n_heads: c_int, n_kv_heads: c_int, head_dim: c_int, rope_theta: f32, rope_scaling_config: LlamaConfig.RopeScalingConfig, quant_config: ?mlx.QuantConfig, weights_hash: *std.StringHashMap(*mlx.Array)) !*Self {
        const self = try mlx_config.allocator.create(Self);
        self.* = .{
            .base = mlx.Module.init(mlx_config.allocator, mlx_config.stream),
            .n_heads = n_heads,
            .n_kv_heads = n_kv_heads,
            .head_dim = head_dim,
            .n_repeat = @divExact(n_heads, n_kv_heads),
            .scale = mlx.arrayNewFloat(1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)))),
            .q_weight = undefined,
            .k_weight = undefined,
            .v_weight = undefined,
            .o_weight = undefined,
            .rope = undefined,
        };
        const q_key = try self.base.allocJoin(key, "q_proj");
        self.q_weight = try mlx.Linear.init(mlx_config, q_key, false, quant_config, weights_hash);
        const k_key = try self.base.allocJoin(key, "k_proj");
        self.k_weight = try mlx.Linear.init(mlx_config, k_key, false, quant_config, weights_hash);
        const v_key = try self.base.allocJoin(key, "v_proj");
        self.v_weight = try mlx.Linear.init(mlx_config, v_key, false, quant_config, weights_hash);
        const o_key = try self.base.allocJoin(key, "o_proj");
        self.o_weight = try mlx.Linear.init(mlx_config, o_key, false, quant_config, weights_hash);
        self.rope = try Llama3RoPE.init(mlx_config, head_dim, rope_theta, rope_scaling_config);
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.q_weight.deinit();
        self.k_weight.deinit();
        self.v_weight.deinit();
        self.o_weight.deinit();
        self.rope.deinit();
        mlx.arrayFree(self.scale);
        self.base.deinit();
        self.base.allocator.destroy(self);
    }

    pub fn forward(self: *Self, result: *mlx.Array, x: mlx.Array, mask: ?mlx.Array, cache: ?*mlx.KVCache, offset: c_int) !void {
        var q = mlx.arrayNew();
        var k = mlx.arrayNew();
        var v = mlx.arrayNew();
        var w = mlx.arrayNew();
        defer {
            mlx.arrayFree(q);
            mlx.arrayFree(k);
            mlx.arrayFree(v);
            mlx.arrayFree(w);
        }
        try self.q_weight.forward(&q, x);
        try self.k_weight.forward(&k, x);
        try self.v_weight.forward(&v, x);
        try mlx.rEshap(&q, q, "b l (h d) -> b h l d", .{ .h = self.n_heads, .d = self.head_dim }, self.base.stream);
        try mlx.rEshap(&k, k, "b l (h d) -> b h l d", .{ .h = self.n_kv_heads, .d = self.head_dim }, self.base.stream);
        try mlx.rEshap(&v, v, "b l (h d) -> b h l d", .{ .h = self.n_kv_heads, .d = self.head_dim }, self.base.stream);
        try self.rope.forward(&q, q, offset);
        try self.rope.forward(&k, k, offset);
        try mlx.multiply(&q, q, self.scale, self.base.stream);
        if (cache) |c| try c.update(&k, &v, null, self.base.stream);
        try mlx.rEpeat(&k, k, "b h l d -> b (repeat h) l d", .{ .repeat = self.n_repeat }, self.base.stream);
        try mlx.rEpeat(&v, v, "b h l d -> b (repeat h) l d", .{ .repeat = self.n_repeat }, self.base.stream);
        try mlx.einsum(&w, .{ q, k }, "b h l d, b h k d -> b h l k", self.base.stream);
        if (mask) |m| try mlx.add(&w, w, m, self.base.stream);
        try mlx.softmax(&w, w, &.{3}, true, self.base.stream);
        try mlx.einsum(&w, .{ w, v }, "b h l k, b h k d -> b h l d", self.base.stream);
        try mlx.rEshap(&w, w, "b h l d -> b l (h d)", .{}, self.base.stream);
        try self.o_weight.forward(result, w);
    }
};

pub const Llama3RoPE = struct {
    const Self = @This();
    base: mlx.Module,
    freqs: mlx.Array,
    rope_base: mlx.OptionalFloat,
    dims: c_int,

    pub fn init(mlx_config: mlx.MLXConfig, dims: c_int, theta: f32, scaling_config: LlamaConfig.RopeScalingConfig) !*Self {
        const self = try mlx_config.allocator.create(Self);
        self.* = .{
            .base = mlx.Module.init(mlx_config.allocator, mlx_config.stream),
            .rope_base = mlx.OptionalFloat{ .has_value = false, .value = 0.0 },
            .freqs = mlx.arrayNew(),
            .dims = dims,
        };
        var wavelens = mlx.arrayNew();
        var high_freq_mask = mlx.arrayNew();
        var mid_freq_mask = mlx.arrayNew();
        var high_freq = mlx.arrayNew();
        var smooth_factors = mlx.arrayNew();
        var mid_freq = mlx.arrayNew();
        defer {
            mlx.arrayFree(wavelens);
            mlx.arrayFree(high_freq_mask);
            mlx.arrayFree(mid_freq_mask);
            mlx.arrayFree(high_freq);
            mlx.arrayFree(smooth_factors);
            mlx.arrayFree(mid_freq);
        }
        try mlx.arange(&self.freqs, 0, @floatFromInt(dims), 2, mlx.FLOAT32, self.base.stream);
        try mlx.divide(&self.freqs, self.freqs, mlx.float(@floatFromInt(dims)), self.base.stream);
        try mlx.power(&self.freqs, mlx.float(theta), self.freqs, self.base.stream);
        try mlx.multiply(&wavelens, mlx.float(2.0 * std.math.pi), self.freqs, self.base.stream);
        try mlx.multiply(&high_freq, self.freqs, mlx.float(scaling_config.factor), self.base.stream);
        try mlx.greater(&high_freq_mask, wavelens, mlx.float(scaling_config.original_max_position_embeddings / scaling_config.low_freq_factor), self.base.stream);
        try mlx.where(&high_freq, high_freq_mask, high_freq, self.freqs, self.base.stream);
        try mlx.lessEqual(&mid_freq_mask, wavelens, mlx.float(scaling_config.original_max_position_embeddings / scaling_config.high_freq_factor), self.base.stream);
        try mlx.logicalOr(&mid_freq_mask, high_freq_mask, mid_freq_mask, self.base.stream);
        try mlx.logicalNot(&mid_freq_mask, mid_freq_mask, self.base.stream);
        try mlx.divide(&smooth_factors, mlx.float(scaling_config.original_max_position_embeddings), wavelens, self.base.stream);
        try mlx.subtract(&smooth_factors, smooth_factors, mlx.float(scaling_config.low_freq_factor), self.base.stream);
        try mlx.divide(&smooth_factors, smooth_factors, mlx.float(scaling_config.high_freq_factor - scaling_config.low_freq_factor), self.base.stream);
        try mlx.subtract(&mid_freq, mlx.float(1.0), smooth_factors, self.base.stream);
        try mlx.divide(&mid_freq, mid_freq, mlx.float(scaling_config.factor), self.base.stream);
        try mlx.add(&mid_freq, mid_freq, smooth_factors, self.base.stream);
        try mlx.divide(&mid_freq, self.freqs, mid_freq, self.base.stream);
        try mlx.where(&high_freq, high_freq_mask, high_freq, self.freqs, self.base.stream);
        return self;
    }

    pub fn forward(self: *Self, result: *mlx.Array, x: mlx.Array, offset: c_int) !void {
        try mlx.fastRope(result, x, self.dims, false, self.rope_base, 1.0, offset, self.freqs, self.base.stream);
        try mlx.astype(result, result.*, mlx.BFLOAT16, self.base.stream);
    }

    pub fn deinit(self: *Self) void {
        mlx.arrayFree(self.freqs);
        self.base.deinit();
        self.base.allocator.destroy(self);
    }
};

pub const TransformerBlock = struct {
    const Self = @This();
    base: mlx.Module,
    attention: *Attention,
    mlp: *MLP,
    input_layernorm: *mlx.RMSNorm,
    post_attention_layernorm: *mlx.RMSNorm,

    pub fn init(mlx_config: mlx.MLXConfig, key: []const u8, layer_idx: usize, model_config: *const LlamaConfig, weights_hash: *std.StringHashMap(*mlx.Array)) !*Self {
        const self = try mlx_config.allocator.create(Self);
        self.* = .{
            .base = mlx.Module.init(mlx_config.allocator, mlx_config.stream),
            .attention = undefined,
            .mlp = undefined,
            .input_layernorm = undefined,
            .post_attention_layernorm = undefined,
        };
        const layer_key = try self.base.allocJoin(key, layer_idx);
        const attn_key = try self.base.allocJoin(layer_key, "self_attn");
        self.attention = try Attention.init(mlx_config, attn_key, model_config.num_attention_heads, model_config.num_key_value_heads, model_config.head_dim, model_config.rope_theta, model_config.rope_scaling, model_config.quantization, weights_hash);
        const mlp_key = try self.base.allocJoin(layer_key, "mlp");
        self.mlp = try MLP.init(mlx_config, mlp_key, model_config.quantization, weights_hash);
        const in_ln_key = try self.base.allocJoin(layer_key, "input_layernorm");
        self.input_layernorm = try mlx.RMSNorm.init(mlx_config, in_ln_key, model_config.rms_norm_eps, weights_hash);
        const post_ln_key = try self.base.allocJoin(layer_key, "post_attention_layernorm");
        self.post_attention_layernorm = try mlx.RMSNorm.init(mlx_config, post_ln_key, model_config.rms_norm_eps, weights_hash);
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.attention.deinit();
        self.mlp.deinit();
        self.input_layernorm.deinit();
        self.post_attention_layernorm.deinit();
        self.base.deinit();
        self.base.allocator.destroy(self);
    }

    pub fn forward(self: *Self, result: *mlx.Array, x: mlx.Array, mask: ?mlx.Array, cache: ?*mlx.KVCache, offset: c_int) !void {
        var attn = mlx.arrayNew();
        var mlp_out = mlx.arrayNew();
        defer {
            mlx.arrayFree(attn);
            mlx.arrayFree(mlp_out);
        }
        try self.input_layernorm.forward(&attn, x);
        try self.attention.forward(&attn, attn, mask, cache, offset);
        try mlx.add(&attn, attn, x, self.base.stream);
        try self.post_attention_layernorm.forward(&mlp_out, attn);
        try self.mlp.forward(&mlp_out, mlp_out);
        try mlx.add(result, mlp_out, attn, self.base.stream);
    }
};

pub const LlamaModel = struct {
    const Self = @This();
    base: mlx.Module,
    embed_tokens: *mlx.Embedding,
    layers: []*TransformerBlock,
    norm: *mlx.RMSNorm,

    pub fn init(mlx_config: mlx.MLXConfig, key: []const u8, model_config: *const LlamaConfig, weights_hash: *std.StringHashMap(*mlx.Array)) !*Self {
        const self = try mlx_config.allocator.create(Self);
        self.* = .{
            .base = mlx.Module.init(mlx_config.allocator, mlx_config.stream),
            .embed_tokens = undefined,
            .layers = undefined,
            .norm = undefined,
        };
        const embed_key = try self.base.allocJoin(key, "embed_tokens");
        self.embed_tokens = try mlx.Embedding.init(mlx_config, embed_key, model_config.quantization, weights_hash);
        const norm_key = try self.base.allocJoin(key, "norm");
        self.norm = try mlx.RMSNorm.init(mlx_config, norm_key, model_config.rms_norm_eps, weights_hash);
        const layers_key = try self.base.allocJoin(key, "layers");
        self.layers = try mlx_config.allocator.alloc(*TransformerBlock, @intCast(model_config.num_hidden_layers));
        for (0..@intCast(model_config.num_hidden_layers)) |i| {
            self.layers[i] = try TransformerBlock.init(mlx_config, layers_key, i, model_config, weights_hash);
        }
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.embed_tokens.deinit();
        for (self.layers) |layer| {
            layer.deinit();
        }
        self.base.allocator.free(self.layers);
        self.norm.deinit();
        self.base.deinit();
        self.base.allocator.destroy(self);
    }

    pub fn forward(self: *Self, result: *mlx.Array, toks: mlx.Array, mask: ?mlx.Array, cache: ?*mlx.Cache) !void {
        const seq_len = mlx.arrayDim(toks, 1);
        const offset = if (cache) |c| c.offset else 0;
        var x = mlx.arrayNew();
        defer mlx.arrayFree(x);
        try self.embed_tokens.forward(&x, toks);
        for (self.layers, 0..) |layer, i| {
            const layer_cache = if (cache) |c| &c.layers[i] else null;
            try layer.forward(&x, x, mask, layer_cache, offset);
        }
        try self.norm.forward(result, x);
        if (cache) |c| c.offset += seq_len;
    }
};

pub const Model = struct {
    const Self = @This();
    base: mlx.Module,
    model: *LlamaModel,
    tie_word_embeddings: bool,
    lm_head: ?*mlx.Linear,

    pub fn init(mlx_config: mlx.MLXConfig, model_config: *const LlamaConfig, weights_hash: *std.StringHashMap(*mlx.Array)) !*Self {
        const self = try mlx_config.allocator.create(Self);
        self.* = .{
            .base = mlx.Module.init(mlx_config.allocator, mlx_config.stream),
            .tie_word_embeddings = model_config.tie_word_embeddings,
            .model = undefined,
            .lm_head = undefined,
        };
        self.model = try LlamaModel.init(mlx_config, "model", model_config, weights_hash);
        if (!model_config.tie_word_embeddings) {
            self.lm_head = try mlx.Linear.init(mlx_config, "lm_head", false, model_config.quantization, weights_hash);
        } else {
            self.lm_head = null;
        }
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.model.deinit();
        if (!self.tie_word_embeddings and self.lm_head != null) {
            self.lm_head.?.deinit();
        }
        self.base.deinit();
        self.base.allocator.destroy(self);
    }

    pub fn forward(self: *Self, result: *mlx.Array, toks: mlx.Array, mask: ?mlx.Array, cache: ?*mlx.Cache) !void {
        var x = mlx.arrayNew();
        defer mlx.arrayFree(x);
        try self.model.forward(&x, toks, mask, cache);
        if (self.tie_word_embeddings) {
            try self.model.embed_tokens.asLinear(result, x);
        } else {
            try self.lm_head.?.forward(result, x);
        }
    }
};

pub const Transformer = struct {
    const Self = @This();
    mlx_config: mlx.MLXConfig,
    model: *Model,
    eos_token_id: []u32,

    pub fn init(allocator: std.mem.Allocator, model_path: []const u8) !Self {
        var buf: [1024]u8 = undefined;
        var mlx_config = try mlx.MLXConfig.init(allocator);
        errdefer mlx_config.deinit();
        const path_config = try std.fmt.bufPrintZ(&buf, "{s}/config.json", .{model_path});
        const model_config = try loadJson(LlamaConfig, allocator, path_config, true);
        defer model_config.deinit();
        const eos_token_id = try allocator.dupe(u32, model_config.value.eos_token_id);
        errdefer allocator.free(eos_token_id);
        const path_weight = try std.fmt.bufPrintZ(&buf, "{s}/model.safetensors", .{model_path});
        var safetensors = try mlx.Safetensors.load(path_weight, mlx_config.stream);
        defer safetensors.deinit();
        var weights_hash = std.StringHashMap(*mlx.Array).init(allocator);
        defer weights_hash.deinit();
        var model = try Model.init(mlx_config, &model_config.value, &weights_hash);
        errdefer model.deinit();
        try safetensors.unload(&weights_hash);
        return .{
            .mlx_config = mlx_config,
            .model = model,
            .eos_token_id = eos_token_id,
        };
    }

    pub fn deinit(self: *Self) void {
        self.model.deinit();
        self.mlx_config.allocator.free(self.eos_token_id);
        self.mlx_config.deinit();
    }

    pub fn generate(self: *Self, initial_tokens: []const u32, num_tokens: usize) ![]u32 {
        std.debug.print("\nInput IDs: {any}\n\n", .{initial_tokens});
        var output_tokens = try self.mlx_config.allocator.alloc(u32, num_tokens);
        errdefer self.mlx_config.allocator.free(output_tokens);
        var cache = try mlx.Cache.init(self.mlx_config.allocator, self.model.model.layers.len, 2);
        defer cache.deinit();
        var toks = try mlx.arrayNewData(initial_tokens.ptr, .{ 1, initial_tokens.len }, mlx.UINT32);
        var logits = mlx.arrayNew();
        var mask = mlx.arrayNew();
        defer {
            mlx.arrayFree(toks);
            mlx.arrayFree(logits);
            mlx.arrayFree(mask);
        }
        var start_time = std.time.milliTimestamp();
        var prompt_ms: f16 = undefined;
        var i: usize = 0;
        while (i < num_tokens) : (i += 1) {
            try mlx.createCausalMask(&mask, mlx.arrayDim(toks, 1), cache.offset, mlx.BFLOAT16, self.mlx_config.stream);
            try self.model.forward(&logits, toks, mask, &cache);
            try mlx.take(&logits, logits, mlx.int(-1), 1, self.mlx_config.stream);
            try mlx.argmax(&logits, logits, 1, false, self.mlx_config.stream);
            try mlx.item(&output_tokens[i], logits);
            try mlx.arraySetData(&toks, &output_tokens[i], .{ 1, 1 }, mlx.UINT32);
            std.debug.print("Generated token {d}/{d}: {d}\n", .{ i + 1, num_tokens, output_tokens[i] });
            if (std.mem.indexOfScalar(u32, self.eos_token_id, output_tokens[i]) != null) {
                i += 1;
                break;
            }
            if (i == 0) {
                const current_time = std.time.milliTimestamp();
                prompt_ms = @floatFromInt(current_time - start_time);
                start_time = current_time;
            }
        }
        const end_time = std.time.milliTimestamp();
        if (i < num_tokens) {
            output_tokens = try self.mlx_config.allocator.realloc(output_tokens, i);
        }
        std.debug.print("\nOutput IDs: {any}\n", .{output_tokens});
        const prompt_tps = @as(f16, @floatFromInt(initial_tokens.len)) / (prompt_ms / 1000.0);
        std.debug.print("\nPrompt:     {d:.2} tokens-per-second ({d} tokens in {d:.2} ms)\n", .{ prompt_tps, initial_tokens.len, prompt_ms });
        if (i > 0) {
            const gen_ms = @as(f16, @floatFromInt(end_time - start_time));
            const gen_tps = @as(f16, @floatFromInt(i)) / (gen_ms / 1000.0);
            std.debug.print("Generation: {d:.2} tokens-per-second ({d} tokens in {d:.2} ms)\n", .{ gen_tps, i, gen_ms });
        }
        return output_tokens;
    }
};

pub const LlamaConfig = struct {
    eos_token_id: []u32,
    hidden_size: c_int = 2048,
    intermediate_size: c_int = 8192,
    num_attention_heads: c_int = 32,
    num_key_value_heads: c_int = 8,
    head_dim: c_int = 64,
    max_position_embeddings: c_int = 131072,
    rms_norm_eps: f32 = 1e-5,
    rope_theta: f32 = 500000.0,
    mlp_bias: bool = false,
    attention_bias: bool = false,
    tie_word_embeddings: bool = true,
    vocab_size: c_int = 128256,
    num_hidden_layers: c_int = 16,
    quantization: ?mlx.QuantConfig = null,
    rope_scaling: RopeScalingConfig,
    pub const RopeScalingConfig = struct {
        factor: f32 = 32.0,
        high_freq_factor: f32 = 4.0,
        low_freq_factor: f32 = 1.0,
        original_max_position_embeddings: f32 = 8192,
    };
};

test "Transformer generating" {
    std.debug.print("\n=== LLAMA.ZIG ===\n\n", .{});
    const allocator = std.testing.allocator;
    const initial_tokens = [_]u32{ 9906, 1917 };
    const num_tokens_to_generate = 10;
    var transformer = try Transformer.init(allocator, "Llama-3.2-1B-Instruct-4bit");
    defer transformer.deinit();
    const generated_tokens = try transformer.generate(&initial_tokens, num_tokens_to_generate);
    defer allocator.free(generated_tokens);
    std.debug.print("\nGenerated sequence: ", .{});
    for (generated_tokens) |token| {
        std.debug.print("{d} ", .{token});
    }
    std.debug.print("\n", .{});
}
