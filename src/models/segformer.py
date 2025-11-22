import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from .nn import default, RMSNorm


# ATTENTION MECHANISMS
def scaled_dot_product_attention(query, key, value, mask=None):
    dim_k = query.size(-1)
    scores = torch.bmm(query, key.transpose(1, 2)) / torch.sqrt(torch.tensor(dim_k, dtype=torch.float32))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    weights = F.softmax(scores, dim=-1)
    return weights.bmm(value)

class AttentionGated(nn.Module):
    def __init__(self, embed_dim_query, embed_dim_key, embed_dim_value,
                 embed_dim_shortcut, head_dim):
        super().__init__()
        self.norm = RMSNorm(head_dim)
        self.q = nn.Sequential(
            nn.Linear(embed_dim_query, head_dim),
            nn.SiLU()
        )
        self.k = nn.Sequential(
            nn.Linear(embed_dim_key, head_dim),
            nn.SiLU()
        )
        self.v = nn.Sequential(
            nn.Linear(embed_dim_value, head_dim),
            nn.SiLU()
        )
        # Fix: Alpha và beta phải từ hidden_value (cùng số tokens với value)
        self.alpha = nn.Sequential(
            nn.Linear(embed_dim_value, head_dim // 2),  # Changed from embed_dim_key
            nn.Linear(head_dim // 2, head_dim),
            nn.Sigmoid()
        )
        self.beta = nn.Sequential(
            nn.Linear(embed_dim_value, head_dim),  # Changed from embed_dim_key
            nn.Sigmoid()
        )
        self.shortcut = nn.Sequential(
            nn.Linear(embed_dim_shortcut, head_dim // 2),
            nn.Linear(head_dim // 2, head_dim),
            nn.Sigmoid()
        )
        self.linear1 = nn.Linear(head_dim, head_dim)

    def forward(self, hidden_query: Tensor, hidden_key: Tensor,
                hidden_value: Tensor, hidden_shortcut: Tensor):
        query = F.normalize(self.q(hidden_query), p=2, dim=-1)
        key = F.normalize(self.k(hidden_key), p=2, dim=-1)
        value = self.v(hidden_value)

        # Fix: Tính alpha và beta từ hidden_value thay vì hidden_key
        alpha = self.alpha(hidden_value)
        beta = self.beta(hidden_value)

        shortcut = self.shortcut(hidden_shortcut)
        if shortcut.shape[1] != query.shape[1]:
            # [B, num_tokens, head_dim] -> [B, head_dim, num_tokens]
            shortcut = shortcut.transpose(1, 2)
            # Interpolate to match query length
            shortcut = F.interpolate(shortcut, size=query.shape[1], mode='linear', align_corners=False)
            # [B, head_dim, num_query_tokens] -> [B, num_query_tokens, head_dim]
            shortcut = shortcut.transpose(1, 2)

        value = value * alpha + beta

        attn_outputs = scaled_dot_product_attention(query=query, key=key, value=value)
        attn_outputs = self.norm(attn_outputs) * shortcut
        attn_outputs = self.linear1(attn_outputs)
        return attn_outputs

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim_query, embed_dim_key, embed_dim_value,
                 embed_dim_shortcut, embed_dim, num_heads):
        super().__init__()
        head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"

        self.heads = nn.ModuleList([
            AttentionGated(embed_dim_query, embed_dim_key, embed_dim_value,
                          embed_dim_shortcut, head_dim) for _ in range(num_heads)
        ])
        self.output_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, hidden_query: Tensor, hidden_key: Tensor,
                hidden_value: Tensor, hidden_shortcut: Tensor):
        x = torch.cat([h(hidden_query, hidden_key, hidden_value, hidden_shortcut)
                       for h in self.heads], dim=-1)
        x = self.output_linear(x)
        return x


# MIXTURE OF EXPERTS
class Expert(nn.Module):
    def __init__(self, hidden_dim: int, ffn_dim: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, ffn_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate = F.silu(self.w1(x))
        val = self.w3(x)
        x = self.w2(gate * val)
        return x

class AdvancedExpert(nn.Module):
    def __init__(self, hidden_dim: int, ffn_dim: int):
        super().__init__()
        self.norm = RMSNorm(hidden_dim)
        self.w1 = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, ffn_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.norm(x)
        gate = F.silu(self.w1(x))
        val = self.w3(x)
        out = self.w2(gate * val)
        return out + residual

class MixtureOfExperts(nn.Module):
    def __init__(self, hidden_dim: int, num_routed_experts: int = 32,
                 num_activated_experts: int = 4, num_shared_expert: int = 8,
                 router_rank: int = 64):
        super().__init__()
        ffn_dim_routed = hidden_dim * 2  # Giảm từ 4 xuống 2
        ffn_dim_shared = hidden_dim * 3  # Giảm từ 8 xuống 3
        self.hidden_dim = hidden_dim
        self.num_routed_experts = num_routed_experts
        self.num_activated_experts = num_activated_experts
        self.num_shared_expert = num_shared_expert

        self.shared_experts = nn.ModuleList([
            AdvancedExpert(hidden_dim, ffn_dim_shared) for _ in range(num_shared_expert)
        ])
        self.routed_experts = nn.ModuleList([
            Expert(hidden_dim, ffn_dim_routed) for _ in range(num_routed_experts)
        ])
        self.router_down = nn.Linear(hidden_dim, router_rank, bias=False)
        self.router_up = nn.Linear(router_rank, num_routed_experts, bias=False)

    def forward(self, x: Tensor):
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)

        # Shared experts - giảm memory bằng cách tính trung bình ngay
        shared_output = torch.zeros_like(x_flat)
        for expert in self.shared_experts:
            shared_output = shared_output + expert(x_flat)
        shared_output = shared_output / self.num_shared_expert

        # Routing với gradient checkpointing
        router_hidden = self.router_down(x_flat)
        router_logits = self.router_up(router_hidden)
        routing_weights, selected_experts = torch.topk(router_logits, self.num_activated_experts, dim=-1)
        routing_weights = F.softmax(routing_weights, dim=-1)

        # Efficient expert computation
        routed_output = torch.zeros_like(x_flat)
        for i in range(self.num_activated_experts):
            expert_indices = selected_experts[:, i]
            expert_weights = routing_weights[:, i:i+1]
            for expert_idx in range(self.num_routed_experts):
                mask = (expert_indices == expert_idx)
                if mask.any():
                    token_indices = mask.nonzero(as_tuple=True)[0]
                    expert_input = x_flat[token_indices]
                    expert_output = self.routed_experts[expert_idx](expert_input)
                    routed_output[token_indices] = routed_output[token_indices] + expert_output * expert_weights[token_indices]

        output = shared_output + routed_output
        output = output.view(batch_size, seq_len, hidden_dim)

        # Simplified auxiliary losses
        expert_counts = torch.bincount(selected_experts.flatten(), minlength=self.num_routed_experts)
        load_balance_loss = expert_counts.float().var()

        aux_loss_dict = {
            'load_balance_loss': load_balance_loss,
            'router_entropy': torch.tensor(0.0, device=x.device),
            'shared_routed_balance_loss': torch.tensor(0.0, device=x.device)
        }
        return output, aux_loss_dict


# TRANSFORMER LAYERS
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim: int = 512, num_heads: int = 8,
                 num_routed_experts: int = 32, num_activated_experts: int = 4,
                 num_shared_expert: int = 8, router_rank: int = 64, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.layer_norm_1 = nn.LayerNorm(normalized_shape=embed_dim)
        self.layer_norm_2 = nn.LayerNorm(normalized_shape=embed_dim)
        self.attention = MultiHeadAttention(
            embed_dim_query=embed_dim, embed_dim_key=embed_dim,
            embed_dim_value=embed_dim, embed_dim_shortcut=embed_dim,
            embed_dim=embed_dim, num_heads=num_heads
        )
        self.MoE = MixtureOfExperts(
            hidden_dim=embed_dim, num_routed_experts=num_routed_experts,
            num_activated_experts=num_activated_experts, num_shared_expert=num_shared_expert,
            router_rank=router_rank
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor):
        residual = x
        x = self.layer_norm_1(x)
        attn_output = self.attention(x, x, x, x)
        attn_output = self.dropout(attn_output)
        x = residual + attn_output

        residual = x
        x = self.layer_norm_2(x)
        moe_output, aux_loss_dict = self.MoE(x)
        moe_output = self.dropout(moe_output)
        x = residual + moe_output
        return x, aux_loss_dict

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim: int = 512, num_heads: int = 8, num_layers: int = 2,
                 num_routed_experts: int = 32, num_activated_experts: int = 4,
                 num_shared_expert: int = 8, router_rank: int = 64, dropout: float = 0.1):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, num_routed_experts,
                                   num_activated_experts, num_shared_expert,
                                   router_rank, dropout) for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor):
        all_aux_losses = []
        for layer in self.layers:
            x, aux_loss_dict = layer(x)
            all_aux_losses.append(aux_loss_dict)
        x = self.final_norm(x)
        return x, all_aux_losses


# CROSS VIT COMPONENTS
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, context=None, kv_include_self=False):
        b, n, _ = x.shape
        h = self.heads
        x = self.norm(x)
        context = default(context, x)
        if kv_include_self:
            context = torch.cat((x, context), dim=1)
        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class ProjectInOut(nn.Module):
    def __init__(self, dim_in, dim_out, fn):
        super().__init__()
        self.fn = fn
        need_projection = dim_in != dim_out
        self.project_in = nn.Linear(dim_in, dim_out) if need_projection else nn.Identity()
        self.project_out = nn.Linear(dim_out, dim_in) if need_projection else nn.Identity()

    def forward(self, x, *args, **kwargs):
        x = self.project_in(x)
        x = self.fn(x, *args, **kwargs)
        x = self.project_out(x)
        return x

class CrossTransformer(nn.Module):
    def __init__(self, sm_dim, lg_dim, depth, heads, dim_head, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                ProjectInOut(sm_dim, lg_dim, Attention(lg_dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                ProjectInOut(lg_dim, sm_dim, Attention(sm_dim, heads=heads, dim_head=dim_head, dropout=dropout))
            ]))

    def forward(self, sm_tokens, lg_tokens):
        (sm_cls, sm_patch_tokens), (lg_cls, lg_patch_tokens) = map(
            lambda t: (t[:, :1], t[:, 1:]), (sm_tokens, lg_tokens)
        )
        for sm_attend_lg, lg_attend_sm in self.layers:
            sm_cls = sm_attend_lg(sm_cls, context=lg_patch_tokens, kv_include_self=True) + sm_cls
            lg_cls = lg_attend_sm(lg_cls, context=sm_patch_tokens, kv_include_self=True) + lg_cls
        sm_tokens = torch.cat((sm_cls, sm_patch_tokens), dim=1)
        lg_tokens = torch.cat((lg_cls, lg_patch_tokens), dim=1)
        return sm_tokens, lg_tokens

class MultiScaleEncoder(nn.Module):
    def __init__(self, *, depth, sm_dim, lg_dim, sm_enc_params, lg_enc_params,
                 cross_attn_heads, cross_attn_depth, cross_attn_dim_head=64, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Transformer(dim=sm_dim, dropout=dropout, **sm_enc_params),
                Transformer(dim=lg_dim, dropout=dropout, **lg_enc_params),
                CrossTransformer(sm_dim=sm_dim, lg_dim=lg_dim, depth=cross_attn_depth,
                               heads=cross_attn_heads, dim_head=cross_attn_dim_head, dropout=dropout)
            ]))

    def forward(self, sm_tokens, lg_tokens):
        for sm_enc, lg_enc, cross_attend in self.layers:
            sm_tokens, lg_tokens = sm_enc(sm_tokens), lg_enc(lg_tokens)
            sm_tokens, lg_tokens = cross_attend(sm_tokens, lg_tokens)
        return sm_tokens, lg_tokens

class ImageEmbedder(nn.Module):
    def __init__(self, *, dim, image_size, patch_size, dropout=0., channels=3):
        super().__init__()
        assert image_size % patch_size == 0, 'Image size must be divisible by patch size'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        return self.dropout(x)

class CrossViT(nn.Module):
    def __init__(self, *, image_size, sm_dim, lg_dim, sm_patch_size=12, sm_enc_depth=1,
                 sm_enc_heads=8, sm_enc_mlp_dim=1024, sm_enc_dim_head=64, lg_patch_size=16,
                 lg_enc_depth=4, lg_enc_heads=8, lg_enc_mlp_dim=1024, lg_enc_dim_head=64,
                 cross_attn_depth=2, cross_attn_heads=8, cross_attn_dim_head=64, depth=3,
                 dropout=0.1, emb_dropout=0.1, channels=3, use_projection=False, output_dim=None):
        super().__init__()
        self.sm_image_embedder = ImageEmbedder(
            dim=sm_dim, channels=channels, image_size=image_size,
            patch_size=sm_patch_size, dropout=emb_dropout
        )
        self.lg_image_embedder = ImageEmbedder(
            dim=lg_dim, channels=channels, image_size=image_size,
            patch_size=lg_patch_size, dropout=emb_dropout
        )
        self.multi_scale_encoder = MultiScaleEncoder(
            depth=depth, sm_dim=sm_dim, lg_dim=lg_dim,
            cross_attn_heads=cross_attn_heads, cross_attn_dim_head=cross_attn_dim_head,
            cross_attn_depth=cross_attn_depth,
            sm_enc_params=dict(depth=sm_enc_depth, heads=sm_enc_heads,
                              mlp_dim=sm_enc_mlp_dim, dim_head=sm_enc_dim_head),
            lg_enc_params=dict(depth=lg_enc_depth, heads=lg_enc_heads,
                              mlp_dim=lg_enc_mlp_dim, dim_head=lg_enc_dim_head),
            dropout=dropout
        )
        self.use_projection = use_projection
        if use_projection:
            assert output_dim is not None, "output_dim must be specified when use_projection=True"
            self.sm_projection = nn.Linear(sm_dim, output_dim)
            self.lg_projection = nn.Linear(lg_dim, output_dim)
            self.output_dim = output_dim
        else:
            self.output_dim = None

    def forward(self, img, return_concat=False):
        sm_tokens = self.sm_image_embedder(img)
        lg_tokens = self.lg_image_embedder(img)
        sm_tokens, lg_tokens = self.multi_scale_encoder(sm_tokens, lg_tokens)
        sm_patch_tokens = sm_tokens[:, 1:, :]
        lg_patch_tokens = lg_tokens[:, 1:, :]

        if return_concat:
            if self.use_projection:
                sm_projected = self.sm_projection(sm_patch_tokens)
                lg_projected = self.lg_projection(lg_patch_tokens)
                return torch.cat([sm_projected, lg_projected], dim=1)
            else:
                raise ValueError("Cannot concat tokens with different dimensions.")
        return sm_patch_tokens, lg_patch_tokens


# TRANSFORMER DECODER
class TransformerDecoderLayer(nn.Module):
    def __init__(self, image_size: int = 512, channels: int = 3,
                 embed_dim: int = 512, num_heads: int = 8,
                 num_routed_experts: int = 32, num_activated_experts: int = 4,
                 num_shared_expert: int = 8, router_rank: int = 64,
                 dropout: float = 0.1, first_layer: bool = False,
                 encoder_feature_channels: int = None, encoder_feature_size: int = None,
                 decoder_feature_channels: int = None, decoder_feature_size: int = None):
        super().__init__()

        self.first_layer = first_layer
        self.embed_dim = embed_dim
        self.image_size = image_size

        # Store decoder feature size for key generation
        assert decoder_feature_size is not None, "decoder_feature_size must be provided"
        self.decoder_feature_size = decoder_feature_size
        self.num_spatial_tokens = decoder_feature_size * decoder_feature_size

        # Layer normalization
        self.layer_norm_q = nn.LayerNorm(normalized_shape=embed_dim)
        self.layer_norm_k = nn.LayerNorm(normalized_shape=embed_dim)
        self.layer_norm_v = nn.LayerNorm(normalized_shape=embed_dim)
        self.layer_norm_shortcut = nn.LayerNorm(normalized_shape=embed_dim)
        self.layer_norm_1 = nn.LayerNorm(normalized_shape=embed_dim)
        self.layer_norm_2 = nn.LayerNorm(normalized_shape=embed_dim)

        # ============ QUERY PROJECTION ============
        # Project query to match spatial token count
        self.query_adapter = nn.Linear(embed_dim, embed_dim)

        # ============ LEARNABLE PROJECTION FOR VALUE ============
        assert decoder_feature_channels is not None and decoder_feature_size is not None, \
            "decoder_feature_channels and decoder_feature_size must be provided"

        self.value_projection = nn.Linear(
            decoder_feature_channels,
            embed_dim
        )

        self.value_num_tokens = self.decoder_feature_size * self.decoder_feature_size

        # ============ LEARNABLE PROJECTION FOR SHORTCUT ============
        if first_layer:
            assert encoder_feature_channels is not None and encoder_feature_size is not None, \
                "encoder_feature_channels and encoder_feature_size must be provided for first_layer=True"

            self.shortcut_projection = nn.Linear(
                encoder_feature_channels * self.decoder_feature_size,
                embed_dim
            )
            self.shortcut_num_tokens = self.decoder_feature_size * self.decoder_feature_size
        else:
            self.shortcut_projection = None
            self.shortcut_num_tokens = None

        # ============ KEY PROJECTION FROM IMAGE ============
        self.key_conv = nn.Sequential(
            nn.Conv2d(channels, embed_dim, kernel_size=3, padding=1),
            nn.GroupNorm(1, embed_dim),
            nn.GELU()
        )
        self.key_pool = nn.AdaptiveAvgPool2d((self.decoder_feature_size, self.decoder_feature_size))

        # Multi-head attention
        self.attention = MultiHeadAttention(
            embed_dim_query=embed_dim, embed_dim_key=embed_dim,
            embed_dim_value=embed_dim, embed_dim_shortcut=embed_dim,
            embed_dim=embed_dim, num_heads=num_heads
        )

        # Mixture of Experts
        self.MoE = MixtureOfExperts(
            hidden_dim=embed_dim,
            num_routed_experts=num_routed_experts,
            num_activated_experts=num_activated_experts,
            num_shared_expert=num_shared_expert,
            router_rank=router_rank
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, image: Tensor, q: Tensor, v: Tensor, shortcut: Tensor):
        batch_size = q.shape[0]

        # ============ QUERY: Adapt to spatial token count ============
        # q shape: [B, num_query_tokens, embed_dim]
        # We need: [B, num_spatial_tokens, embed_dim]

        if q.shape[1] != self.num_spatial_tokens:
            # Use adaptive pooling in sequence dimension
            q_transposed = q.transpose(1, 2)  # [B, embed_dim, num_query_tokens]
            q_pooled = F.adaptive_avg_pool1d(q_transposed, self.num_spatial_tokens)
            query = q_pooled.transpose(1, 2)  # [B, num_spatial_tokens, embed_dim]
        else:
            query = q

        query = self.query_adapter(query)
        residual = query
        query = self.layer_norm_q(query)

        # ============ KEY: Generate from original image ============
        key = self.key_conv(image)
        key = self.key_pool(key)  # [B, embed_dim, decoder_size, decoder_size]

        B_k, C_k, H_k, W_k = key.shape
        key = key.view(B_k, C_k, H_k * W_k).transpose(1, 2)  # [B, H*W, embed_dim]
        key = self.layer_norm_k(key)

        # ============ VALUE: Project from decoder feature map ============
        v_pooled = F.adaptive_avg_pool2d(v, (self.decoder_feature_size, self.decoder_feature_size))
        B_v, C_v, H_v, W_v = v_pooled.shape
        v_flat = v_pooled.view(B_v, C_v, H_v * W_v).transpose(1, 2)  # [B, H*W, C]
        value = self.value_projection(v_flat)  # [B, decoder_feature_size², embed_dim]
        value = self.layer_norm_v(value)

        # ============ SHORTCUT: Process based on layer type ============
        if self.first_layer and self.shortcut_projection is not None:
            B_s, C_s, H_s, W_s = shortcut.shape
            shortcut_pooled = F.adaptive_avg_pool2d(
                shortcut,
                (self.decoder_feature_size, self.decoder_feature_size)
            )
            B_s, C_s, H_sp, W_sp = shortcut_pooled.shape
            shortcut_flat = shortcut_pooled.view(B_s, C_s * H_sp, W_sp).transpose(1, 2)
            shortcut_processed = self.shortcut_projection(shortcut_flat)
        else:
            # Align shortcut tokens with spatial tokens
            if shortcut.shape[1] != self.num_spatial_tokens:
                shortcut_transposed = shortcut.transpose(1, 2)
                shortcut_pooled = F.adaptive_avg_pool1d(shortcut_transposed, self.num_spatial_tokens)
                shortcut_processed = shortcut_pooled.transpose(1, 2)
            else:
                shortcut_processed = shortcut

        shortcut_processed = self.layer_norm_shortcut(shortcut_processed)

        # ============ ATTENTION ============
        attn_output = self.attention(query, key, value, shortcut_processed)
        attn_output = self.dropout(attn_output)
        x = residual + attn_output
        x = self.layer_norm_1(x)

        # ============ MOE ============
        residual = x
        moe_output, aux_loss_dict = self.MoE(x)
        moe_output = self.dropout(moe_output)
        x = residual + moe_output
        x = self.layer_norm_2(x)

        return x, aux_loss_dict
