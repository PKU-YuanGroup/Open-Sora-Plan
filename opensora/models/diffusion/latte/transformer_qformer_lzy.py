
import torch
import torch.nn as nn

# from diffusers.models.attention import FeedForward
from .modules import Attention, FeedForward

class BasicQFormerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_attention_heads=16,
        attention_head_dim=64,
        dropout=0.0,
        attention_bias=False,
        upcast_attention=False,
        attention_out_bias=True,
        norm_elementwise_affine=True,
        norm_eps=1e-5,
        activation_fn="gelu",
        attention_mode="xformers", 
        use_rope=False,
        rope_scaling=None,
        final_dropout=False,
        ff_inner_dim=None,
        ff_bias=False,
        compress_kv_factor=None,
        only_visual_attention=True,
    ):
        super().__init__()

        # self attn
        self.norm1_latents = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
        self.norm1_visual_context = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=dim,
            upcast_attention=upcast_attention,
            attention_mode=attention_mode,
            use_rope=use_rope, 
            rope_scaling=rope_scaling, 
            out_bias=attention_out_bias,
            compress_kv_factor=compress_kv_factor, 
        )

        self.only_visual_attention = only_visual_attention
        if not only_visual_attention:
            # cross attn if cross_attention_dim is not None, otherwise self attn
            self.norm2_latents = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
            self.norm2_encoder_hidden_states = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
                attention_mode=attention_mode,  
                use_rope=False,  
                compress_kv_factor=None,
            ) 
        else:
            self.norm2_latents = None
            self.norm2_encoder_hidden_states = None
            self.attn2 = None

        self.norm3 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            # inner_dim=ff_inner_dim,
            # bias=ff_bias,
        )


    def forward(
        self, 
        latents, 
        visual_context, 
        attention_mask=None,
        encoder_hidden_states=None, 
        encoder_attention_mask=None,
        position_q=None,
        position_k=None,
        last_shape=None,
    ):
        '''
            params:
                latents: queries of q former 
                    shape (b, n1, c)
                visual_context: visual features
                    shape (b, n2, c)
                encoder_hidden_states: text features or others
                    shape (b, n3, c)

            return:
                latents: updated queries
        '''
        # self attn
        norm_latents = self.norm1_latents(latents)
        norm_visual_context = self.norm1_visual_context(visual_context)
        kv_input = torch.cat([norm_visual_context, norm_latents], dim=-2)

        attn_output = self.attn1(
            norm_latents,
            encoder_hidden_states=kv_input,
            attention_mask=attention_mask,
            position_q=position_q,
            position_k=position_k,
            last_shape=last_shape,                                                     
        )

        latents = latents + attn_output

        if not self.only_visual_attention:
            # cross attn
            norm_latents = self.norm2_latents(latents)
            if encoder_hidden_states is not None:
                norm_encoder_hidden_states = self.norm2_encoder_hidden_states(encoder_hidden_states)
            else:
                norm_encoder_hidden_states = self.norm2_encoder_hidden_states(visual_context)
            
            kv_input = torch.cat([norm_encoder_hidden_states, norm_latents], dim=-2)
            attn_output = self.attn2(
                norm_latents,
                encoder_hidden_states=kv_input,
                attention_mask=encoder_attention_mask,
                position_q=None,
                position_k=None,
                last_shape=None,
            )

            latents = latents + attn_output

        # feed forward
        norm_latents = self.norm3(latents)
        ff_output = self.ff(norm_latents)

        latents = latents + ff_output

        return latents



class QFormer(nn.Module):
    def __init__(
        self,
        dim=1024,
        num_queries=1,
        visual_context_dim=768,
        out_dim=1024,
        block_num=6,
        num_attention_heads=16,
        attention_head_dim=64,
        dropout=0.0,
        only_visual_attention=True,
        encoder_hidden_states_dim=None,
        use_rope=False,
        rope_scaling=None,
        attention_mode="xformers",
        compress_kv_factor=1,
    ):
        super().__init__()

        self.only_visual_attention = only_visual_attention

        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim ** 0.5)

        self.proj_in_visual_context = nn.Linear(visual_context_dim, dim)
        if not only_visual_attention:
            self.proj_in_encoder_hidden_states = nn.Linear(encoder_hidden_states_dim, dim)

        self.proj_out = nn.Linear(dim, out_dim)

        self.layers = nn.ModuleList([])

        for d in range(block_num):
            self.layers.append(
                BasicQFormerBlock(
                    dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    dropout=dropout,
                    only_visual_attention=only_visual_attention,
                    attention_mode=attention_mode, 
                    use_rope=use_rope, 
                    rope_scaling=rope_scaling, 
                    compress_kv_factor=(compress_kv_factor, compress_kv_factor) if d >= block_num // 2 and compress_kv_factor != 1 else None, # follow pixart-sigma, apply in second-half layers
                )
            )
            
        self.norm_out = nn.LayerNorm(out_dim)

    def forward(self, visual_context, encoder_hidden_states=None):

        b = visual_context.shape[0]
        latents = self.latents.repeat(b, 1, 1)

        visual_context = self.proj_in_visual_context(visual_context)
        if not self.only_visual_attention:
            encoder_hidden_states = self.proj_in_encoder_hidden_states(encoder_hidden_states)

        for layer in self.layers:
            latents = layer(latents, visual_context, encoder_hidden_states)
        
        latents = self.proj_out(latents)
        latents = self.norm_out(latents)

        return latents

if __name__ == "__main__":
    qformer = QFormer(
        dim=1024,
        num_queries=256,
        visual_context_dim=768,
        encoder_hidden_states_dim=768,
        out_dim=1024,
        block_num=8,
        num_attention_heads=64,
        attention_head_dim=16,
        dropout=0.0,
        max_seq_length=257,
        apply_pos_emb=False,
    )

    print(qformer)

    visual_context = torch.randn(2, 4096, 768)
    encoder_hidden_states = torch.randn(2, 4096, 768)

    output = qformer(visual_context, encoder_hidden_states)

    num_params = sum(p.numel() for p in qformer.parameters() if p.requires_grad)
    print("Number of trainable parameters: ", num_params / 1e6, "M")