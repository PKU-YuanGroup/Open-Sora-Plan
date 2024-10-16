from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from mindspeed_mm.models.vision.vision_encoders.internvit_model import InternRMSNorm, InternVitSelfAttention, InternVitTransformerLayer


def get_language_layer_spec() -> ModuleSpec:
    mlp = get_mlp_layer_spec()
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=InternRMSNorm,
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=ColumnParallelLinear,
                    core_attention=DotProductAttention,
                    linear_proj=RowParallelLinear,
                    q_layernorm=IdentityOp,
                    k_layernorm=IdentityOp
                )
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_cross_attn_layernorm=IdentityOp,
            pre_mlp_layernorm=InternRMSNorm,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
            sharded_state_dict_keys_map={
                'input_layernorm.': 'self_attention.linear_qkv_layer_norm_',
                'pre_mlp_layernorm.': 'mlp.linear_fc1.layer_norm_'
            }
        )
    )


def get_vit_layer_spec(config) -> ModuleSpec:
    mlp = get_mlp_layer_spec()
    return ModuleSpec(
        module=InternVitTransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=InternRMSNorm,
            self_attention=ModuleSpec(
                module=InternVitSelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=ColumnParallelLinear,
                    core_attention=DotProductAttention,
                    linear_proj=RowParallelLinear,
                    q_layernorm=InternRMSNorm if config.qk_layernorm else IdentityOp,
                    k_layernorm=InternRMSNorm if config.qk_layernorm else IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=InternRMSNorm,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
        )
    )


def get_mlp_layer_spec():
    return ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(
            linear_fc1=ColumnParallelLinear,
            linear_fc2=RowParallelLinear,
        )
    )