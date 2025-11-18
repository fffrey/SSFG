from networkx.classes.function import selfloop_edges
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import dgl
import dgl.nn as dglnn
from transformers.models.marian.modeling_marian import MarianSinusoidalPositionalEmbedding as SinusoidalPositionalEmbedding
from transformers.models.bart.modeling_bart import (
    shift_tokens_right,
    BartConfig,
    BartEncoder,
    BartEncoderLayer,
    BartPretrainedModel,
    _expand_mask, _make_causal_mask,
    BartLearnedPositionalEmbedding, 
    BartAttention,
    BartDecoder,
    BartDecoderLayer,
)
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput, Seq2SeqModelOutput,BaseModelOutputWithPastAndCrossAttentions
from utils.CONSTANT import DISCOURSE_RELATIONS

from transformers import BartConfig


name_2_activation_fn_mapping = {
    'tanh':F.tanh,
    'relu':F.relu,
    'gelu':F.gelu,
}

class GraphTransformerConfig(BartConfig):

    def __init__(
        self,
        backbone_model ='../facebook/bart-large',
        # backbone_model='../pretrained_model/bart_large',

        output_attentions = True,

        model_type='transformer',
        # all_bart_base config
        gt_activation_dropout = 0.1 ,
        gt_activation_function = 'gelu' ,
        gt_add_bias_logits = False ,
        gt_add_final_layer_norm = False ,
        gt_attention_dropout = 0.1 ,
        gt_d_model = 768 ,
        gt_decoder_attention_heads = 12 ,
        gt_decoder_ffn_dim = 3072 ,
        gt_decoder_layerdrop = 0.0 ,
        gt_dropout = 0.1 ,
        gt_encoder_attention_heads = 12 ,
        gt_encoder_ffn_dim = 3072 ,
        gt_encoder_layerdrop = 0.0 ,
        gt_encoder_layers = 6 ,
        gt_init_std = 0.02 ,
        gt_is_encoder_decoder = True ,
        gt_normalize_before = False ,
        gt_normalize_embedding = True ,
        gt_scale_embedding = False,
        conv_activation_fn = 'relu',
        num_beams = 5,
        rezero = 1,
        max_length = 100,
        min_length = 5,
        utt_pooling = 'average',
        gt_pos_embed = '',
        **kwargs,
    ):
        for k,v in kwargs.items():
            if not hasattr(self,k):
                setattr(self,k,v)
        
        pretrained_model_config = BartConfig.from_pretrained(backbone_model)
        for k,v in vars(pretrained_model_config).items():
            if not hasattr(self,k):
                setattr(self,k,v)
        self.gt_pos_embed = gt_pos_embed
        self.conv_activation_fn = conv_activation_fn
        self.utt_pooling = utt_pooling
        self.backbone_model =backbone_model
        self.model_type=model_type
        self.gt_activation_dropout =gt_activation_dropout
        self.gt_activation_function =gt_activation_function
        self.gt_add_bias_logits =gt_add_bias_logits
        self.gt_add_final_layer_norm =gt_add_final_layer_norm
        self.gt_attention_dropout =gt_attention_dropout
        self.gt_d_model =gt_d_model
        self.gt_decoder_attention_heads =gt_decoder_attention_heads
        self.gt_decoder_ffn_dim =gt_decoder_ffn_dim
        self.gt_decoder_layerdrop =gt_decoder_layerdrop
        
        self.gt_dropout =gt_dropout
        self.gt_encoder_attention_heads =gt_encoder_attention_heads
        self.gt_encoder_ffn_dim =gt_encoder_ffn_dim
        self.gt_encoder_layerdrop =gt_encoder_layerdrop
        self.gt_encoder_layers =gt_encoder_layers
        self.gt_init_std =gt_init_std
        self.gt_is_encoder_decoder =gt_is_encoder_decoder
        self.min_length = min_length
        self.gt_normalize_before = gt_normalize_before
        self.gt_normalize_embedding =gt_normalize_embedding
        self.gt_scale_embedding =gt_scale_embedding
        self.num_beams = num_beams
        self.max_length = max_length
        self.rezero = rezero

class GraphTransformerMultiHeadAttentionLayer(nn.Module):
  #.....


class GraphTransformerLayer(BartEncoderLayer):

    def __init__(self,config):
        super().__init__(config)
        self.self_attn = GraphTransformerMultiHeadAttentionLayer(config)
        self.fc1 = nn.Linear(self.embed_dim, config.gt_encoder_ffn_dim)
        self.fc2 = nn.Linear(config.gt_encoder_ffn_dim, self.embed_dim)
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
        **kwargs,
        ):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (:obj:`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """
        adj_mats = kwargs.get('adj_mats',None)
        residual = hidden_states
        hidden_states, attn_weights = self.self_attn(
            hidden_states,hidden_states,hidden_states,
            adj_mats,
            mask=attention_mask,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs

class GraphTransformer(BartEncoder):

    def __init__(self,config):
        super().__init__(config)
        self.layers = nn.ModuleList([GraphTransformerLayer(config) for _ in range(int(config.gt_encoder_layers))])
        del self.embed_positions

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        bs,num_utt = input_shape
        if hasattr(self,'embed_positions'):
            if isinstance(self.embed_positions,SinusoidalPositionalEmbedding):
                embed_pos = self.embed_positions(input_shape)
            else:
                embed_pos = self.embed_positions(torch.arange(num_utt).view(1,-1).repeat(bs,1).to(inputs_embeds.device))
            hidden_states = inputs_embeds + embed_pos
        else:
            hidden_states = inputs_embeds

        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask =  (attention_mask == 1).unsqueeze(1).unsqueeze(2)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                if getattr(self.config, "gradient_checkpointing", False) and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                        #dis_embed = self.dis_embed,
                        #speaker_embed = self.speaker_embed,
                        **kwargs,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )

