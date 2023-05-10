# https://github.com/IlyaGusev/rulm/blob/master/self_instruct/scripts/train.py
from transformers import (AutoTokenizer, 
                          AutoModelForCausalLM, 
                          LlamaTokenizer, 
                          CanineModel,
                          PreTrainedTokenizer)
from datasets import load_dataset
from typing import Optional, List, Union, Tuple
import torch

from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
import transformers.models.canine.modeling_canine as canine
from transformers.models.llama.modeling_llama import LlamaPreTrainedModel, LlamaDecoderLayer, LlamaRMSNorm, LlamaRotaryEmbedding
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.canine.modeling_canine import CanineEncoder, CharactersToMolecules, ConvProjection, CaninePooler
from transformers.models.canine.configuration_canine import CanineConfig
from transformers.models.canine.tokenization_canine import CanineTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.tokenization_utils import AddedToken
import copy
import math
from torch import nn
from torch.nn import CrossEntropyLoss


class LlamaWithCanineConfig(LlamaConfig):
    def __init__(self,
                 num_hash_buckets: int = 16384,
                 num_hash_functions: int = 8,
                 type_vocab_size: int = 16,
                 layer_norm_eps: float = 1e-12,
                 hidden_dropout_prob: float = 0.1,
                 local_transformer_stride: int = 128,
                 downsampling_rate: int = 4,
                 attention_probs_dropout_prob: float = 0.1,
                 upsampling_kernel_size: int = 4,
                 max_position_embeddings_canine: int = 16384,
                 **kwargs):

        self.num_hash_buckets = num_hash_buckets
        self.num_hash_functions = num_hash_functions
        self.type_vocab_size = type_vocab_size
        self.layer_norm_eps = layer_norm_eps
        self.hidden_dropout_prob = hidden_dropout_prob
        self.local_transformer_stride = local_transformer_stride
        self.downsampling_rate = downsampling_rate
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.upsampling_kernel_size = upsampling_kernel_size
        self.max_position_embeddings_canine = max_position_embeddings_canine

        super().__init__(**kwargs)


def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaLocalAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, 
                 config: LlamaConfig, 
                 attend_from_chunk_width: int = 128,
                 attend_from_chunk_stride: int = 128,
                 attend_to_chunk_width: int = 128,
                 attend_to_chunk_stride: int = 128):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)
        
        self.attend_from_chunk_width = attend_from_chunk_width
        self.attend_from_chunk_stride = attend_from_chunk_stride
        self.attend_to_chunk_width = attend_to_chunk_width
        self.attend_to_chunk_stride = attend_to_chunk_stride

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _get_attention_outputs(self,
                               from_tensor: torch.Tensor,
                               to_tensor: torch.Tensor,
                               attention_mask: Optional[torch.Tensor] = None,
                               position_ids: Optional[torch.LongTensor] = None):
        bsz, q_len, _ = to_tensor.size()

        query_states = self.q_proj(from_tensor).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(to_tensor).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(to_tensor).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # [bsz, nh, t, hd]

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        return attn_output, attn_weights


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        from_seq_length = to_seq_length = hidden_states.shape[1]
        from_tensor = to_tensor = hidden_states

        # Create chunks (windows) that we will attend *from* and then concatenate them.
        from_chunks = []
        from_start = 0
        for chunk_start in range(from_start, from_seq_length, self.attend_from_chunk_stride):
            chunk_end = min(from_seq_length, chunk_start + self.attend_from_chunk_width)
            from_chunks.append((chunk_start, chunk_end))

        # Determine the chunks (windows) that will will attend *to*.
        to_chunks = []
        for chunk_start in range(0, to_seq_length, self.attend_to_chunk_stride):
            chunk_end = min(to_seq_length, chunk_start + self.attend_to_chunk_width)
            to_chunks.append((chunk_start, chunk_end))

        if len(from_chunks) != len(to_chunks):
            raise ValueError(
                f"Expected to have same number of `from_chunks` ({from_chunks}) and "
                f"`to_chunks` ({from_chunks}). Check strides."
            )

        # next, compute attention scores for each pair of windows and concatenate
        attention_output_chunks = []
        attention_probs_chunks = []
        for (from_start, from_end), (to_start, to_end) in zip(from_chunks, to_chunks):
            from_tensor_chunk = from_tensor[:, from_start:from_end, :]
            to_tensor_chunk = to_tensor[:, to_start:to_end, :]
            # `attention_mask`: <float>[batch_size, from_seq, to_seq]
            # `attention_mask_chunk`: <float>[batch_size, from_seq_chunk, to_seq_chunk]
            attention_mask_chunk = attention_mask[:, :, from_start:from_end, to_start:to_end]

            device = hidden_states.device
            position_ids_chunk = torch.arange(
                0, from_end - from_start, dtype=torch.long, device=device
            ).unsqueeze(0)

            attention_outputs_chunk = self._get_attention_outputs(from_tensor_chunk,
                                                                  to_tensor_chunk,
                                                                  attention_mask_chunk,
                                                                  position_ids_chunk)


            attention_output_chunks.append(attention_outputs_chunk[0])
            if output_attentions:
                attention_probs_chunks.append(attention_outputs_chunk[1])

        attention_output = torch.cat(attention_output_chunks, dim=1)
        attention_output = self.o_proj(attention_output)
        
        return attention_output, tuple(attention_probs_chunks) if output_attentions else None, None

    
class LlamaModelWithCanine(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]
    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig, add_pooling_layer=True):
        super().__init__(config)
        self.padding_idx = config.pad_token_id

        canine_config = copy.deepcopy(config)
        canine_config.max_position_embeddings = canine_config.max_position_embeddings_canine

        self.char_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        self.initial_char_encoder = LlamaLocalAttention(canine_config)
        self.chars_to_molecules = CharactersToMolecules(canine_config)

        self.projection = ConvProjection(canine_config)
        # shallow/low-dim transformer encoder to get a final character encoding
        self.final_char_encoder = LlamaDecoderLayer(canine_config)
 
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )

            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def _create_3d_attention_mask_from_input_mask(self, from_tensor, to_mask):
        """
        Create 3D attention mask from a 2D tensor mask.
        Args:
            from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
            to_mask: int32 Tensor of shape [batch_size, to_seq_length].
        Returns:
            float Tensor of shape [batch_size, from_seq_length, to_seq_length].
        """
        batch_size, from_seq_length = from_tensor.shape[0], from_tensor.shape[1]

        to_seq_length = to_mask.shape[1]

        to_mask = torch.reshape(to_mask, (batch_size, 1, to_seq_length)).float()

        # We don't assume that `from_tensor` is a mask (although it could be). We
        # don't actually care if we attend *from* padding tokens (only *to* padding)
        # tokens so we create a tensor of all ones.
        broadcast_ones = torch.ones(size=(batch_size, from_seq_length, 1), dtype=torch.float32, device=to_mask.device)

        # Here we broadcast along two dimensions to create the mask.
        mask = broadcast_ones * to_mask

        return mask

    def _downsample_attention_mask(self, char_attention_mask: torch.Tensor, downsampling_rate: int):
        """Downsample 2D character attention mask to 2D molecule attention mask using MaxPool1d layer."""

        # first, make char_attention_mask 3D by adding a channel dim
        batch_size, char_seq_len = char_attention_mask.shape
        poolable_char_mask = torch.reshape(char_attention_mask, (batch_size, 1, char_seq_len))

        # next, apply MaxPool1d to get pooled_molecule_mask of shape (batch_size, 1, mol_seq_len)
        pooled_molecule_mask = torch.nn.MaxPool1d(kernel_size=downsampling_rate, stride=downsampling_rate)(
            poolable_char_mask.float()
        )
        
        # finally, squeeze to get tensor of shape (batch_size, mol_seq_len)
        molecule_attention_mask = torch.squeeze(pooled_molecule_mask, dim=-1)

        return molecule_attention_mask

    def _repeat_molecules(self, molecules: torch.Tensor, char_seq_length: torch.Tensor) -> torch.Tensor:
        """Repeats molecules to make them the same length as the char sequence."""

        rate = self.config.downsampling_rate

        molecules_without_extra_cls = molecules[:, 1:, :]
        # `repeated`: [batch_size, almost_char_seq_len, molecule_hidden_size]
        repeated = torch.repeat_interleave(molecules_without_extra_cls, repeats=rate, dim=-2)

        # So far, we've repeated the elements sufficient for any `char_seq_length`
        # that's a multiple of `downsampling_rate`. Now we account for the last
        # n elements (n < `downsampling_rate`), i.e. the remainder of floor
        # division. We do this by repeating the last molecule a few extra times.
        last_molecule = molecules[:, -1:, :]
        remainder_length = torch.fmod(torch.tensor(char_seq_length), torch.tensor(rate)).item()
        remainder_repeated = torch.repeat_interleave(
            last_molecule,
            # +1 molecule to compensate for truncation.
            repeats=remainder_length + rate,
            dim=-2,
        )

        # `repeated`: [batch_size, char_seq_len, molecule_hidden_size]
        return torch.cat([repeated, remainder_repeated], dim=-2)
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        assert past_key_values is None
        assert not use_cache
        assert position_ids is None
        assert inputs_embeds is None
        # use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()
        
        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)
        
        # if past_key_values is not None:
        #     past_key_values_length = past_key_values[0][0].shape[2]
        #     seq_length_with_past = seq_length_with_past + past_key_values_length

            
        ################################## CANINE ENCODER INJECTED #####################################
        
        input_shape = input_ids.size()
        
        # extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)
        # print("Extended attention mask: ", extended_attention_mask)
    
        # `input_char_embeddings`: shape (batch_size, char_seq, char_dim)
        input_char_embeddings = self.char_embeddings(input_ids)
        
        
        extended_attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, input_char_embeddings.shape[1]), input_char_embeddings, past_key_values_length
        )
        
        molecule_attention_mask = self._downsample_attention_mask(
            attention_mask, downsampling_rate=self.config.downsampling_rate
        ).squeeze(1)
        
        molecule_position_ids = torch.arange(0, molecule_attention_mask.shape[-1], device=device).unsqueeze(0)
        
        

        # Contextualize character embeddings using shallow Transformer.
        # We use a 3D attention mask for the local attention.
        # `input_char_encoding`: shape (batch_size, char_seq_len, char_dim)
        # char_attention_mask = self._create_3d_attention_mask_from_input_mask(input_ids, attention_mask)
        
        init_chars_encoder_outputs = self.initial_char_encoder(
            input_char_embeddings,
            attention_mask=extended_attention_mask,
            past_key_value=None,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        
        # init_chars_encoder_outputs = self.initial_char_encoder(
        #     input_char_embeddings,
        #     attention_mask=char_attention_mask,
        #     output_attentions=output_attentions,
        #     # output_hidden_states=output_hidden_states,
        # )
        input_char_encoding = init_chars_encoder_outputs[0]

        # Downsample chars to molecules.
        # The following lines have dimensions: [batch, molecule_seq, molecule_dim].
        # In this transformation, we change the dimensionality from `char_dim` to
        # `molecule_dim`, but do *NOT* add a resnet connection. Instead, we rely on
        # the resnet connections (a) from the final char transformer stack back into
        # the original char transformer stack and (b) the resnet connections from
        # the final char transformer stack back into the deep BERT stack of
        # molecules.
        #
        # Empirically, it is critical to use a powerful enough transformation here:
        # mean pooling causes training to diverge with huge gradient norms in this
        # region of the model; using a convolution here resolves this issue. From
        # this, it seems that molecules and characters require a very different
        # feature space; intuitively, this makes sense.
        inputs_embeds = self.chars_to_molecules(input_char_encoding)
        
        ######################################### CANINE ENCODER END #############################################
        
        extended_molecule_attention_mask = self._prepare_decoder_attention_mask(
            molecule_attention_mask, (batch_size, inputs_embeds.shape[1]), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    extended_molecule_attention_mask,
                    molecule_position_ids,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=extended_molecule_attention_mask,
                    position_ids=molecule_position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        
        
        ################################## CANINE DECODER INJECTED #####################################

        # Upsample molecules back to characters.
        # `repeated_molecules`: shape (batch_size, char_seq_len, mol_hidden_size)
        repeated_molecules = self._repeat_molecules(hidden_states, char_seq_length=input_shape[-1])

        # Concatenate representations (contextualized char embeddings and repeated molecules):
        # `concat`: shape [batch_size, char_seq_len, molecule_hidden_size+char_hidden_final]
        concat = torch.cat([input_char_encoding, repeated_molecules], dim=-1)

        # Project representation dimension back to hidden_size
        # `sequence_output`: shape (batch_size, char_seq_len, hidden_size])
        sequence_output = self.projection(concat)

        # Apply final shallow Transformer
        # `sequence_output`: shape (batch_size, char_seq_len, hidden_size])
        final_chars_encoder_outputs = self.final_char_encoder(
            sequence_output,
            attention_mask=extended_attention_mask,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        sequence_output = final_chars_encoder_outputs[0]
        hidden_states = sequence_output
        
        ################################## CANINE DECODER END ############################################


        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class LlamaWithCanineForCausalLM(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModelWithCanine(config)

        self.lm_head_renamed = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head_renamed

    def set_output_embeddings(self, new_embeddings):
        self.lm_head_renamed = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        Returns:
        Example:
        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM
        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)
        >>> prompt = "Hey, are you consciours? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")
        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you consciours? Can you talk to me?\nI'm not consciours, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head_renamed(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)

            print(shift_logits)
            print(shift_labels)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past

    
class ByT5Tokenizer(PreTrainedTokenizer):
    """
    Construct a ByT5 tokenizer. ByT5 simply uses raw bytes utf-8 encoding.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the end of sequence.
            The token used is the `sep_token`.

            </Tip>

        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        extra_ids (`int`, *optional*, defaults to 100):
            Add a number of extra ids added to the end of the vocabulary for use as sentinels. These tokens are
            accessible as "<extra_id_{%d}>" where "{%d}" is a number between 0 and extra_ids-1. Extra tokens are
            indexed from the end of the vocabulary up to beginning ("<extra_id_0>" is the last token in the vocabulary
            like in ByT5 preprocessing see
            [here](https://github.com/google-research/text-to-text-transfer-transformer/blob/9fd7b14a769417be33bc6c850f9598764913c833/t5/data/preprocessors.py#L2117)).
        additional_special_tokens (`List[str]`, *optional*):
            Additional special tokens used by the tokenizer.
    """

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        extra_ids=125,
        additional_special_tokens=None,
        **kwargs,
    ) -> None:
        # Add extra_ids to the special token list
        if extra_ids > 0 and additional_special_tokens is None:
            additional_special_tokens = [f"<extra_id_{i}>" for i in range(extra_ids)]
        elif extra_ids > 0 and additional_special_tokens is not None:
            # Check that we have the right number of extra_id special tokens
            extra_tokens = len(set(filter(lambda x: bool("extra_id" in str(x)), additional_special_tokens)))
            if extra_tokens != extra_ids:
                raise ValueError(
                    f"Both extra_ids ({extra_ids}) and additional_special_tokens ({additional_special_tokens}) are"
                    " provided to ByT5Tokenizer. In this case the additional_special_tokens must include the"
                    " extra_ids tokens"
                )

        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token

        super().__init__(
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            extra_ids=extra_ids,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

        self._extra_ids = extra_ids

        self._utf_vocab_size = 2**8  # utf is 8 bits

        # define special tokens dict
        self.special_tokens_encoder: Dict[int, str] = {
            self.pad_token: 0,
            self.eos_token: 1,
            self.unk_token: 2,
        }
        self._num_special_tokens = len(self.special_tokens_encoder)
        n = len(additional_special_tokens)
        for i, token in enumerate(additional_special_tokens):
            self.special_tokens_encoder[token] = self.vocab_size + i - n
        self.special_tokens_decoder: Dict[str, int] = {v: k for k, v in self.special_tokens_encoder.items()}

    @property
    def vocab_size(self):
        return self._utf_vocab_size + self._num_special_tokens + self._extra_ids

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # normal case: some special tokens
        if token_ids_1 is None:
            return ([0] * len(token_ids_0)) + [1]
        return ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]

    def _add_eos_if_not_present(self, token_ids: List[int]) -> List[int]:
        """Do not add eos again if user already added it."""
        if len(token_ids) > 0 and token_ids[-1] == self.eos_token_id:
            warnings.warn(
                f"This sequence already has {self.eos_token}. In future versions this behavior may lead to duplicated"
                " eos tokens being added."
            )
            return token_ids
        else:
            return token_ids + [self.eos_token_id]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. ByT5 does not
        make use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of zeros.
        """
        eos = [self.eos_token_id]

        if token_ids_1 is None:
            return len(token_ids_0 + eos) * [0]
        return len(token_ids_0 + eos + token_ids_1 + eos) * [0]

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A sequence has the following format:

        - single sequence: `X </s>`
        - pair of sequences: `A </s> B </s>`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        token_ids_0 = self._add_eos_if_not_present(token_ids_0)
        if token_ids_1 is None:
            return token_ids_0
        else:
            token_ids_1 = self._add_eos_if_not_present(token_ids_1)
            return token_ids_0 + token_ids_1

    def _tokenize(self, text: str) -> List[str]:
        """Take as input a string and return a list of strings (tokens) for words/sub-words"""
        tokens = [chr(i) for i in text.encode("utf-8")]
        return tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        if token in self.special_tokens_encoder:
            token_id = self.special_tokens_encoder[token]
        elif token in self.added_tokens_encoder:
            token_id = self.added_tokens_encoder[token]
        elif len(token) != 1:
            token_id = self.unk_token_id
        else:
            token_id = ord(token) + self._num_special_tokens
        return token_id

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if index in self.special_tokens_decoder:
            token = self.special_tokens_decoder[index]
        else:
            token = chr(index - self._num_special_tokens)
        return token

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        bstring = b""
        for token in tokens:
            if token in self.special_tokens_decoder:
                tok_string = self.special_tokens_decoder[token].encode("utf-8")
            elif token in self.added_tokens_decoder:
                tok_string = self.special_tokens_decoder[token].encode("utf-8")
            elif token in self.special_tokens_encoder:
                tok_string = token.encode("utf-8")
            elif token in self.added_tokens_encoder:
                tok_string = token.encode("utf-8")
            else:
                tok_string = bytes([ord(token)])
            bstring += tok_string
        string = bstring.decode("utf-8", errors="ignore")
        return string

    # ByT5Tokenizer has no vocab file
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        return ()