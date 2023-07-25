"""A more detailed implementation of recurrent stuff, extending the simple recurrent block implementation in scriptable_bert.py"""


from random import randrange
from typing import Optional

import torch

from .components import (
    AttentionComponent,
    EmbeddingComponent,
    FFNComponent,
    PredictionHeadComponent,
    _get_nonlin_fn,
    _get_norm_fn,
    get_extended_attention_mask,
    get_layer_fn,
)
from .scriptable_bert import ScriptableLMForSequenceClassification, _init_module

# import torchdynamo


INPLACE = False


def construct_scriptable_recurrent(cfg_arch, vocab_size, downstream_classes=None):
    """See the config file for details on what is possible."""
    cfg_arch.embedding.vocab_size = vocab_size
    cfg_arch.num_labels = downstream_classes

    if downstream_classes is None:
        model = BPTTforPreTraining(ScriptableRecurrentLM(cfg_arch), cfg_arch)
    else:
        model = ScriptableLMForSequenceClassification(ScriptableRecurrentLM(cfg_arch), cfg_arch)
    return model


"""This is the simplified version that should be the default for all models later..."""


class TransformerLayerSimplified(torch.nn.Module):
    """A transformer-encoder structure based on the components from above."""

    def __init__(self, idx, cfg_arch):
        super().__init__()
        self.dropout = torch.nn.Dropout(cfg_arch.hidden_dropout_prob, inplace=INPLACE)
        self.norm1 = _get_norm_fn(cfg_arch.norm)(cfg_arch.hidden_size, eps=cfg_arch.norm_eps)
        self.norm2 = _get_norm_fn(cfg_arch.norm)(cfg_arch.hidden_size, eps=cfg_arch.norm_eps)
        self.attn = AttentionComponent(
            idx,
            cfg_arch.hidden_size,
            cfg_arch.attention,
            cfg_arch.use_bias,
        )
        nonlin_fn = _get_nonlin_fn(cfg_arch.nonlin)
        if (idx + 1) % cfg_arch.ffn_layer_frequency == 0:
            self.ffn = FFNComponent(
                cfg_arch.hidden_size,
                cfg_arch.intermed_size,
                nonlin_fn,
                cfg_arch.use_bias,
            )
        else:
            self.ffn = torch.nn.Identity()

        self.norm_scheme = cfg_arch.norm_scheme
        self.fn_training, self.fn_eval = get_layer_fn(
            self.norm_scheme, prob=cfg_arch.hidden_dropout_prob, scripting=cfg_arch.layer_fusion, dn=False, drop=False
        )
        self.register_buffer("alpha", torch.tensor(1.0))
        self.LAYOUT = self.attn.LAYOUT

    def forward(self, states, attention_mask: Optional[torch.Tensor] = None):
        if self.norm_scheme == "pre":  # On Layer Normalization in the Transformer Architecture
            if self.training:
                states = self.fn_training(states, self.attn(self.norm1(states), attention_mask), self.alpha, self.alpha)
                states = self.fn_training(states, self.ffn(self.norm2(states)), self.alpha, self.alpha)
            else:
                states = self.fn_eval(states, self.attn(self.norm1(states), attention_mask), self.alpha, self.alpha)
                states = self.fn_eval(states, self.ffn(self.norm2(states)), self.alpha, self.alpha)
        else:
            if self.training:
                states = self.norm1(self.fn_training(states, self.attn(states, attention_mask), self.alpha, self.alpha))
                states = self.norm2(self.fn_training(states, self.ffn(states), self.alpha, self.alpha))
            else:
                states = self.norm1(self.fn_eval(states, self.attn(states, attention_mask), self.alpha, self.alpha))
                states = self.norm2(self.fn_eval(states, self.ffn(states), self.alpha, self.alpha))

        return states


class ScriptableRecurrentLM(torch.nn.Module):
    """Definitely can represent BERT, but also a lot of other things. To be used for MLM schemes."""

    def __init__(self, cfg_arch):
        super().__init__()
        self.cfg = cfg_arch

        self.embedding = EmbeddingComponent(cfg_arch.embedding, cfg_arch.norm, cfg_arch.norm_eps)
        if cfg_arch.embedding.embedding_dim == cfg_arch.hidden_size:
            self.input_projection = torch.nn.Identity()
        else:
            self.input_projection = torch.nn.Linear(
                cfg_arch.embedding.embedding_dim,
                cfg_arch.hidden_size,
                bias=cfg_arch.use_bias,
            )

        self.recurrent_layer = SequentialwithMask([TransformerLayerSimplified(idx, cfg_arch) for idx in range(cfg_arch.recurrent_layers)])

        self.seq_first = self.recurrent_layer.LAYOUT == "[S B H]" if len(self.recurrent_layer) > 0 else False

    def forward(self, input_ids, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None):
        if attention_mask is not None:
            attention_mask = get_extended_attention_mask(attention_mask, input_ids.shape, self.cfg.attention.causal_attention)

        hidden_states = self.forward_embed(input_ids)
        # Main transformer blocks::
        for i in range(self.cfg.maximal_recurrence):
            hidden_states = self.forward_step(hidden_states, attention_mask)
        hidden_states = self.exit(hidden_states)
        return hidden_states

    def forward_embed(self, input_ids):
        hidden_states = self.input_projection(self.embedding(input_ids))

        if self.seq_first:
            hidden_states = hidden_states.transpose(0, 1).contiguous()
        return hidden_states

    def exit(self, hidden_states):
        if self.seq_first:
            hidden_states = hidden_states.transpose(0, 1).contiguous()
        return hidden_states

    def forward_step(self, hidden_states, attention_mask: Optional[torch.Tensor] = None):
        """Take another step forward on a set of given hidden states."""
        hidden_states = self.recurrent_layer(hidden_states, attention_mask)
        return hidden_states


class BPTTforPreTraining(torch.nn.Module):
    """Modified pretraining for depth-recurrent models. Only works with models that expose the ScriptableRecurrentLM interface."""

    def __init__(self, encoder, cfg_arch):
        super().__init__()
        self.cfg = cfg_arch

        self.encoder = encoder
        if not cfg_arch.skip_head_transform:
            self.prediction_head = PredictionHeadComponent(cfg_arch)
        else:
            self.prediction_head = torch.nn.Linear(
                cfg_arch.hidden_size,
                cfg_arch.embedding.embedding_dim,
                bias=cfg_arch.use_bias,
            )

        if self.cfg.tie_weights:
            self.decoder = torch.nn.Linear(cfg_arch.embedding.embedding_dim, cfg_arch.embedding.vocab_size, bias=cfg_arch.decoder_bias)
            self.decoder.weight = self.encoder.embedding.word_embedding.weight
        else:
            self.decoder = torch.nn.Linear(cfg_arch.hidden_size, cfg_arch.embedding.vocab_size, bias=cfg_arch.decoder_bias)

        self.loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
        self.sparse_prediction = self.cfg.sparse_prediction
        self.vocab_size = cfg_arch.embedding.vocab_size

        for name, module in self.named_modules():
            _init_module(
                name,
                module,
                self.cfg.init.type,
                self.cfg.init.std,
                self.cfg.hidden_size,
                self.cfg.recurrent_layers,
            )

        if cfg_arch.training_scheme == "bptt-deepthinking":
            self._forward_method = self._forward_deepthinking
        elif cfg_arch.training_scheme == "fixed-recurrence":
            self._forward_method = self._forward_fixed
        else:
            raise ValueError(f"Invalid training scheme {cfg_arch.training_scheme} given.")

    def forward(self, input_ids, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None):
        return self._forward_method(input_ids, attention_mask, labels)

    def _forward_token_exit(self, input_ids, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None):
        """Requires the encoder to be a ScriptableRecurrentLM. Requires labels!"""
        if attention_mask is not None:
            attention_mask = get_extended_attention_mask(attention_mask, input_ids.shape, self.cfg.attention.causal_attention)

        if self.encoder.seq_first:
            seq_first_labels = labels.transpose(0, 1).contiguous()

        hidden_states = self.forward_embed(input_ids)
        # Main transformer blocks::
        total_loss = 0

        for i in range(self.cfg.maximal_recurrence):
            hidden_states = self.forward_step(hidden_states, attention_mask)
            early_exit_states = hidden_states.view(-1, outputs.shape[-1])
            masked_lm_loss_per_token = self.loss_fn(early_exit_states, seq_first_labels.view(-1))

        return dict(loss=masked_lm_loss)

    def _forward_deepthinking(self, input_ids, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None):
        """Requires the encoder to be a ScriptableRecurrentLM. Requires labels!"""
        if attention_mask is not None:
            attention_mask = get_extended_attention_mask(attention_mask, input_ids.shape, self.cfg.attention.causal_attention)

        # Draw n, k dt steps
        n = randrange(0, self.cfg.maximal_recurrence)  # get features from n iterations to use as input
        k = randrange(
            1, min(self.cfg.maximal_recurrence - n + 1, self.cfg.maximal_recurrence // 2)
        )  # do k iterations using intermediate features as input
        # print(n, k)
        input_states = self.encoder.forward_embed(input_ids)
        hidden_states = input_states.detach()
        # First n steps:
        with torch.no_grad():
            for _ in range(n):
                hidden_states = input_states.detach() + self.encoder.forward_step(hidden_states, attention_mask)
        # Next k steps:
        for _ in range(k):
            hidden_states = input_states + self.encoder.forward_step(hidden_states, attention_mask)

        outputs = self.encoder.exit(hidden_states).view(-1, hidden_states.shape[-1])

        if self.sparse_prediction:
            masked_lm_loss = self._prediction_dynamic(outputs, labels)
        else:
            masked_lm_loss = self._prediction_fixed(outputs, labels)

        return dict(loss=masked_lm_loss)

    def _forward_fixed(self, input_ids, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None):
        """Requires the encoder to be a ScriptableRecurrentLM. Requires labels!"""
        if attention_mask is not None:
            attention_mask = get_extended_attention_mask(attention_mask, input_ids.shape, self.cfg.attention.causal_attention)

        outputs = self.encoder(input_ids, attention_mask)

        if self.sparse_prediction:
            masked_lm_loss = self._prediction_dynamic(outputs, labels)
        else:
            masked_lm_loss = self._prediction_fixed(outputs, labels)

        return dict(loss=masked_lm_loss)

    # Sparse prediction can have an unpredictable number of entries in each batch
    # depending on how MLM is running
    # for this reason, the code has to fall back to eager mode there
    # @torchdynamo.disable
    def _prediction_dynamic(self, outputs: torch.Tensor, labels: Optional[torch.Tensor] = None):
        if labels is not None:
            labels = labels.view(-1)
            mask_positions = labels.view(-1) != self.loss_fn.ignore_index
            outputs = outputs[mask_positions]
            labels = labels[mask_positions]

        outputs = self.decoder(self.prediction_head(outputs))
        if labels is not None:
            masked_lm_loss = self.loss_fn(outputs, labels)
        else:
            masked_lm_loss = outputs.new_zeros((1,))
        return masked_lm_loss

    def _prediction_fixed(self, outputs: torch.Tensor, labels: Optional[torch.Tensor] = None):
        outputs = self.decoder(self.prediction_head(outputs))
        if labels is not None:
            masked_lm_loss = self.loss_fn(outputs, labels.view(-1))
        else:
            masked_lm_loss = outputs.new_zeros((1,))
        return masked_lm_loss


class SequentialwithMask(torch.nn.Module):
    """Modified sequential class."""

    def __init__(self, list_of_modules):
        super().__init__()
        self.seq_modules = torch.nn.ModuleList(list_of_modules)
        self.LAYOUT = self.seq_modules[0].LAYOUT

    def forward(self, states, attention_mask: Optional[torch.Tensor] = None):
        for module in self.seq_modules:
            states = module(states, attention_mask)
        return states

    @torch.jit.export
    def __len__(self):
        return len(self.seq_modules)
