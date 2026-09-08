# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import copy

import pytest
import torch

from nemo.collections.tts.modules.magpietts_modules import AcousticDecoderTransformer
from nemo.collections.tts.modules.transformer_2501 import Transformer


class _SemanticVectorQuantizer:
    @staticmethod
    def decode(indices, input_len):
        del input_len
        return indices.float().permute(1, 0, 2)


def _make_decoder(n_layers=12):
    d_model = 16
    transformer = Transformer(
        n_layers=n_layers,
        d_model=d_model,
        d_ffn=32,
        sa_n_heads=4,
        kernel_size=1,
        p_dropout=0.0,
        has_xattn=False,
        is_causal=True,
        apply_norm_out=True,
        use_learnable_pos_emb=True,
    )
    return AcousticDecoderTransformer(
        input_dim=d_model,
        d_model=d_model,
        semantic_dim=1,
        num_codebooks=12,
        codebook_size=8,
        transformer=transformer,
    )


def test_requires_one_three_layer_block_per_refinement_stage():
    with pytest.raises(ValueError, match='12 total'):
        _make_decoder(n_layers=6)


def test_selects_codebooks_with_fixed_one_three_four_four_schedule():
    decoder = _make_decoder()
    confidence = torch.arange(12, dtype=torch.float).view(1, 1, 12)
    unresolved = torch.ones_like(confidence, dtype=torch.bool)

    selected_counts = []
    for count in decoder.prediction_schedule:
        selected = decoder.select_codebooks(confidence, unresolved, count)
        selected_counts.append(selected.sum().item())
        unresolved &= ~selected

    assert selected_counts == [1, 3, 4, 4]
    assert not unresolved.any()


def test_uses_one_cumulative_transformer_and_feedback_between_stages():
    torch.manual_seed(0)
    decoder = _make_decoder().eval()
    layer_order = []
    stage_one_output = None
    stage_two_input = None
    handles = []

    for index, layer in enumerate(decoder.transformer.layers):
        handles.append(
            layer.register_forward_hook(lambda module, args, output, index=index: layer_order.append(index))
        )

    def capture_stage_one_output(module, args, output):
        nonlocal stage_one_output
        stage_one_output = output['output'].detach().clone()

    def capture_stage_two_input(module, args):
        nonlocal stage_two_input
        stage_two_input = args[0].detach().clone()

    handles.append(decoder.transformer.layers[2].register_forward_hook(capture_stage_one_output))
    handles.append(decoder.transformer.layers[3].register_forward_pre_hook(capture_stage_two_input))

    decoder(
        inputs=torch.randn(1, 2, 16),
        audio_lens=torch.tensor([2]),
        semantic_tokens=torch.randint(0, 8, (1, 1, 2)),
        vector_quantizer=_SemanticVectorQuantizer(),
    )
    for handle in handles:
        handle.remove()

    assert layer_order == list(range(12))
    assert stage_one_output is not None
    assert stage_two_input is not None
    assert not torch.equal(stage_two_input, stage_one_output)
    assert not hasattr(decoder, 'transformers')
    assert not hasattr(decoder, 'maskgit_step_embedding')


def test_staged_loss_is_finite_and_trains_every_layer():
    torch.manual_seed(0)
    decoder = _make_decoder()
    inputs = torch.randn(2, 4, 16)
    audio_lens = torch.tensor([4, 3])
    semantic_tokens = torch.randint(0, 8, (2, 1, 4))
    acoustic_tokens = torch.randint(0, 8, (2, 12, 4))

    predicted, logits, loss = decoder(
        inputs,
        audio_lens,
        semantic_tokens,
        _SemanticVectorQuantizer(),
        acoustic_tokens=acoustic_tokens,
    )
    loss.backward()

    assert predicted.shape == acoustic_tokens.shape
    assert logits.shape == (2, 4, 12 * 8)
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    assert all(
        any(parameter.grad is not None for parameter in layer.parameters()) for layer in decoder.transformer.layers
    )


def test_shared_kv_cache_matches_full_sequence_inference():
    torch.manual_seed(0)
    full_decoder = _make_decoder().eval()
    cached_decoder = copy.deepcopy(full_decoder)
    inputs = torch.randn(1, 3, 16)
    semantic_tokens = torch.randint(0, 8, (1, 1, 3))

    full_predictions, full_logits, _ = full_decoder(
        inputs,
        torch.tensor([3]),
        semantic_tokens,
        _SemanticVectorQuantizer(),
    )

    cached_decoder.reset_cache(use_cache=True)
    cached_predictions = []
    cached_logits = []
    for timestep in range(inputs.size(1)):
        predictions, logits, _ = cached_decoder(
            inputs[:, timestep : timestep + 1],
            torch.ones(1, dtype=torch.long),
            semantic_tokens[:, :, timestep : timestep + 1],
            _SemanticVectorQuantizer(),
        )
        cached_predictions.append(predictions)
        cached_logits.append(logits)
        assert cached_decoder.cache_sequence_length() == timestep + 1

    cached_predictions = torch.cat(cached_predictions, dim=-1)
    cached_logits = torch.cat(cached_logits, dim=1)
    assert torch.equal(cached_predictions, full_predictions)
    assert torch.allclose(cached_logits, full_logits, atol=1e-5, rtol=1e-5)
