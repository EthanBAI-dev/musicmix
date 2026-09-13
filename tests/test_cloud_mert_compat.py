"""Opt-in integration: official MERT code/config, tiny random weights, CPU.

MUSICMIX_MERT_COMPAT=1 python -m pytest tests/test_cloud_mert_compat.py
Downloads code/config only, not pretrained weights. Use the pinned Colab env.
"""
import os

import pytest
import torch

from scripts.cloud_train import Tagger, seed_all


@pytest.mark.skipif(os.environ.get('MUSICMIX_MERT_COMPAT') != '1',
                    reason='Opt-in network/code integration test')
@pytest.mark.parametrize('model_id', ['m-a-p/MERT-v1-95M', 'm-a-p/MERT-v1-330M'])
def test_official_mert_forward_backward(model_id):
    from huggingface_hub import model_info
    from transformers import AutoConfig, AutoModel
    revision = model_info(model_id).sha
    cfg = AutoConfig.from_pretrained(model_id, revision=revision, trust_remote_code=True)
    cfg.hidden_size = 16
    cfg.intermediate_size = 32
    cfg.num_hidden_layers = 2
    cfg.num_attention_heads = 2
    cfg.num_conv_pos_embedding_groups = 2
    cfg.conv_dim = [8] * len(cfg.conv_dim)
    cfg.num_conv_pos_embeddings = 16
    cfg.apply_spec_augment = False
    cfg.layerdrop = 0.
    for mode in ['frozen', 'last', 'full']:
        seed_all(0)
        backbone = AutoModel.from_config(cfg, trust_remote_code=True, code_revision=revision)
        model = Tagger(backbone, 3, mode, 1).train()
        x = torch.randn(2, 2400)
        mask = torch.ones_like(x, dtype=torch.bool)
        mask[0, 1900:] = False
        result = model(x, mask)
        assert result.shape == (2, 3) and torch.isfinite(result).all()
        result.sum().backward()
        assert model.head[-1].weight.grad is not None
        last = list(backbone.encoder.layers[-1].parameters())
        assert any(p.grad is not None for p in last) == (mode != 'frozen')
