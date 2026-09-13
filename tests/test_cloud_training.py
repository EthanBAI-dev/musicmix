"""Offline tests: no downloaded weights, real WAV input and training/checkpoint loop."""
import io
import json
import sys
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch
from torch import nn

from scripts.cloud_download import validate_archive
from scripts.cloud_train import AudioSet, Tagger, main, restore_rng, rng_state, seed_all
from src.datasets.jamendo import Track


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4)
        self.feature = nn.Linear(1, 4)
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])

    def forward(self, input_values, attention_mask):
        h = self.feature(input_values[:, ::400, None])
        for layer in self.encoder.layers:
            h = torch.tanh(layer(h))
        return SimpleNamespace(last_hidden_state=h)

    def _get_feature_vector_attention_mask(self, length, mask):
        return mask[:, ::400][:, :length].bool()


@pytest.mark.parametrize('mode', ['frozen', 'last', 'full'])
def test_gradient_scope(mode):
    seed_all(9)
    model = Tagger(TinyBackbone(), 2, mode, 1).train()
    x = torch.randn(2, 2400)
    model(x, torch.ones_like(x, dtype=torch.bool)).sum().backward()
    assert model.head[-1].weight.grad is not None
    assert (model.backbone.feature.weight.grad is not None) == (mode == 'full')
    assert (model.backbone.encoder.layers[-1].weight.grad is not None) == (mode != 'frozen')


def test_rng_roundtrip():
    seed_all(9)
    state = rng_state()
    expected = torch.rand(3)
    restore_rng(state)
    torch.testing.assert_close(expected, torch.rand(3))


def test_tar_audio_and_validation(tmp_path):
    content = io.BytesIO()
    sf.write(content, np.sin(np.arange(48000)/20), 24000, format='WAV')
    folder = tmp_path/'audio_tar'
    folder.mkdir()
    path = folder/'raw_30s_audio-00.tar'
    with tarfile.open(path, 'w') as tf:
        member = tarfile.TarInfo('00/sample.wav')
        member.size = len(content.getvalue())
        tf.addfile(member, io.BytesIO(content.getvalue()))
    assert validate_archive(path, {'00/sample.wav'})['tracks'] == 1
    with pytest.raises(ValueError, match='Missing'):
        validate_archive(path, {'missing'})
    track = Track('1', '1', '1', '00/sample.wav', 2., ('a',))
    ds = AudioSet([track], np.ones((1, 1), np.float32), tmp_path, 1, 3, False, 0)
    x, mask, y = ds[0]
    assert x.shape == (3, 24000)
    assert mask.all() and y.shape == (1,)


def test_train_resume_and_explicit_test(tmp_path, monkeypatch):
    # Fake only the expensive pretrained loader, not dataset, optimizer or metrics.
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        AutoModel=SimpleNamespace(from_pretrained=lambda *a, **k: TinyBackbone())))
    root = tmp_path/'data'
    split = root/'meta'/'splits'/'split-0'
    split.mkdir(parents=True)
    for k, part in enumerate(['train', 'validation', 'test']):
        rows = ['TRACK_ID\tARTIST_ID\tALBUM_ID\tPATH\tDURATION\tTAGS']
        for i in range(8 if part == 'train' else 4):
            file = root/'audio'/f'{k:02d}'/f'{i}.wav'
            file.parent.mkdir(parents=True, exist_ok=True)
            sf.write(file, np.sin(np.arange(24000)/(i+10)), 24000)
            rows.append(f'{k}-{i}\tartist-{k}-{i}\talbum-{k}-{i}\t{k:02d}/{i}.wav\t1\tgenre---{i%2}')
        (split/f'autotagging_top50tags-{part}.tsv').write_text('\n'.join(rows)+'\n')
    out = tmp_path/'run'
    argv = ['train', '--root', str(root), '--out', str(out), '--revision', 'a'*40,
            '--epochs', '1', '--seconds', '1', '--batch-size', '2', '--accumulation', '3',
            '--last-n', '1', '--allow-cpu', '--eval-segments', '2', '--save-every', '1']
    monkeypatch.setattr(sys, 'argv', argv)
    main()
    assert (out/'best.pt').exists()
    assert not (out/'test_metrics.json').exists()
    monkeypatch.setattr(sys, 'argv', argv+['--resume'])
    main()
    monkeypatch.setattr(sys, 'argv', argv+['--evaluate-test'])
    main()
    assert json.loads((out/'test_metrics.json').read_text())['n_valid_tags'] == 2

    # Simulate teardown immediately after an optimizer-boundary checkpoint.
    import scripts.cloud_train as training
    real_save = training.atomic_save
    def interrupt(value, path):
        real_save(value, path)
        if path.name == 'last.pt':
            raise KeyboardInterrupt('simulated runtime disconnect')
    interrupted = tmp_path/'interrupted'
    resumed_argv = [str(interrupted) if a == str(out) else a for a in argv]
    monkeypatch.setattr(training, 'atomic_save', interrupt)
    monkeypatch.setattr(sys, 'argv', resumed_argv)
    with pytest.raises(KeyboardInterrupt):
        main()
    monkeypatch.setattr(training, 'atomic_save', real_save)
    monkeypatch.setattr(sys, 'argv', resumed_argv+['--resume'])
    main()
    original = torch.load(out/'best.pt', weights_only=False)['model']
    resumed = torch.load(interrupted/'best.pt', weights_only=False)['model']
    for key in original:
        torch.testing.assert_close(original[key], resumed[key], rtol=0, atol=0)
    monkeypatch.setattr(sys, 'argv', argv+['--resume', '--seed', '1'])
    with pytest.raises(ValueError, match='Configuration'):
        main()


def test_notebook_valid_python():
    import ast
    from pathlib import Path
    path = Path(__file__).resolve().parents[1]/'colab/train_mert.ipynb'
    notebook = json.loads(path.read_text())
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            ast.parse(''.join(cell['source']))
