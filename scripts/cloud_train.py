"""Research-only raw-audio MERT training; see colab/README.md.

Train/validation select models; test runs only through --evaluate-test.
Checkpoints are trusted local pickle files, never load untrusted checkpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import random
import subprocess
import tarfile
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.datasets.jamendo import load_split
from src.eval.tagging import macro_average_precision, macro_f1, tune_thresholds, valid_tag_mask


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_save(value, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.partial')
    torch.save(value, tmp)
    os.replace(tmp, path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available() and state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


class AudioSet(Dataset):
    """Read unpacked audio or durable uncompressed Drive tar archives.

    No archive extraction / deletion. Only two tar handles cached per worker.
    Chunk-grouped training order limits Drive seeks; no full dataset in RAM.
    """
    def __init__(self, tracks, labels, root, seconds, segments, train, seed, sr=24000):
        self.tracks, self.labels, self.root = tracks, labels, Path(root)
        self.seconds, self.segments, self.training = seconds, segments, train
        self.seed, self.sr, self.epoch = seed, sr, 0
        self.handles = OrderedDict()

    def __len__(self):
        return len(self.tracks)

    def source(self, track):
        direct = track.audio_path(self.root)
        if direct.is_file():
            return str(direct)
        archive = self.root / 'audio_tar' / f'raw_30s_audio-{track.chunk:02d}.tar'
        if archive not in self.handles:
            if len(self.handles) >= 2:
                self.handles.popitem(last=False)[1][0].close()
            tf = tarfile.open(archive, 'r:')
            members = {m.name.removeprefix('./'): m for m in tf if m.isfile()}
            self.handles[archive] = (tf, members)
        self.handles.move_to_end(archive)
        tf, members = self.handles[archive]
        with tf.extractfile(members[track.path]) as f:
            return io.BytesIO(f.read())

    def __getitem__(self, idx):
        import librosa
        import soundfile as sf
        with sf.SoundFile(self.source(self.tracks[idx])) as f:
            length = max(1, int(self.seconds * f.samplerate))
            span = max(0, len(f) - length)
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, idx]))
            starts = ([int(rng.integers(span + 1))] if self.training else
                      ([span // 2] if self.segments == 1 else
                       np.linspace(0, span, self.segments, dtype=int)))
            clips = []
            for start in starts:
                f.seek(int(start))
                y = f.read(length, dtype='float32', always_2d=True).mean(axis=1)
                if not len(y) or not np.isfinite(y).all():
                    raise ValueError(f'Invalid audio: {self.tracks[idx].path}')
                y = librosa.resample(y, orig_sr=f.samplerate, target_sr=self.sr)
                # Match Wav2Vec2FeatureExtractor normalization on valid samples.
                y = (y - y.mean()) / np.sqrt(y.var() + 1e-7)
                need = round(self.seconds * self.sr)
                y = y[:need]
                n = len(y)
                clips.append((np.pad(y, (0, need-n)), np.arange(need) < n))
        return (torch.from_numpy(np.stack([c[0] for c in clips])),
                torch.from_numpy(np.stack([c[1] for c in clips])),
                torch.from_numpy(self.labels[idx]))


class Tagger(nn.Module):
    def __init__(self, backbone, tags, mode, last_n):
        super().__init__()
        self.backbone, self.mode = backbone, mode
        backbone.requires_grad_(mode == 'full')
        if mode == 'last':
            if not 1 <= last_n <= len(backbone.encoder.layers):
                raise ValueError('last-n outside encoder layer range')
            for layer in backbone.encoder.layers[-last_n:]:
                layer.requires_grad_(True)
        self.head = nn.Sequential(nn.LayerNorm(backbone.config.hidden_size), nn.Dropout(.3),
                                  nn.Linear(backbone.config.hidden_size, 512), nn.GELU(),
                                  nn.Dropout(.3), nn.Linear(512, tags))

    def train(self, mode=True):
        super().train(mode)
        # Frozen prefix stays deterministic even during partial fine-tuning.
        self.backbone.eval()
        if mode and self.mode == 'full':
            self.backbone.train()
        elif mode and self.mode == 'last':
            for layer in self.backbone.encoder.layers:
                if any(p.requires_grad for p in layer.parameters()):
                    layer.train()
        return self

    def forward(self, x, mask):
        h = self.backbone(input_values=x, attention_mask=mask.long()).last_hidden_state
        valid = self.backbone._get_feature_vector_attention_mask(h.shape[1], mask.long())
        pooled = (h * valid.unsqueeze(-1)).sum(1) / valid.sum(1).clamp_min(1).unsqueeze(-1)
        return self.head(pooled)


@torch.no_grad()
def predict(model, ds, args, device):
    model.eval()
    ys, ps = [], []
    for x, mask, y in DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers):
        # Serial segment inference bounds validation VRAM; aggregate probabilities.
        p = []
        for k in range(x.shape[1]):
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                logits = model(x[:, k].to(device), mask[:, k].to(device))
            p.append(logits.float().sigmoid().cpu().numpy())
        ys.append(y.numpy())
        ps.append(np.mean(p, axis=0))
    return np.concatenate(ys), np.concatenate(ps)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--model', default='m-a-p/MERT-v1-95M',
                   choices=['m-a-p/MERT-v1-95M', 'm-a-p/MERT-v1-330M'])
    p.add_argument('--revision', required=True, help='Immutable Hugging Face commit SHA')
    p.add_argument('--mode', choices=['frozen', 'last', 'full'], default='last')
    p.add_argument('--last-n', type=int, default=2)
    p.add_argument('--subset', default='autotagging_top50tags')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--accumulation', type=int, default=8)
    p.add_argument('--seconds', type=float, default=10)
    p.add_argument('--eval-segments', type=int, default=3)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--head-lr', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--save-every', type=int, default=200, help='Optimizer steps')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--evaluate-test', action='store_true')
    p.add_argument('--smoke-limit', type=int, default=0, help='Non-reportable small pipeline test')
    p.add_argument('--allow-cpu', action='store_true')
    args = p.parse_args()
    import re
    if not re.fullmatch(r'[0-9a-f]{40}', args.revision):
        p.error('--revision must be a 40-character commit SHA')
    if min(args.epochs, args.batch_size, args.accumulation, args.eval_segments, args.save_every) < 1 or args.seconds < 1:
        p.error('Invalid positive training parameters')
    if not torch.cuda.is_available() and not args.allow_cpu:
        p.error('Select a GPU runtime; --allow-cpu is for development only')
    seed_all(args.seed)  # Before pretrained loader and random head initialization.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parts, vocab = load_split(args.subset, root=args.root, only_local=False)
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        for attr in ('track_id', 'artist_id'):
            if {getattr(t, attr) for t in parts[a]} & {getattr(t, attr) for t in parts[b]}:
                raise ValueError(f'{attr} leakage: {a}/{b}')
    if args.smoke_limit:
        parts = {k: v[:args.smoke_limit] for k, v in parts.items()}
    # Check exact paths, not merely whether their chunk directory exists.
    for name in (['validation', 'test'] if args.evaluate_test else ['train', 'validation']):
        for t in parts[name]:
            if not (t.audio_path(args.root).is_file() or
                    (args.root/'audio_tar'/f'raw_30s_audio-{t.chunk:02d}.tar').is_file()):
                raise FileNotFoundError(f'Missing {name} audio: {t.path}')
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
           if k not in ('resume', 'evaluate_test', 'out', 'root', 'allow_cpu', 'workers')}
    manifest = {k: [dict(id=t.track_id, artist=t.artist_id, path=t.path,
                         duration=t.duration, tags=t.tags) for t in v]
                for k, v in parts.items()}
    code_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    signature = digest(dict(config=cfg, tracks=manifest, tags=vocab.tags, code_sha=code_sha))
    checkpoint = args.out / ('best.pt' if args.evaluate_test else 'last.pt')
    state = None
    if args.resume or args.evaluate_test:
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if state['signature'] != signature:
            raise ValueError('Configuration / split changed; use a new output directory')
    elif (args.out/'run.json').exists():
        raise FileExistsError('Run exists: use --resume or a fresh --out')
    from transformers import AutoModel
    backbone = AutoModel.from_pretrained(args.model, revision=args.revision,
                                        trust_remote_code=True)
    # Explicit, stable augmentation policy rather than inherited pretraining defaults.
    backbone.config.apply_spec_augment = False
    backbone.config.layerdrop = 0.0
    model = Tagger(backbone, len(vocab), args.mode, args.last_n).to(device)
    opt = torch.optim.AdamW([
        dict(params=[v for v in backbone.parameters() if v.requires_grad], lr=args.lr),
        dict(params=model.head.parameters(), lr=args.head_lr)], weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    if not state:
        git = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
        revision = git.stdout.strip() if git.returncode == 0 else None
        (args.out/'run.json').write_text(json.dumps(dict(config=cfg, signature=signature,
            tags=vocab.tags, tracks=manifest, git=revision, training_code_sha256=code_sha,
            torch=torch.__version__,
            device=torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'), indent=2))
        (args.out/'environment.txt').write_text(subprocess.check_output(
            [os.sys.executable, '-m', 'pip', 'freeze'], text=True))
    ds = {k: AudioSet(v, vocab.encode(v), args.root, args.seconds, args.eval_segments,
                      k == 'train', args.seed) for k, v in parts.items()}
    epoch, cursor, best, step = 0, 0, -1., 0
    if state:
        model.load_state_dict(state['model'])
        opt.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        epoch, cursor, best, step = (state[k] for k in ('epoch', 'cursor', 'best', 'step'))
        restore_rng(state['rng'])
    if args.evaluate_test:
        y, score = predict(model, ds['test'], args, device)
        thresholds = state['thresholds']
        np.savez_compressed(args.out/'test_predictions.npz', y=y, score=score,
            ids=[t.track_id for t in parts['test']], thresholds=thresholds)
        report = dict(mAP=macro_average_precision(y, score),
                      macroF1=macro_f1(y, score, thresholds),
                      n_valid_tags=int(valid_tag_mask(y).sum()),
                      smoke=bool(args.smoke_limit), checkpoint='best.pt')
        (args.out/'test_metrics.json').write_text(json.dumps(report, indent=2))
        print(report)
        return
    def save(path, e, c, thresholds=None):
        atomic_save(dict(model=model.state_dict(), optimizer=opt.state_dict(),
            scaler=scaler.state_dict(), rng=rng_state(), epoch=e, cursor=c,
            step=step, best=best, signature=signature, config=cfg,
            tags=vocab.tags, thresholds=thresholds), path)
    while epoch < args.epochs:
        start = time.monotonic()
        ds['train'].epoch = epoch
        rng = np.random.default_rng(args.seed + epoch)
        order = []
        chunks = sorted({t.chunk for t in parts['train']})
        for chunk in rng.permutation(chunks):
            indices = [i for i, t in enumerate(parts['train']) if t.chunk == chunk]
            order.extend(rng.permutation(indices).tolist())
        batches = [order[i:i+args.batch_size] for i in range(0, len(order), args.batch_size)]
        loader = DataLoader(ds['train'], batch_sampler=batches[cursor:], num_workers=args.workers,
                            generator=torch.Generator().manual_seed(args.seed + epoch))
        model.train()
        opt.zero_grad(set_to_none=True)
        for b, (x, mask, y) in enumerate(loader, start=cursor):
            # Last accumulation group may contain fewer batches AND a short batch.
            group_start = (b // args.accumulation) * args.accumulation
            group_end = min(group_start + args.accumulation, len(batches))
            group_samples = sum(len(batch) for batch in batches[group_start:group_end])
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                logits = model(x[:, 0].to(device), mask[:, 0].to(device))
                loss = nn.functional.binary_cross_entropy_with_logits(logits, y.to(device))
                loss = loss * len(y) / group_samples
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss at epoch {epoch} batch {b}')
            scaler.scale(loss).backward()
            if b + 1 == group_end:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % args.save_every == 0:
                    save(args.out/'last.pt', epoch, b+1)
                if step % 20 == 0:
                    print(f'epoch={epoch+1} batch={b+1}/{len(batches)} step={step}', flush=True)
        y, score = predict(model, ds['validation'], args, device)
        val = macro_average_precision(y, score)
        if not math.isfinite(val):
            raise ValueError('No valid validation labels; enlarge smoke sample')
        thresholds = tune_thresholds(y, score)
        if val > best:
            best = val
            save(args.out/'best.pt', epoch+1, 0, thresholds)
            np.savez_compressed(args.out/'validation_predictions.npz', y=y, score=score,
                ids=[t.track_id for t in parts['validation']], thresholds=thresholds)
        row = dict(epoch=epoch+1, val_mAP=val, seconds=time.monotonic()-start, best=best)
        with (args.out/'history.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        print(row, flush=True)
        epoch, cursor = epoch+1, 0
        save(args.out/'last.pt', epoch, 0)


if __name__ == '__main__':
    main()
