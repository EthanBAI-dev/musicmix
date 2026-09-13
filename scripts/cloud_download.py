"""Download all Jamendo audio archives durably, without unpacking 500+ GB.

Resume .partial downloads; validate exact metadata members and archive bounds.
SHA256 receipt is a local integrity record, NOT an official authenticity checksum.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile

from scripts.download_jamendo import chunk_url, download_meta, remote_size
from src.datasets.jamendo import _read_tsv, load_split


def validate_archive(path, expected):
    size = path.stat().st_size
    if size % 512:
        raise ValueError(f'Truncated tar: {path}')
    with tarfile.open(path, 'r:') as tf:
        members = {m.name.removeprefix('./'): m for m in tf if m.isfile()}
        missing = set(expected) - members.keys()
        if missing:
            raise ValueError(f'Missing {len(missing)} tracks, e.g. {sorted(missing)[:3]}')
        if any(m.size <= 0 or m.offset_data + m.size > size for m in members.values()):
            raise ValueError(f'Invalid member extent: {path}')
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return dict(bytes=size, sha256=h.hexdigest(), tracks=len(members))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--stop', type=int, default=100)
    p.add_argument('--verify-existing', action='store_true')
    args = p.parse_args()
    if not 0 <= args.start < args.stop <= 100:
        p.error('Require 0 <= start < stop <= 100')
    args.root.mkdir(parents=True, exist_ok=True)
    # Keep ALL nested split metadata on Drive. Do not replace it on every resume.
    if not (args.root/'meta'/'splits'/'split-0'/'autotagging_top50tags-test.tsv').exists():
        download_meta(args.root)
    parts, _ = load_split('autotagging_top50tags', root=args.root, only_local=False)
    all_tracks = _read_tsv(args.root/'meta'/'autotagging_real.tsv')
    archive_dir = args.root/'audio_tar'
    archive_dir.mkdir(exist_ok=True)
    for chunk in range(args.start, args.stop):
        target = archive_dir/f'raw_30s_audio-{chunk:02d}.tar'
        receipt = target.with_suffix('.json')
        expected = {t.path for t in all_tracks if t.chunk == chunk}
        expected.update(t.path for v in parts.values() for t in v if t.chunk == chunk)
        if not expected:
            raise ValueError(f'No expected members for chunk {chunk}')
        if target.exists() and receipt.exists():
            old = json.loads(receipt.read_text())
            if old['bytes'] == target.stat().st_size and not args.verify_existing:
                print(f'{chunk:02d}: previously verified (size check)', flush=True)
                continue
        url = chunk_url('mtg-fast', 'audio', chunk)
        partial = target.with_suffix('.tar.partial')
        if not target.exists():
            subprocess.run(['curl', '-fL', '--retry', '5', '--connect-timeout', '30',
                '--speed-limit', '1024', '--speed-time', '120', '--no-progress-meter',
                '-C', '-', '-o', str(partial), url], check=True)
            expected_size = remote_size(url)
            if expected_size and expected_size != partial.stat().st_size:
                raise ValueError('Remote size mismatch; partial preserved')
        candidate = target if target.exists() else partial
        report = validate_archive(candidate, expected)
        if target.exists() and receipt.exists():
            if json.loads(receipt.read_text())['sha256'] != report['sha256']:
                raise ValueError(f'Checksum changed: {target}; investigate before re-download')
        if candidate == partial:
            os.replace(partial, target)
        report['url'] = url
        tmp = receipt.with_suffix('.json.partial')
        tmp.write_text(json.dumps(report, indent=2))
        os.replace(tmp, receipt)
        print(f'{chunk:02d}: verified {report}', flush=True)


if __name__ == '__main__':
    main()
