"""Aggregate independent seed shards of the CAL-09.2 canonical 48-scan design.

Works with GitHub Actions matrix artifacts; refuses duplicate/missing cases,
divergent configuration checksums or noncanonical angular steps.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from tools.validate_cal092_observability import (
    SEEDS, DEFAULT_VARIANTS, RESOLUTIONS_M, FORMAT,
    summarize, report,
)


def aggregate(paths):
    if not paths:
        raise ValueError('No shards supplied')
    sha = None
    fingerprint = None
    rows = []
    for p in paths:
        d = json.loads(Path(p).read_text(encoding='utf-8'))
        if d.get('format') != FORMAT:
            raise ValueError(f'Non CAL-09.2 results: {p}')
        source = d.get('source_config_sha256')
        if sha is None:
            sha = source
        if source != sha:
            raise ValueError('Source config SHA-256 does not match across shards')
        fp = d.get('run_fingerprint')
        if fingerprint is None:
            fingerprint = fp
        if fp != fingerprint:
            raise ValueError('Sampling/scanner configuration differs across shards')
        rows.extend(d['cases'])
    if len(rows) != len(SEEDS) * len(DEFAULT_VARIANTS) * len(RESOLUTIONS_M):
        raise ValueError(f'Expected exactly 48 rows, got {len(rows)}')
    result = summarize(rows, seeds=SEEDS, variants=DEFAULT_VARIANTS,
                       resolutions=RESOLUTIONS_M, angular_step_deg=.1)
    if not result['checks']['complete_canonical_design']:
        raise ValueError('Not a canonical eight-seed CAL-09.2 aggregation')
    if not result['checks']['matched_layouts_and_station_counts']:
        raise ValueError('Paired scans mismatch physical layout/stations')
    if not result['checks']['sampled_direct_ray_provenance_valid']:
        raise ValueError('Invalid ray-to-triangle provenance in at least one scene')
    result['source_config_sha256'] = sha
    result['run_fingerprint'] = fingerprint
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('shards', nargs='+', type=Path)
    args = ap.parse_args()
    result = aggregate(args.shards)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'cal092_results.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    (args.out / 'CAL092_REPORT.md').write_text(report(result), encoding='utf-8')
    print(report(result), flush=True)


if __name__ == '__main__':
    main()
