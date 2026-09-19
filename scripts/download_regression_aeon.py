"""Download configured regression datasets and export verified aeon splits.

Run from the repository root: python scripts/download_regression_aeon.py
Each dataset runs in isolation; reruns reuse successful exports and raw downloads.
"""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'regression'
CACHE = ROOT / '.rd'
sys.path.insert(0, str(ROOT / 'src'))
os.environ['HF_HOME'] = str(CACHE / 'hf')
os.environ['HF_HUB_CACHE'] = str(CACHE / 'hf' / 'hub')
os.environ['KAGGLEHUB_CACHE'] = str(CACHE / 'kaggle')
os.environ['MPLCONFIGDIR'] = str(CACHE / 'matplotlib')
os.environ['SCP_CONFIG_HOME'] = str(CACHE / 'spectrochempy' / 'config')
os.environ['SCPY_LOGS'] = str(CACHE / 'spectrochempy' / 'logs')
Path(os.environ['SCP_CONFIG_HOME']).mkdir(parents=True, exist_ok=True)
Path(os.environ['SCPY_LOGS']).mkdir(parents=True, exist_ok=True)
os.environ['USERPROFILE'] = str(CACHE / 'user')
os.environ['HOME'] = str(CACHE / 'user')
Path(os.environ['USERPROFILE']).mkdir(parents=True, exist_ok=True)


def worker(name):
    import numpy as np
    # Windows environments may not permit symlinks; use the hub's copy fallback.
    import huggingface_hub.file_download as hf_download
    hf_download.are_symlinks_supported = lambda cache_dir=None: False
    # zenodo-get's HTTP client must match httpx-retries' transport classes.
    import httpx
    import zenodo_get.downloader as zenodo_downloader
    import zenodo_get.zget as zenodo_api
    zenodo_downloader.httpx = httpx
    zenodo_api.httpx = httpx
    # Kaggle can return a ZIP containing the requested XLSX instead of the XLSX.
    import pandas as pd
    import zipfile
    import io
    read_excel = pd.read_excel
    def read_wrapped_excel(source, *args, **kwargs):
        if isinstance(source, (str, Path)) and zipfile.is_zipfile(source):
            with zipfile.ZipFile(source) as archive:
                members = archive.namelist()
                if len(members) == 1 and members[0].lower().endswith('.xlsx'):
                    return read_excel(io.BytesIO(archive.read(members[0])), *args, **kwargs)
        return read_excel(source, *args, **kwargs)
    pd.read_excel = read_wrapped_excel
    # Preserve source split order while avoiding row-by-row conversion of wide spectra.
    import datasets
    load_hf_dataset = datasets.load_dataset
    def load_hf_frames(*args, **kwargs):
        result = load_hf_dataset(*args, **kwargs)
        if isinstance(result, datasets.DatasetDict):
            return {key: split.to_pandas() for key, split in result.items()}
        return result
    datasets.load_dataset = load_hf_frames
    def write_ts(X, y, path):
        try:
            save_to_ts_file(X, y, label_type='regression', path=str(path.parent),
                            problem_name=path.stem[:-6], file_suffix=path.stem[-6:])
            return
        except ValueError as exc:
            if 'y type is binary' not in str(exc):
                raise
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as f:
            f.write('@problemName ' + path.stem + '\n@timestamps false\n@missing false\n'
                    '@univariate true\n@equalLength true\n@seriesLength ' + str(X.shape[-1]) +
                    '\n@targetlabel true\n@data\n')
            for row, target in zip(X[:, 0, :], y):
                f.write(','.join(str(float(v)) for v in row) + ':' + str(float(target)) + '\n')
    from raman_data import raman_data
    from aeon.datasets import save_to_ts_file, load_from_ts_file
    import raman_bench.benchmark as module
    from raman_bench.config import load_config
    from raman_bench.seeds import get_seeds

    config = load_config(str(ROOT / 'configs' / 'benchmark_v0.1.json'))
    dataset = raman_data(name, cache_dir=str(CACHE / 'raw'))
    directory = OUT / name
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / 'data.npz', X=dataset.spectra,
                        y=dataset.targets, raman_shifts=dataset.raman_shifts,
                        target_names=np.asarray(dataset.target_names, dtype=str))
    # Reuse the loaded dataset while preserving the benchmark split implementation.
    original = module.raman_data
    module.raman_data = lambda dataset_name, **kw: dataset if dataset_name == name else original(dataset_name, **kw)
    rows = []
    targets = 1 if dataset.targets.ndim == 1 else dataset.targets.shape[1]
    for idx in range(targets):
        key = f'{name}_{idx}'
        target = str(dataset.target_names[idx])
        if key in config.get('exclude_keys', []) or target in config.get('exclude_targets', []):
            rows.append(dict(key=key, target=target, status='excluded_by_benchmark'))
            continue
        for resample, seed in enumerate(get_seeds(config)):
            bench = module.RamanBenchmark(dataset_names_classification=[],
                dataset_names_regression=[name], cache_dir=str(CACHE / 'splits'),
                random_state=seed, test_size=config['test_size'], group_regression_splits=True)
            train, test = bench._load_dataset_from_key(key)
            record = dict(key=key, target=target, resample=resample, seed=int(seed))
            for label, frame in [('TRAIN', train), ('TEST', test)]:
                if frame is None or frame.empty:
                    raise ValueError(f'{key}: empty {label} split')
                X = frame.iloc[:, :-1].to_numpy(dtype=float)[:, None, :]
                y = frame.iloc[:, -1].to_numpy(dtype=float)
                if not np.isfinite(X).all() or not np.isfinite(y).all():
                    raise ValueError(f'{key}: non-finite values')
                folder = directory / key
                folder.mkdir(exist_ok=True)
                stem = f'{key}{resample}'
                path = folder / f'{stem}_{label}.ts'
                write_ts(X, y, path)
                loaded_X, loaded_y = load_from_ts_file(str(path))
                np.testing.assert_allclose(loaded_X, X)
                np.testing.assert_allclose(loaded_y, y)
                record[label.lower() + '_file'] = str(path)
                record['n_' + label.lower()] = len(y)
            record.update(status='verified', n_timepoints=X.shape[-1])
            rows.append(record)
    (directory / 'manifest.json').write_text(json.dumps(rows, indent=2))
    print(f'COMPLETE {name}: {len(rows)} target/resample records', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset')
    parser.add_argument('--report-only', action='store_true')
    parser.add_argument('--timeout', type=int, default=1200)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    if args.dataset:
        worker(args.dataset)
        return
    names = json.loads((ROOT / 'configs/datasets/regression_all.json').read_text())
    records = []
    for name in names:
        log = OUT / f'{name}.log'
        if (OUT / name / 'manifest.json').exists():
            entries = json.loads((OUT / name / 'manifest.json').read_text())
            status = 'verified' if any(r['status'] == 'verified' for r in entries) else 'downloaded_excluded'
        elif args.report_only:
            status = 'incomplete'
        else:
            print(f'DOWNLOADING {name}', flush=True)
            with log.open('w', encoding='utf-8') as stream:
                try:
                    result = subprocess.run([sys.executable, __file__, '--dataset', name],
                        cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=args.timeout)
                    status = 'verified' if result.returncode == 0 else 'failed'
                except subprocess.TimeoutExpired:
                    status = 'timeout'
        records.append(dict(dataset=name, status=status, log=str(log)))
        (OUT / 'download_manifest.json').write_text(json.dumps(records, indent=2))
        print(f'{status.upper()} {name}', flush=True)
    print(json.dumps({s: sum(r['status'] == s for r in records)
                      for s in sorted({r['status'] for r in records})}), flush=True)


if __name__ == '__main__':
    main()
