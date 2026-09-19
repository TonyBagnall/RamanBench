"""Retry the eight missing classification exports in the workspace."""
import logging
from pathlib import Path

import download_regression_aeon as environment
import httpx
import zenodo_get.downloader as downloader
import zenodo_get.zget as zenodo_api
import export_ramanbench_to_aeon_ts as exporter
from raman_bench.config import load_config

downloader.httpx = httpx
zenodo_api.httpx = httpx
NAMES = {
    'alzheimer', 'covid19_salvia', 'parkinson', 'hair_dyes_sers',
    'head_neck_cancer', 'serum_alzheimer_disease', 'serum_prostate_cancer',
    'wheat_lines',
}

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    original = exporter._load_metric_problem_keys
    def selected(path):
        frame = original(path)
        return frame[frame['key'].isin({name + '_0' for name in NAMES})]
    exporter._load_metric_problem_keys = selected
    config = load_config(str(environment.ROOT / 'configs/benchmark_v0.1.json'))
    config['cache_dir'] = str(environment.ROOT / '.fresh_loader_check')
    # Keep retry manifests separate from the earlier partial export manifest.
    write_manifests = exporter._write_manifests
    def retry_manifests(records, output_dir, writer_name, **kwargs):
        write_manifests(records, output_dir / 'classification_retry', writer_name, **kwargs)
    exporter._write_manifests = retry_manifests
    raise SystemExit(exporter.export(config, environment.ROOT / 'results',
        Path(exporter._default_classification_metrics()), all_resamples=True))
