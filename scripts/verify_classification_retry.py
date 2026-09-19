"""Compare all retried classification splits with their benchmark source arrays."""
import json
import numpy as np
from aeon.datasets import load_from_ts_file
import retry_classification as retry
from raman_bench.benchmark import RamanBenchmark
from raman_bench.config import load_config
from raman_bench.seeds import get_seeds

root = retry.environment.ROOT
config = load_config(str(root / 'configs/benchmark_v0.1.json'))
records = []
for resample, seed in enumerate(get_seeds(config)):
    benchmark = RamanBenchmark(
        dataset_names_classification=sorted(retry.NAMES), dataset_names_regression=[],
        test_size=config['test_size'], random_state=seed,
        cache_dir=str(root / '.fresh_loader_check'),
        min_samples_per_class=config.get('min_samples_per_class', 9),
        group_regression_splits=config.get('group_regression_splits', True))
    for name in sorted(retry.NAMES):
        key = name + '_0'
        train, test = benchmark._load_dataset_from_key(key)
        for suffix, frame in [('TRAIN', train), ('TEST', test)]:
            source_X, source_y = retry.exporter._dataframe_to_xy(frame)
            path = root / 'results/classification' / key / f'{key}{resample}_{suffix}.ts'
            X, y = load_from_ts_file(str(path))
            np.testing.assert_allclose(X[:, 0, :], source_X)
            np.testing.assert_array_equal(y.astype(str), source_y.astype(str))
            del X, y, source_X, source_y
        records.append(dict(dataset=name, resample=resample, status='verified'))
        print(f'VERIFIED {key} resample {resample}', flush=True)
out = root / 'results/classification_retry/verification.json'
out.write_text(json.dumps(records, indent=2))
