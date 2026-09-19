"""Package recovered classification files with ZIP64 and verify their checksums."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

root = Path(__file__).resolve().parents[1]
names = (
    'alzheimer_0', 'covid19_salvia_0', 'hair_dyes_sers_0',
    'head_neck_cancer_0', 'parkinson_0', 'serum_alzheimer_disease_0',
    'serum_prostate_cancer_0', 'wheat_lines_0',
)
files = []
for name in names:
    directory = root / 'results/classification' / name
    expected = [directory / f'{name}{seed}_{split}.ts'
                for seed in range(3) for split in ('TRAIN', 'TEST')]
    for path in expected:
        if not path.is_file():
            raise FileNotFoundError(path)
    files.extend(expected)

destination = root / 'results/classification_recovered_eight.zip'
with ZipFile(destination, 'x', compression=ZIP_DEFLATED, compresslevel=1,
             allowZip64=True) as archive:
    for path in files:
        archive.write(path, path.relative_to(root / 'results').as_posix())
        print(f'Packed {path.name}', flush=True)
with ZipFile(destination) as archive:
    assert len(archive.infolist()) == 48
    bad = archive.testzip()
    if bad:
        raise RuntimeError(f'ZIP checksum verification failed: {bad}')
print(f'VERIFIED {destination}: 48 files, {destination.stat().st_size} bytes', flush=True)
