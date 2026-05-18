from pathlib import Path
import shutil
import zipfile
import requests

URLS = [
    "https://github.com/MIND-Lab/Raman-Spectra-Data/archive/refs/heads/main.zip",
    "https://codeload.github.com/MIND-Lab/Raman-Spectra-Data/zip/refs/heads/main",
]

root = Path(".cache") / "datasets_raw" / "mind_shared_debug"
zip_path = root / "Raman-Spectra-Data.zip"

# Clean debug cache only.
if root.exists():
    shutil.rmtree(root)
root.mkdir(parents=True, exist_ok=True)

for url in URLS:
    print(f"\nTrying: {url}")
    r = requests.get(url, timeout=60, allow_redirects=True)
    print("status:", r.status_code)
    print("final url:", r.url)
    print("content-type:", r.headers.get("content-type"))
    print("bytes:", len(r.content))

    if r.status_code == 200 and r.content[:2] == b"PK":
        zip_path.write_bytes(r.content)
        print("Downloaded valid-looking zip:", zip_path)
        break

    print("First 200 response bytes:")
    print(r.content[:200])
else:
    raise RuntimeError("Both GitHub zip URLs failed.")

print("\nIs zipfile:", zipfile.is_zipfile(zip_path))

with zipfile.ZipFile(zip_path) as zf:
    names = zf.namelist()
    print("\nFirst 30 zip entries:")
    for name in names[:30]:
        print(" ", name)

    zf.extractall(root)

print("\nExtracted top level:")
for p in root.iterdir():
    print(" ", p)

matches = list(root.rglob("pd_ad_dataset"))
print("\npd_ad_dataset matches:")
for p in matches:
    print(" ", p)

if not matches:
    raise FileNotFoundError("pd_ad_dataset was not found after extraction.")

dataset_dir = matches[0]
patient_dirs = [p for p in dataset_dir.iterdir() if p.is_dir()]
print("\nDataset dir:", dataset_dir)
print("Patient folders:", len(patient_dirs))

print("\nFirst 5 patient folders and files:")
for p in patient_dirs[:5]:
    print(" ", p.name, sorted(x.name for x in p.iterdir())[:10])

required = {"spectra.csv", "raman_shift.csv", "user_information.csv"}
bad = []
for p in patient_dirs:
    have = {x.name for x in p.iterdir()}
    missing = required - have
    if missing:
        bad.append((p.name, missing))

print("\nFolders missing required files:", len(bad))
for item in bad[:10]:
    print(" ", item)