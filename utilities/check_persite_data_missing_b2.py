import os
import json

base = os.path.join("datasets", "lsodos_persitejsons_250930")
missing_b2 = []

files = [f for f in os.listdir(base) if f.endswith(".json")]
files = files[:]  # only first 100 files

for fname in files:
    path = os.path.join(base, fname)
    try:
        with open(path) as f:
            data = json.load(f)
        tdos = data.get("tdos_per_site", {})
        if "9" not in tdos:
            missing_b2.append(fname)
    except Exception as e:
        print(f"Error reading {path}: {e}")

print("Files with no B2 (no key '9'):")
for f in missing_b2:
    print(f)
print(f"Total files with missing B2: {len(missing_b2)} out of {len(files)}")