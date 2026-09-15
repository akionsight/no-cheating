import scipy.io as sio, glob
from collections import Counter

files = glob.glob("results_v2/*.mat")
print(f"Total .mat files: {len(files)}")

violated = 0
by_diff = Counter()
for f in files:
    d = sio.loadmat(f)
    diff = str(d["difficulty"][0]) if "difficulty" in d else "?"
    v = int(d["violated"].item())
    by_diff[diff] += v
    if v:
        violated += 1
        print("VIOLATED:", f)

print(f"\n{violated} / {len(files)} rollouts violated")
print("Violations by difficulty:", dict(by_diff))