## FOR HPC results
import glob, re
import numpy as np

prior = 'flow'

vals = []
for fn in sorted(glob.glob(f"/work3/s214659/AdvancedML/results/test_elbo_{prior}_seed*.txt")):
    s = open(fn).read()
    m = re.search(r"test_elbo=([-\d\.eE]+)", s)
    if m:
        vals.append(float(m.group(1)))

vals = np.array(vals)
print("N =", len(vals))
print("mean =", vals.mean())
print("std  =", vals.std(ddof=1))