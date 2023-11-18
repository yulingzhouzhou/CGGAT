import jarvis
from jarvis.core.atoms import Atoms
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

path = 'dft_OPT'
poscar_files = [f for f in os.listdir(path) if f.endswith("vasp")]
ele_types_count = defaultdict(int)

for poscar_file in poscar_files:
    file_path = os.path.join(path, poscar_file)
    structure = Atoms.from_poscar(file_path)
    ele_types = len(structure.composition.to_dict())
    ele_types_count[ele_types] += 1

ele_quantities = list(ele_types_count.keys())
file_counts = list(ele_types_count.values())
print(ele_quantities, file_counts)

plt.figure(figsize=(9, 7))
plt.bar(ele_quantities, file_counts, color='#6140ef') # #6140ef
plt.xlabel("Number of elements")
plt.ylabel("Count")
# plt.rcParams['font.family'] = 'Time New Roman'
# plt.rcParams['font.size'] = 18
plt.savefig("dft_OPT.eps", format='eps')
plt.show()