import math
from math import nan
from jarvis.db.figshare import data as jdata
from jarvis.core.atoms import Atoms
from pymatgen.io.cif import CifWriter

d = jdata("megnet")
prop = "gap pbe"
# "gap pbe"
# prop = "e_form"
# "bulk modulus",
# "shear modulus",
max_samples = 40000
f1 = open("id_prop.csv", "w")
# print(len(d))
# print(d[0].keys())
# exit()
count = 0
for i in d:
    atoms = Atoms.from_dict(i["atoms"])
    jid = i["id"]
    poscar_name = jid
    target = i[prop]
    try:
        target = float(target)
        if not math.isnan(target):
                atoms = atoms.pymatgen_converter()
                atoms = CifWriter(atoms)
                atoms.write_file(jid+".cif")
                f1.write("%s,%6f\n" % (poscar_name, target))
                count += 1
                if count == max_samples:
                    break
    except ValueError:
        pass
f1.close()