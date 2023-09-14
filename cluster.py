# collecting the different SMARTS patterns associated with clusters of espaloma
# parameter values. partly adapted from the bottom of from_besmarts.py

import os
import shutil

from openff.qcsubmit.results import OptimizationResultCollection
from openff.toolkit import ForceField
from tqdm import tqdm
from vflib.draw import draw_rdkit

from main import espaloma_label


def find(lst, fun):
    """Return the first element in `lst` satisfying the predicate `fun` or None
    otherwise"""
    for elt in lst:
        if fun(elt):
            return elt
    return None


sage = ForceField("openff-2.1.0.offxml")
opt = OptimizationResultCollection.parse_file("filtered-opt.json")
molecules = opt.to_molecules()

with open("bonds.dat", "r") as inp:
    for line in inp:
        if line.startswith("#"):
            continue
        # take just the first data line for now
        break

sp = line.split()
smirks = sp[0]

out_dir = "bond0"
shutil.rmtree(out_dir)
os.mkdir(out_dir)

seen = set()
for m, molecule in tqdm(
    enumerate(molecules), total=len(molecules), desc="Labeling molecules"
):
    matches = molecule.chemical_environment_matches(smirks)
    if matches:
        _, esp = espaloma_label(molecule)
        esp = [
            (b.atom1 - 1, b.atom2 - 1)
            for b in esp["bonds"]
            if (b.atom1 - 1, b.atom2 - 1) in matches and b.k < 720.0
        ]
        if esp:
            inchi = molecule.to_inchikey()
            if inchi not in seen:
                draw_rdkit(
                    molecule,
                    smirks,
                    matches=esp,
                    filename=f"{out_dir}/mol{m:04d}.png",
                )
                seen.add(inchi)


# for bond0 it's already looking like the much lower espaloma values are for a
# carbon bound to a nitrogen. maybe I should try breaking the pattern on that

# looks like I can do something like "[#7X3]-[#6X4:1]-[#1:2]" where the
# nitrogen doesn't have a number (:1 or :2)
