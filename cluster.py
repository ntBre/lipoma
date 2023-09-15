# collecting the different SMARTS patterns associated with clusters of espaloma
# parameter values. partly adapted from the bottom of from_besmarts.py

import os
import shutil

from openff.qcsubmit.results import OptimizationResultCollection
from openff.toolkit import Molecule
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


def deduplicate_by(lst, fn):
    "Deduplicate `lst` by the key generated from `fn`."
    keys = set()
    ret = []
    for elt in lst:
        key = fn(elt)
        if key in keys:
            continue
        keys.add(key)
        ret.append(elt)
    return ret


def load_smirks(filename):
    """Extracts the initial SMIRKS pattern from lines of the form below.

    [#6X4:1]-[#1:2] 50377  719.64  740.36  740.36  740.36
    """
    with open(filename, "r") as inp:
        return [line.split()[0] for line in inp if not line.startswith("#")]


opt = OptimizationResultCollection.parse_file("filtered-opt.json")

# demand execution here so I can deduplicate and reuse molecules
molecules = [m for m in tqdm(opt.to_molecules(), desc="Converting molecules")]
print(len(molecules), " initially")
molecules = deduplicate_by(molecules, Molecule.to_inchikey)
print(len(molecules), " after dedup")

smirkss = load_smirks("bonds.dat")
for s, smirks in tqdm(enumerate(smirkss), desc="Processing smirks"):
    out_dir = f"output/cluster/bond{s:02d}"
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)

    for m, molecule in enumerate(molecules):
        matches = molecule.chemical_environment_matches(smirks)
        if matches:
            _, esp = espaloma_label(molecule)
            esp = [
                (b.atom1 - 1, b.atom2 - 1)
                for b in esp["bonds"]
                if (b.atom1 - 1, b.atom2 - 1) in matches and b.k < 720.0
            ]
            if esp:
                draw_rdkit(
                    molecule,
                    smirks,
                    matches=esp,
                    filename=f"{out_dir}/mol{m:04d}.png",
                )
    break


# for bond0 it's already looking like the much lower espaloma values are for a
# carbon bound to a nitrogen. maybe I should try breaking the pattern on that

# looks like I can do something like "[#7X3]-[#6X4:1]-[#1:2]" where the
# nitrogen doesn't have a number (:1 or :2)
