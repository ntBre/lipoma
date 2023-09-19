# collecting the different SMARTS patterns associated with clusters of espaloma
# parameter values. partly adapted from the bottom of from_besmarts.py

import os
import shutil

import numpy as np
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
    """Extracts the SMIRKS pattern, Sage value, and Espaloma average from lines
    of the form below.

    # Smirks        Count    Sage    Rest
    [#6X4:1]-[#1:2] 50377  719.64  740.36  740.36  740.36

    """
    ret = []
    with open(filename, "r") as inp:
        for line in inp:
            if line.startswith("#"):
                continue
            sp = line.split()
            ret.append(
                (sp[0], float(sp[2]), np.mean([float(x) for x in sp[3:]]))
            )
        return ret


def load_thresh(filename):
    """load smirks, threshold pairs from `filename`. Returns a sequence of
    smirks, fn pairs, where `fn` is a function performing the appropriate
    comparison to threshold

    """
    ret = []
    with open(filename) as inp:
        for line in inp:
            sp = line.split()
            smirks = sp[0]
            v = float(sp[1][1:])
            if sp[1].startswith("<"):
                fn = lambda x, v=v: x < v  # noqa - I want lambda
            elif sp[1].startswith(">"):
                fn = lambda x, v=v: x > v  # noqa - ""
            else:
                raise ValueError(f"unrecognized operator `{sp[1][0]}`")
            ret.append((smirks, fn))
    return ret


def main():
    smirkss = load_thresh("data/labeled_bonds.dat")

    opt = OptimizationResultCollection.parse_file("datasets/filtered-opt.json")

    # demand execution here so I can deduplicate and reuse molecules
    molecules = [
        m for m in tqdm(opt.to_molecules(), desc="Converting molecules")
    ]
    print(len(molecules), " initially")
    molecules = deduplicate_by(molecules, Molecule.to_inchikey)
    print(len(molecules), " after dedup")

    for s, (smirks, fn) in tqdm(
        enumerate(smirkss), total=len(smirkss), desc="Processing smirks"
    ):
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
                    if (b.atom1 - 1, b.atom2 - 1) in matches and fn(b.k)
                ]
                if esp:
                    draw_rdkit(
                        molecule,
                        smirks,
                        matches=esp,
                        filename=f"{out_dir}/mol{m:04d}.png",
                    )


if __name__ == "__main__":
    main()
