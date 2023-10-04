# loading raw output from besmarts and turning it into a force field

import logging
import re
from collections import defaultdict

import numpy as np
import vflib  # noqa: F401
from openff.qcsubmit.results import OptimizationResultCollection
from openff.toolkit import ForceField
from openff.units import unit
from tqdm import tqdm

logging.getLogger("openff").setLevel(logging.ERROR)


def get_last_tree(filename):
    """Get the final parameter tree from `filename`.

    Trees are delimited at the start by a line beginning with `Tree:` and at
    the end by a line beginning with `=====` (5 =)

    """
    in_tree = False
    tree_lines = []
    with open(filename, "r") as inp:
        for line in inp:
            if line.startswith("Tree:"):
                tree_lines = []
                in_tree = True
            elif in_tree:
                if line.startswith("====="):
                    in_tree = False
                    continue
                tree_lines.append(line)
    return tree_lines


def parse_tree(tree):
    """Parse a parameter tree from besmarts into a sequence of SMARTS, mean
    pairs.

    """
    pairs = []
    # looking for untagged SMARTS like "[!r5]~[*]"
    pat = re.compile(r"\[([^\]:]+?)\]")
    for line in tree:
        sp = line.split()
        # this is the mean bond length
        mean = float(sp[5])
        smarts = sp[14]
        c = 1
        while re.search(pat, smarts):
            smarts = re.sub(pat, f"[\\1:{c}]", smarts, count=1)
            c += 1

        pairs.append((smarts, mean))
    return pairs


def initial_force_field():
    """Generate an initial force field from the besmarts patterns and their
    corresponding average bond lengths and angles. The force constants are
    initialized to 1.0

    """
    bond_tree = get_last_tree("espaloma_bonds.log")
    angle_tree = get_last_tree("espaloma_angles.log")

    ff = ForceField("openff-2.1.0.offxml")
    bh = ff.get_parameter_handler("Bonds")
    ah = ff.get_parameter_handler("Angles")
    bh.parameters.clear()
    ah.parameters.clear()

    for i, (smirks, mean) in enumerate(parse_tree(bond_tree)):
        kcal = unit.kilocalorie / unit.mole / unit.angstrom**2
        bh.add_parameter(
            dict(
                smirks=smirks,
                length=mean * unit.angstrom,
                k=1.0 * kcal,
                id=f"b{i}",
            )
        )

    for i, (smirks, mean) in enumerate(parse_tree(angle_tree)):
        kcal = unit.kilocalorie / unit.mole / unit.radian**2
        ah.add_parameter(
            dict(
                smirks=smirks,
                angle=mean * unit.degree,
                k=1.0 * kcal,
                id=f"a{i}",
            )
        )
    return ff


esp_init = initial_force_field()
sage = ForceField("openff-2.1.0.offxml")
opt = OptimizationResultCollection.parse_file("filtered-opt.json")

esp_bonds = defaultdict(list)
esp_angles = defaultdict(list)
molecules = opt.to_molecules()
for m, mol in tqdm(
    enumerate(molecules), total=len(molecules), desc="Labeling molecules"
):
    top = mol.to_topology()
    esp = esp_init.label_molecules(top)[0]
    sag = sage.label_molecules(top)[0]

    for (i, j), v in esp["Bonds"].items():
        esp_bonds[v.smirks].append(sag["Bonds"][(i, j)].k.magnitude)

    for (i, j, k), v in esp["Angles"].items():
        esp_angles[v.smirks].append(sag["Angles"][(i, j, k)].k.magnitude)

bh = esp_init.get_parameter_handler("Bonds")
for smirks, values in esp_bonds.items():
    k = np.mean(values)
    bh[smirks].k = k * bh[smirks].k.units

ah = esp_init.get_parameter_handler("Angles")
for smirks, values in esp_angles.items():
    k = np.mean(values)
    ah[smirks].k = k * ah[smirks].k.units

esp_init.to_file("esp_ba.offxml")
