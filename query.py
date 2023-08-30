# asking espaloma questions about our parameters
import itertools
from collections import defaultdict

from openff.toolkit import ForceField, Molecule
from tqdm import tqdm

from main import espaloma_label, load_dataset


def molecules(ds):
    """Yields molecules from `ds` to avoid allocating big lists up front"""
    for value in ds.entries.values():
        for v in value:
            yield Molecule.from_mapped_smiles(
                v.cmiles, allow_undefined_stereo=True
            )


ff = ForceField("openff-2.1.0.offxml")
ds = load_dataset("filtered-opt.json", "optimization")

# cutoff for considering espaloma's result to be different from ours
EPS = 10.0
verbose = False

# map of smirks -> disagreement count
diffs = defaultdict(int)
for mol in tqdm(itertools.islice(molecules(ds), None)):
    labels = ff.label_molecules(mol.to_topology())[0]
    bonds = labels["Bonds"]
    # angles = labels["Angles"]
    # torsions = labels["ProperTorsions"]
    sage_bonds = {}
    for k, v in bonds.items():
        i, j = k
        # t = (i, j, v.length.magnitude, v.k.magnitude)
        sage_bonds[(i, j)] = (v.k.magnitude, v.smirks)

    _, d = espaloma_label(mol)
    espaloma_bonds = {}
    for bond in d["bonds"]:
        i, j, _, k = bond.from_zero().as_tuple()
        espaloma_bonds[(i, j)] = k

    assert espaloma_bonds.keys() == sage_bonds.keys()

    if verbose:
        print(f"{'i':>5}{'j':>5}{'Sage':>12}{'Espaloma':>12}{'Diff':>12}")
    for k, v in sage_bonds.items():
        v, smirks = v
        diff = abs(v - espaloma_bonds[k])
        i, j = k
        if diff > EPS:
            if verbose:
                print(f"{i:5}{j:5}{v:12.8}{espaloma_bonds[k]:12.8}{diff:12.8}")
            diffs[smirks] += 1


print("\nDifference Summary")
ml = max([len(s) for s in diffs.keys()])
items = [pair for pair in diffs.items()]
items.sort(key=lambda x: x[1], reverse=True)
for smirks, count in items:
    print(f"{smirks:{ml}}{count:5}")
