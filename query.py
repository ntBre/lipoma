# asking espaloma questions about our parameters

import itertools
import sys
from collections import defaultdict
from typing import Tuple

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
total = sum(len(s) for s in ds.entries.values())

# cutoff for considering espaloma's result to be different from ours
EPS = 10.0
verbose = False


def compare_bonds() -> Tuple[dict[str, list[float]], dict[str, float]]:
    """Compare bond paramters assigned by ff and espaloma.

    Returns a map of smirks->[espaloma values], and a map of smirks->sage_value
    """
    sage_values = {}
    # map of smirks -> disagreement count
    diffs = defaultdict(list)
    for mol in tqdm(
        itertools.islice(molecules(ds), None),
        desc="Comparing bonds",
        total=total,
    ):
        labels = ff.label_molecules(mol.to_topology())[0]
        # angles = labels["Angles"]
        # torsions = labels["ProperTorsions"]

        bonds = labels["Bonds"]
        sage_bonds = {}
        for k, v in bonds.items():
            i, j = k
            # t = (i, j, v.length.magnitude, v.k.magnitude)
            sage_bonds[(i, j)] = (v.k.magnitude, v.smirks)

        _, d = espaloma_label(mol)
        espaloma = {}
        for bond in d["bonds"]:
            i, j, _, k = bond.from_zero().as_tuple()
            espaloma[(i, j)] = k

        assert espaloma.keys() == sage_bonds.keys()

        if verbose:
            print(f"{'i':>5}{'j':>5}{'Sage':>12}{'Espaloma':>12}{'Diff':>12}")
        for k, v in sage_bonds.items():
            v, smirks = v
            diff = abs(v - espaloma[k])
            i, j = k
            if diff > EPS:
                if verbose:
                    print(f"{i:5}{j:5}{v:12.8}{espaloma[k]:12.8}{diff:12.8}")
                diffs[smirks].append(espaloma[k])
                sage_values[smirks] = v

    return diffs, sage_values


def print_summary(diffs, sage_values, outfile=None):
    """Print a summary of diffs and sage_values to `outfile` or stdout if None.

    The output format is `SMIRKS Count Sage Rest`, where Rest is all of the
    espaloma values for a given SMIRKS pattern
    """
    needs_close = False
    if outfile is None:
        outfile = sys.stdout
    elif isinstance(outfile, str):
        outfile = open(outfile, "w")
        needs_close = True

    print("# Difference Summary", file=outfile)
    # compute the max len of smirks patterns for pretty printing
    ml = max([len(s) for s in diffs.keys()])
    print(
        f"# {'SMIRKS':<{ml - 2}}{'Count':>5}{'Sage':>8}{'Rest':>8}",
        file=outfile,
    )
    items = [pair for pair in diffs.items()]
    items.sort(key=lambda x: len(x[1]), reverse=True)
    for smirks, values in items:
        count = len(values)
        print(
            f"{smirks:{ml}}{count:5}{sage_values[smirks]:8.2f}",
            end="",
            file=outfile,
        )
        for v in values:
            print(f"{v:8.2f}", end="", file=outfile)
        print(file=outfile)

    if needs_close:
        outfile.close()


# counting occurences of disagreement is somewhat interesting, but more useful
# might be recording the espaloma values that disagree. then I could do some
# kind of statistics on that. maybe our parameter is just the average of the
# espaloma parameters, for example. if not, maybe we need to shift our
# parameter toward the average espaloma value. or if espaloma has an especially
# large range of values, that would be an indicator that we need to break up
# one of our parameters
diffs, sage_values = compare_bonds()
print_summary(diffs, sage_values, outfile="bonds.dat")
