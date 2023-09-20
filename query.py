# asking espaloma questions about our parameters

import sys
from collections import defaultdict
from typing import Tuple

import click
from openff.toolkit import ForceField, Molecule
from tqdm import tqdm
from vflib import load_dataset

from cluster import deduplicate_by
from main import espaloma_label


class BondsEq:
    sage_label = "Bonds"
    espaloma_label = "bonds"
    header_keys = ["i", "j"]

    def to_pair(bond):
        i, j, k, _ = bond.from_zero().as_tuple()
        return (i, j), k

    def insert_sage(sage, k, v):
        sage[k] = (v.length.magnitude, v.smirks)


class AnglesEq:
    sage_label = "Angles"
    espaloma_label = "angles"
    header_keys = ["i", "j", "k"]

    def to_pair(angle):
        i, j, k, key, _ = angle.from_zero().as_tuple()
        return (i, j, k), key

    def insert_sage(sage, k, v):
        sage[k] = (v.angle.to("radians").magnitude, v.smirks)


class Bonds:
    sage_label = "Bonds"
    espaloma_label = "bonds"
    header_keys = ["i", "j"]

    def to_pair(bond):
        i, j, _, k = bond.from_zero().as_tuple()
        return (i, j), k

    def insert_sage(sage, k, v):
        sage[k] = (v.k.magnitude, v.smirks)


class Angles:
    sage_label = "Angles"
    espaloma_label = "angles"
    header_keys = ["i", "j", "k"]

    def to_pair(angle):
        i, j, k, _, key = angle.from_zero().as_tuple()
        return (i, j, k), key

    def insert_sage(sage, k, v):
        sage[k] = (v.k.magnitude, v.smirks)


class Torsions:
    sage_label = "ProperTorsions"
    espaloma_label = "torsions"
    header_keys = ["i", "j", "k", "l"]

    def to_pair(torsion):
        i, j, k, m, per, _phase, fc = torsion.from_zero().as_tuple()
        return (i, j, k, m, per), fc

    def insert_sage(sage, key, v):
        for fc in ["k1", "k2", "k3"]:
            val = getattr(v, fc, None)
            if val is not None:
                per = getattr(v, f"periodicity{fc[-1]}")
                i, j, k, m = key
                sage[(i, j, k, m, per)] = (
                    val.magnitude,
                    v.smirks,
                )

    def fix_keys(espaloma, sage):
        return {k: v for k, v in espaloma.items() if k in sage}


class Driver:
    def __init__(
        self,
        forcefield: str,
        dataset: str,
        eps: float = 10.0,
        verbose: bool = False,
    ):
        self.forcefield = ForceField(forcefield)
        self.molecules = deduplicate_by(
            load_dataset(dataset, "optimization").to_molecules(),
            Molecule.to_inchikey,
        )
        # cutoff for considering espaloma's result to be different from ours
        self.eps = eps
        self.verbose = verbose

    @property
    def total_molecules(self):
        return len(self.molecules)

    def print_header(self, cls):
        cls.print_header()
        for h in cls.header_keys:
            print(f"{h:>5}", end="")
        print(f"{'Sage':>12}{'Espaloma':>12}{'Diff':>12}")

    def print_row(self, cls, k, v, espaloma, diff):
        for elt in k:
            print(f"{elt:5}", end="")
        print(f"{v:12.8}{espaloma[k]:12.8}{diff:12.8}")

    def compare(self, cls) -> Tuple[dict[str, list[float]], dict[str, float]]:
        """Compare paramters of type `cls` assigned by `self.forcefield` and
        espaloma.

        Returns a map of smirks->[espaloma values], and a map of
        smirks->sage_value

        """
        sage_values = {}
        # map of smirks -> disagreement count
        diffs = defaultdict(list)
        for mol in tqdm(
            self.molecules,
            desc=f"Comparing {cls.espaloma_label}",
            total=self.total_molecules,
        ):
            labels = self.forcefield.label_molecules(mol.to_topology())[0]
            labels = labels[cls.sage_label]
            sage = {}
            for k, v in labels.items():
                cls.insert_sage(sage, k, v)

            _, d = espaloma_label(mol)
            espaloma = {}
            for bond in d[cls.espaloma_label]:
                k, v = cls.to_pair(bond)
                espaloma[k] = v

            # needed for torsions
            if hasattr(cls, "fix_keys"):
                espaloma = cls.fix_keys(espaloma, sage)

            assert espaloma.keys() == sage.keys()

            if self.verbose:
                self.print_header(cls)

            for k, v in sage.items():
                v, smirks = v
                diff = abs(v - espaloma[k])
                if diff > self.eps:
                    if self.verbose:
                        self.print_row(cls, k, v, espaloma, diff)
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


@click.command()
@click.option("--force-constants", "-f", is_flag=True, default=True)
def main(force_constants):
    if force_constants:
        pairs = [
            (Bonds, "data/bonds_dedup.dat"),
            (Angles, "data/angles_dedup.dat"),
            (Torsions, "data/torsions_dedup.dat"),
        ]
        eps = 10.0
    else:
        pairs = [
            (BondsEq, "data/bonds_eq.dat"),
            (AnglesEq, "data/angles_eq.dat"),
        ]
        # this is too large. I might need to vary it per target (cls.eps) or
        # just set it to 0 for now
        eps = 0.1

    driver = Driver(
        forcefield="openff-2.1.0.offxml",
        dataset="datasets/filtered-opt.json",
        eps=eps,
        verbose=False,
    )
    for param, outfile in pairs:
        diffs, sage_values = driver.compare(param)
        print_summary(diffs, sage_values, outfile=outfile)


if __name__ == "__main__":
    main()
