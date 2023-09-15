# asking espaloma questions about our parameters

import sys
import warnings
from collections import defaultdict
from typing import Tuple

warnings.filterwarnings("ignore")

from openff.toolkit import ForceField, Molecule  # noqa: E402
from tqdm import tqdm  # noqa: E402
from vflib import load_dataset  # noqa: E402

from cluster import deduplicate_by  # noqa: E402
from main import espaloma_label  # noqa: E402


# there's probably a better name for this
class Driver:
    def __init__(
        self,
        forcefield: str,
        dataset: str,
        eps: float = 10.0,
        verbose: bool = False,
    ):
        self.forcefield = ForceField(forcefield)
        # demand execution so I can deduplicate
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

    def compare_bonds(self) -> Tuple[dict[str, list[float]], dict[str, float]]:
        """Compare bond paramters assigned by ff and espaloma.

        Returns a map of smirks->[espaloma values], and a map of
        smirks->sage_value
        """
        sage_values = {}
        # map of smirks -> disagreement count
        diffs = defaultdict(list)
        for mol in tqdm(
            self.molecules,
            desc="Comparing bonds",
            total=self.total_molecules,
        ):
            labels = self.forcefield.label_molecules(mol.to_topology())[0]
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

            if self.verbose:
                print(
                    f"{'i':>5}{'j':>5}{'Sage':>12}{'Espaloma':>12}{'Diff':>12}"
                )
            for k, v in sage_bonds.items():
                v, smirks = v
                diff = abs(v - espaloma[k])
                if diff > self.eps:
                    if self.verbose:
                        i, j = k
                        print(
                            f"{i:5}{j:5}{v:12.8}{espaloma[k]:12.8}{diff:12.8}"
                        )
                    diffs[smirks].append(espaloma[k])
                    sage_values[smirks] = v

        return diffs, sage_values

    # this is basically a copy pasta from compare_bonds. it would be nice to
    # factor out some commonality, but many of the internals are different
    def compare_angles(
        self,
    ) -> Tuple[dict[str, list[float]], dict[str, float]]:
        """Compare angle paramters assigned by ff and espaloma.

        Returns a map of smirks->[espaloma values], and a map of
        smirks->sage_value
        """
        sage_values = {}
        diffs = defaultdict(list)
        for mol in tqdm(
            self.molecules,
            desc="Comparing angles",
            total=self.total_molecules,
        ):
            labels = self.forcefield.label_molecules(mol.to_topology())[0]
            angles = labels["Angles"]
            sage_angles = {}
            for key, v in angles.items():
                sage_angles[key] = (v.k.magnitude, v.smirks)

            _, d = espaloma_label(mol)
            espaloma = {}
            for angle in d["angles"]:
                i, j, k, _, key = angle.from_zero().as_tuple()
                espaloma[(i, j, k)] = key

            assert espaloma.keys() == sage_angles.keys()

            for key, v in sage_angles.items():
                v, smirks = v
                diff = abs(v - espaloma[key])
                if diff > self.eps:
                    diffs[smirks].append(espaloma[key])
                    sage_values[smirks] = v

        return diffs, sage_values

    # this is basically a copy pasta from compare_bonds. it would be nice to
    # factor out some commonality, but many of the internals are different
    def compare_torsions(
        self,
    ) -> Tuple[dict[str, list[float]], dict[str, float]]:
        """Compare proper torsion paramters assigned by ff and espaloma.

        Returns a map of smirks->[espaloma values], and a map of
        smirks->sage_value
        """
        sage_values = {}
        diffs = defaultdict(list)
        for mol in tqdm(
            self.molecules,
            desc="Comparing torsions",
            total=self.total_molecules,
        ):
            labels = self.forcefield.label_molecules(mol.to_topology())[0]
            torsions = labels["ProperTorsions"]
            sage_torsions = {}
            for key, v in torsions.items():
                for fc in ["k1", "k2", "k3"]:
                    val = getattr(v, fc, None)
                    if val is not None:
                        per = getattr(v, f"periodicity{fc[-1]}")
                        i, j, k, m = key
                        sage_torsions[(i, j, k, m, per)] = (
                            val.magnitude,
                            v.smirks,
                        )

            _, d = espaloma_label(mol)
            espaloma = {}
            for torsion in d["torsions"]:
                i, j, k, m, _per, _phase, fc = torsion.from_zero().as_tuple()
                espaloma[(i, j, k, m, _per)] = fc

            # espaloma does something funny with the
            # torsions/periodicity/phase, so just take the keys that correspond
            # to values in sage
            espaloma = {
                k: v for k, v in espaloma.items() if k in sage_torsions
            }
            ekeys = espaloma.keys()
            skeys = sage_torsions.keys()
            assert len(ekeys) == len(
                skeys
            ), f"# espaloma keys ({len(ekeys)}) != # sage keys ({len(skeys)})"
            assert ekeys == skeys

            for key, v in sage_torsions.items():
                v, smirks = v
                diff = abs(v - espaloma[key])
                if diff > self.eps:
                    diffs[smirks].append(espaloma[key])
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

# another idea, which I'm pursuing right now, is just to get all of the average
# values from espaloma and put them into a force field. then I'll just
# benchmark this force field and see if espaloma does a better job than all of
# our optimizations

if __name__ == "__main__":
    driver = Driver(
        forcefield="openff-2.1.0.offxml",
        dataset="filtered-opt.json",
        eps=10.0,
        verbose=False,
    )
    diffs, sage_values = driver.compare_bonds()
    print_summary(diffs, sage_values, outfile="bonds_dedup.dat")

    diffs, sage_values = driver.compare_angles()
    print_summary(diffs, sage_values, outfile="angles_dedup.dat")

    diffs, sage_values = driver.compare_torsions()
    print_summary(diffs, sage_values, outfile="torsions_dedup.dat")
