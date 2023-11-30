# asking espaloma questions about our parameters

import json
import os
import re
import sys
import warnings
from collections import defaultdict

import click
from openff.toolkit import ForceField, Molecule
from tqdm import tqdm

warnings.filterwarnings("ignore")

# suppress numpy warnings
with warnings.catch_warnings():
    from vflib import load_dataset

    from cluster import deduplicate_by
    from main import espaloma_label
    from parse_query import get_parameter_map
    from record import Record


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
        for fc in ["k1", "k2", "k3", "k4", "k5", "k6"]:
            val = getattr(v, fc, None)
            if val is not None:
                per = getattr(v, f"periodicity{fc[-1]}")
                i, j, k, m = key
                sage[(i, j, k, m, per)] = (
                    val.magnitude,
                    f"{v.smirks}-{fc}",  # tag the smirks with the fc
                )


class Impropers(Torsions):
    sage_label = "ImproperTorsions"
    espaloma_label = "impropers"
    header_keys = ["i", "j", "k", "l"]


class Records(defaultdict):
    """Records is a defaultdict of smirks->Record with additional methods for
    converting to and from JSON"""

    def __init__(self, *args, **kwargs):
        super().__init__(Record, *args, **kwargs)

    def to_json(self, filename):
        with open(filename, "w") as out:
            json.dump(self, out, indent=2, default=lambda r: r.asdict())

    def from_file(filename):
        # this _cannot_ be the best way to do this, but I can't figure out the
        # right way
        ret = Records()
        with open(filename, "r") as inp:
            d = json.load(inp)
            for k, v in d.items():
                ret[k] = Record(**v)
            return ret


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
            tqdm(
                load_dataset(dataset, "optimization").to_molecules(),
                desc="Deduplicating molecules",
            ),
            Molecule.to_inchikey,
        )
        # cutoff for considering espaloma's result to be different from ours
        self.eps = eps
        self.verbose = verbose

        # caches of labels to prevent repeated labeling for multiple parameter
        # types. initialized to None and filled in on the first call to compare
        self.espaloma_labels = [None] * self.total_molecules
        self.sage_labels = [None] * self.total_molecules

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

    def compare(self, cls) -> Records:
        """Compare paramters of type `cls` assigned by `self.forcefield` and
        espaloma.

        Returns a [Records]
        """
        ids = get_parameter_map(self.forcefield)
        ret = Records()
        for m, mol in tqdm(
            enumerate(self.molecules),
            desc=f"Comparing {cls.espaloma_label}",
            total=self.total_molecules,
        ):
            if self.sage_labels[m] is None:
                self.sage_labels[m] = self.forcefield.label_molecules(
                    mol.to_topology()
                )[0]

            labels = self.sage_labels[m][cls.sage_label]
            sage = {}
            for k, v in labels.items():
                cls.insert_sage(sage, k, v)

            if self.espaloma_labels[m] is None:
                _, self.espaloma_labels[m] = espaloma_label(mol)

            d = self.espaloma_labels[m]
            espaloma = {}
            for bond in d[cls.espaloma_label]:
                k, v = cls.to_pair(bond)
                espaloma[k] = v

            if self.verbose:
                self.print_header(cls)

            # the problem is that my search for a matching smirks pattern is
            # overwriting the earlier values. I might actually have to
            # restructure this more substantially. I have the key k, which is
            # (i, j, k, l, p) where p is periodicity, but I need to know which
            # fc value the periodicity corresponds to. that's okay when the
            # value is in Sage because I can check the Sage periodicity values
            # and match up the fc values that way. When there's no Sage, value
            # I guess I just assign them numbers above what Sage does have.
            #
            # A bigger problem then is probably that I also need an idivf value
            # for each force constant, and I don't know where to get that.
            # Almost all of them in Sage are 1.0, but a couple of them are 3.0.
            # I *can* get the periodicity and phase from espaloma, but I'm not
            # sure about this idivf value. Oh, there's a default_idivf="auto"
            # tag for the ProperTorsions as a whole, so I guess I can somewhat
            # safely leave it out, or also somewhat safely assign it a value of
            # 1.0 for each of them. idivf "specifies a torsion multiplicity by
            # which the barrier height should be divided," according to the
            # SMIRNOFF standard
            for k, _ in espaloma.items():
                if (value := sage.get(k)) is None:
                    v = None
                    # v can be none, but we need to find a matching smirks by
                    # replacing the periodicity and searching again
                    for i in range(6):
                        ki = list(k[:])  # copy, I hope
                        ki[4] = i + 1
                        ki = tuple(ki)
                        if (value := sage.get(ki)) is not None:
                            _, smirks = value
                            break
                else:
                    v, smirks = value
                # smirks is actually a tagged smirks for torsions to separate
                # the k values
                id_key = re.sub(r"-k[1-6]$", "", smirks)
                if v is None:
                    diff = 0
                else:
                    diff = abs(v - espaloma[k])
                if diff >= self.eps:
                    if self.verbose:
                        self.print_row(cls, k, v, espaloma, diff)
                    # this seems to happen for impropers for some reason
                    smiles = mol.to_smiles(mapped=True)
                    ret[smirks].molecules.append(smiles)
                    ret[smirks].espaloma_values.append(espaloma[k])
                    # trim periodicity off of torsions, others should be fine
                    ret[smirks].envs.append(list(k)[:4])
                    ret[smirks].sage_value = v
                    ret[smirks].ident = ids[id_key]

        return ret


def print_summary(records: Records, outfile=None):
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
    ml = max([len(s) for s in records.keys()])
    print(
        f"# {'SMIRKS':<{ml - 2}} {'Count':>5}{'Sage':>8}{'Rest':>8}",
        file=outfile,
    )
    items = [pair for pair in records.items()]
    items.sort(key=lambda x: len(x[1].espaloma_values), reverse=True)
    for smirks, record in items:
        count = len(record.espaloma_values)
        if records[smirks].sage_value is None:
            fmt = ""
        else:
            fmt = "8.2f"
        print(
            f"{smirks:{ml}} {count:5} {records[smirks].sage_value:{fmt}}",
            end="",
            file=outfile,
        )
        for v in record.espaloma_values:
            print(f"{v:8.2f}", end="", file=outfile)
        print(file=outfile)

    if needs_close:
        outfile.close()


@click.command()
@click.option("--force-constants", "-f", is_flag=True, default=True)
@click.option("--dataset", "-d", default="datasets/filtered-opt.json")
@click.option("--out-dir", "-o", default="data/esp")
def main(force_constants, dataset, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if force_constants:
        pairs = [
            (Bonds, f"{out_dir}/bonds_dedup.dat"),
            (Angles, f"{out_dir}/angles_dedup.dat"),
            (Torsions, f"{out_dir}/torsions_dedup.dat"),
            (Impropers, f"{out_dir}/impropers_dedup.dat"),
        ]
        eps = 0.0
    else:
        pairs = [
            (BondsEq, f"{out_dir}/bonds_eq.dat"),
            (AnglesEq, f"{out_dir}/angles_eq.dat"),
        ]
        # this is too large. I might need to vary it per target (cls.eps) or
        # just set it to 0 for now
        eps = 0.0

    driver = Driver(
        forcefield="openff-2.1.0.offxml",
        dataset=dataset,
        eps=eps,
        verbose=False,
    )
    for param, outfile in pairs:
        records = driver.compare(param)
        print_summary(records, outfile=outfile)

        js = os.path.splitext(outfile)[0]
        records.to_json(f"{js}.json")


if __name__ == "__main__":
    main()
