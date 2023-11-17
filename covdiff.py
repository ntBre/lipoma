# report the change in coverage for two force fields
#
# this is an attempt to quantify whether changes in parameters (ordering,
# splitting) improves the quality of the force field
#
# usage: covdiff.py ff1 ff2 -d dataset.json
#  - ff1 and ff2 are passed directly to openff.toolkit.ForceField, so they can
#    be filenames or built-in force fields
#  - dataset.json should be a dataset

import logging
import warnings
from collections import defaultdict

import click
from tqdm import tqdm
from vflib import load_dataset
from vflib.coverage import ParameterType
from vflib.utils import Timer

from cluster import deduplicate_by

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    from openff.toolkit import ForceField, Molecule

logging.getLogger("openff").setLevel(logging.ERROR)


class Cover:
    def __init__(self, dataset: str):
        dataset = load_dataset(dataset)
        self.mols = deduplicate_by(
            tqdm(dataset.to_molecules(), desc="Converting molecules"),
            Molecule.to_inchikey,
        )

    # adapted from vflib
    def check(self, forcefield, parameter_type: ParameterType):
        """Check parameter coverage in `forcefield` using `dataset`."""

        print("checking coverage with")
        print(f"forcefield = {forcefield}")

        timer = Timer()

        ff = ForceField(forcefield, allow_cosmetic_attributes=True)

        timer.say("finished loading collection")

        h = ff.get_parameter_handler(parameter_type.value)
        tors_ids = [p.id for p in h.parameters]

        timer.say("finished to_records")

        results = defaultdict(int)
        for molecule in tqdm(self.mols, desc="Counting results"):
            all_labels = ff.label_molecules(molecule.to_topology())[0]
            torsions = all_labels[parameter_type.value]
            for torsion in torsions.values():
                results[torsion.id] += 1

        timer.say("finished counting results")

        got = len(results)
        want = len(tors_ids)
        pct = 100.0 * float(got) / float(want)
        print(f"{got} / {want} ({pct:.1f}%) ids covered:")

        for id in tors_ids:
            smirk = h.get_parameter(dict(id=id))[0].smirks
            print(f"{id:5}{results[id]:5}   {smirk}")

        missing_ids = [k for k in results.keys() if results[k] == 0]
        missing_smirks = [
            h.get_parameter(dict(id=p))[0].smirks for p in missing_ids
        ]
        print("\nmissing ids:")
        for i, (id, smirk) in enumerate(zip(missing_ids, missing_smirks)):
            print(f"{i:5}{id:>7}   {smirk}")

        timer.say("finished")


@click.command()
@click.argument("ff1")
@click.argument("ff2")
@click.option("--dataset", "-d")
def main(ff1, ff2, dataset):
    c = Cover(dataset)
    c.check(ff1, ParameterType.Bonds)


if __name__ == "__main__":
    main()
