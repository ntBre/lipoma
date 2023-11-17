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

from cluster import deduplicate_by

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    from openff.toolkit import ForceField, Molecule

logging.getLogger("openff").setLevel(logging.ERROR)


class Cover:
    def __init__(self, dataset: str):
        dataset = load_dataset(dataset)
        mols = deduplicate_by(
            tqdm(dataset.to_molecules(), desc="Converting molecules"),
            Molecule.to_inchikey,
        )
        # this only takes 1 second, so very minor savings. not a big deal to
        # omit this if we ever need the molecules themselves
        self.mols = [
            m.to_topology()
            for m in tqdm(mols, desc="Converting to topologies")
        ]

    def check(self, ff1, ff2):
        r1, i1, s1 = self.check_one(ff1)
        r2, i2, s2 = self.check_one(ff2)

        # there are several cases to consider when reporting the results:
        # 1. id is present in both -> check if count differs
        # 2. id present only in r1
        # 3. id present only in r2
        # 4. the ids are the same but smirks differ
        #
        # iterating over one of them makes it difficult to identify ids only in
        # the other set, especially if we want to keep the order recognizable
        #
        # I really want to produce something that looks like a diff, with
        # insertions and deletions shown in the right order
        #
        # case 4 is also complicated. I think the easiest fix will be treating
        # these as separate parameters. maybe we essentially ignore ids
        # generally

        print()

        for i in i1:
            if i in i2 and s1[i] == s2[i] and r1[i] != r2[i]:
                # present in both with same smirks
                smirk = s1[i]
                print(f"  {i:5}{r2[i] - r1[i]:+5}   {smirk}")
            elif i not in i2:
                # next easiest case, only present in i1
                print(f"- {i:5}{-r1[i]:+5}   {smirk}")

        for i in i2:
            if i not in i1:
                # third easiest case, only present in i2
                smirk = s2[i]
                print(f"+ {i:5}{r2[i]:+5}   {smirk}")

        # this is a decent start, but it might be more interesting to see where
        # the matches went, not just the change

    # adapted from vflib
    def check_one(
        self, forcefield: str
    ) -> tuple[dict[str, int], list[str], dict[str, str]]:
        """Check parameter coverage in `forcefield` using `self.mols`.

        Returns the main results dict mapping parameter ids to coverage counts,
        a list of ids in `forcefield`, and a map of parameter id to parameter
        smirks for printing.

        """

        ff = ForceField(forcefield, allow_cosmetic_attributes=True)

        ptypes = ["Bonds", "Angles", "ProperTorsions", "ImproperTorsions"]

        ids = []
        smirks = {}
        for ptype in ptypes:
            h = ff.get_parameter_handler(ptype)
            ids.extend((p.id for p in h.parameters))
            smirks.update({p.id: p.smirks for p in h.parameters})

        results = defaultdict(int)
        for molecule in tqdm(self.mols, desc="Counting results"):
            all_labels = ff.label_molecules(molecule)[0]
            for ptype in ptypes:
                ps = all_labels[ptype]
                for p in ps.values():
                    results[p.id] += 1

        return results, ids, smirks

    def print_results(self, results, ids, smirks):
        got = len(results)
        want = len(ids)
        pct = 100.0 * float(got) / float(want)
        print(f"{got} / {want} ({pct:.1f}%) ids covered:")

        for id in ids:
            smirk = smirks[id]
            print(f"{id:5}{results[id]:5}   {smirk}")

        missing_ids = [k for k in results.keys() if results[k] == 0]
        missing_smirks = [smirks[p] for p in missing_ids]
        print("\nmissing ids:")
        for i, (id, smirk) in enumerate(zip(missing_ids, missing_smirks)):
            print(f"{i:5}{id:>7}   {smirk}")


@click.command()
@click.argument("ff1")
@click.argument("ff2")
@click.option("--dataset", "-d")
def main(ff1, ff2, dataset):
    c = Cover(dataset)
    c.check(ff1, ff2)


if __name__ == "__main__":
    main()
