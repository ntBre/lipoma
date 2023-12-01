# like apply but applying everything, including all 6 torsions from espaloma. I
# tried adding this to the original apply, but it was getting too confusing.
# this is a clean start

import re

import click
from tqdm import tqdm

from query import Driver, Records, Torsions, get_parameter_map


# the problem is that my search for a matching smirks pattern is overwriting
# the earlier values. I might actually have to restructure this more
# substantially. I have the key k, which is (i, j, k, l, p) where p is
# periodicity, but I need to know which fc value the periodicity corresponds
# to. that's okay when the value is in Sage because I can check the Sage
# periodicity values and match up the fc values that way. When there's no Sage,
# value I guess I just assign them numbers above what Sage does have.
#
# A bigger problem then is probably that I also need an idivf value for each
# force constant, and I don't know where to get that. Almost all of them in
# Sage are 1.0, but a couple of them are 3.0. I *can* get the periodicity and
# phase from espaloma, but I'm not sure about this idivf value. Oh, there's a
# default_idivf="auto" tag for the ProperTorsions as a whole, so I guess I can
# somewhat safely leave it out, or also somewhat safely assign it a value of
# 1.0 for each of them. idivf "specifies a torsion multiplicity by which the
# barrier height should be divided," according to the SMIRNOFF standard
def compare(self, cls) -> Records:
    ids = get_parameter_map(self.forcefield)
    ret = Records()
    for m, mol in tqdm(
        enumerate(self.molecules),
        desc=f"Comparing {cls.espaloma_label}",
        total=self.total_molecules,
    ):
        sage = {}
        for k, v in self.sage_labels(m, mol)[cls.sage_label].items():
            cls.insert_sage(sage, k, v)

        espaloma = {}
        for bond in self.espaloma_labels(m, mol)[cls.espaloma_label]:
            k, v = cls.to_pair(bond)
            espaloma[k] = v

        for k, lock in espaloma.items():
            if isinstance(lock, tuple):
                fc, phase = lock  # should be the case for torsions
            else:
                fc = lock
            if (value := sage.get(k)) is None:
                v = None
                # v can be none, but we need to find a matching smirks by
                # replacing the periodicity and searching again
                found = False
                pr = k[4]
                for i in range(6):
                    ki = list(k[:])  # copy, I hope
                    ki[4] = i + 1
                    ki = tuple(ki)
                    if (value := sage.get(ki)) is not None:
                        _, smirks = value
                        found = True
                        break
                if not found:
                    raise ValueError(f"couldn't find {k} in Sage")
            else:
                v, smirks = value

            # decorate the smirks with the periodicity and phase angle in
            # radians
            smirks += f"-pr{pr}-p{phase}"

            # smirks is actually a tagged smirks for torsions to separate
            # the k values
            id_key = re.sub(r"-k[1-6](-pr[1-6])(-p[-0-9.]+)?$", "", smirks)
            smiles = mol.to_smiles(mapped=True)
            ret[smirks].molecules.append(smiles)
            ret[smirks].espaloma_values.append(fc)
            # trim periodicity off of torsions, others should be fine
            ret[smirks].envs.append(list(k)[:4])
            ret[smirks].sage_value = v
            ret[smirks].ident = ids[id_key]

    return ret


Driver.compare = compare


@click.command()
@click.option("--dataset", "-d", default="datasets/filtered-opt.json")
def main(dataset):
    driver = Driver(
        forcefield="openff-2.1.0.offxml",
        dataset=dataset,
        eps=0.0,
        verbose=False,
    )
    records = driver.compare(Torsions)
    for smirks in records.keys():
        print(smirks)


if __name__ == "__main__":
    main()
