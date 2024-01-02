# like apply but applying everything, including all 6 torsions from espaloma. I
# tried adding this to the original apply, but it was getting too confusing.
# this is a clean start

from collections import defaultdict
from dataclasses import dataclass

import click
from openff.units import unit
from tqdm import tqdm

from query import Driver, Records, Torsions


@dataclass
class Data:
    values: dict[int, float]

    def __init__(self):
        self.envs = defaultdict(list)
        self.values = defaultdict(list)
        self.phases = defaultdict(list)


def compare(self, cls) -> Records:
    ret = defaultdict(Data)
    for m, mol in tqdm(
        enumerate(self.molecules),
        desc=f"Comparing {cls.espaloma_label}",
        total=self.total_molecules,
    ):
        # map of atom index tuple to smirks
        sage = {
            k: v.smirks
            for k, v in self.sage_labels(m, mol)[cls.sage_label].items()
        }

        espaloma = {}
        for bond in self.espaloma_labels(m, mol)[cls.espaloma_label]:
            k, v = cls.to_pair(bond)
            espaloma[k] = v

        for k, (fc, phase) in espaloma.items():
            key = tuple(list(k)[:4])
            smirks = sage[key]
            ret[smirks].envs[k[4]].append(key)
            ret[smirks].values[k[4]].append(fc)
            ret[smirks].phases[k[4]].append(phase)

    return ret


Driver.compare = compare


def mode(lst):
    "return the most common element in `lst`"
    d = defaultdict(int)
    for elt in lst:
        d[elt] += 1
    m = max(d, key=d.get)
    return m


def mean(lst):
    return sum(lst) / len(lst)


@click.command()
@click.option("--dataset", "-d", default="datasets/filtered-opt.json")
@click.option("--output", "-o", default="apple.offxml")
def main(dataset, output):
    driver = Driver(
        forcefield="openff-2.1.0.offxml",
        dataset=dataset,
        eps=0.0,
        verbose=False,
    )
    records = driver.compare(Torsions)

    th = driver.forcefield.get_parameter_handler("ProperTorsions")
    for smirks, data in records.items():
        if driver.verbose:
            print(smirks, "=>", th[smirks].id)

        th[smirks].periodicity.clear()
        th[smirks].phase.clear()
        th[smirks].k.clear()

        assert len(set(data.values.keys())) == 6
        assert len(set(data.phases.keys())) == 6
        assert len(data.values) == len(data.phases)
        assert len(data.values.values()) == len(data.phases.values())

        for k, v in data.phases.items():
            per = k
            phase = mode(v)
            fc = mean(data.values[k])
            th[smirks].periodicity.append(per)
            th[smirks].phase.append(phase * unit.radian)
            th[smirks].k.append(fc * unit.kilocalories_per_mole)
            if driver.verbose:
                print(f"\tper {per}, phase {phase}, k {fc}")

    driver.forcefield.to_file(output)


if __name__ == "__main__":
    main()
