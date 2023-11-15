# apply parameter output from parse_query to an OpenFF force field

import re

import click
from openff.toolkit import ForceField
from openff.units import unit


def load_params(filename):
    ret = []
    with open(filename, "r") as inp:
        for line in inp:
            smirks, avg, *_rest = line.split()
            ret.append((smirks, float(avg)))
    return ret


@click.command()
@click.option("--bonds", "-b")
@click.option("--bonds-eq", "-c", default=None)
@click.option("--angles", "-a")
@click.option("--angles-eq", "-d", default=None)
@click.option("--torsions", "-t")
@click.option("--output", "-o")
def main(bonds, angles, torsions, output, bonds_eq, angles_eq):
    sage = ForceField("openff-2.1.0.offxml")
    bh = sage.get_parameter_handler("Bonds")
    ah = sage.get_parameter_handler("Angles")
    th = sage.get_parameter_handler("ProperTorsions")

    for smirk, avg in load_params(bonds):
        p = bh[smirk].k
        bh[smirk].k = avg * p.units

    for smirk, avg in load_params(angles):
        p = ah[smirk].k
        ah[smirk].k = avg * p.units

    for smirk, avg in load_params(torsions):
        id_key = re.sub(r"-k[123]$", "", smirk)
        k = smirk[-2:]
        p = getattr(th[id_key], k)
        setattr(th[id_key], k, avg * p.units)

    if bonds_eq is not None:
        for smirk, avg in load_params(bonds_eq):
            p = bh[smirk].length
            bh[smirk].length = avg * p.units

    if angles_eq is not None:
        for smirk, avg in load_params(angles_eq):
            p = ah[smirk].angle
            ah[smirk].angle = (avg * unit.radians).to(unit.degrees)

    sage.to_file(output)


if __name__ == "__main__":
    main()
