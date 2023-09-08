# apply parameter output from parse_query to an OpenFF force field

from openff.toolkit import ForceField


def load_params(filename):
    ret = []
    with open(filename, "r") as inp:
        for line in inp:
            smirks, avg = line.split()
            ret.append((smirks, float(avg)))
    return ret


sage = ForceField("openff-2.1.0.offxml")
bh = sage.get_parameter_handler("Bonds")
ah = sage.get_parameter_handler("Angles")
th = sage.get_parameter_handler("ProperTorsions")

for smirk, avg in load_params("bonds/output.dat"):
    p = bh[smirk].k
    bh[smirk].k = avg * p.units

for smirk, avg in load_params("angles/output.dat"):
    p = ah[smirk].k
    ah[smirk].k = avg * p.units

for smirk, avg in load_params("torsions/output.dat"):
    p = th[smirk].k1
    th[smirk].k1 = avg * p.units

sage.to_file("out.offxml")
