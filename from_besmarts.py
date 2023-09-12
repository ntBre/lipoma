# loading raw output from besmarts and turning it into a force field

import re

from openff.toolkit import ForceField
from openff.units import unit


def get_last_tree(filename):
    """Get the final parameter tree from `filename`.

    Trees are delimited at the start by a line beginning with `Tree:` and at
    the end by a line beginning with `=====` (5 =)

    """
    in_tree = False
    tree_lines = []
    with open(filename, "r") as inp:
        for line in inp:
            if line.startswith("Tree:"):
                tree_lines = []
                in_tree = True
            elif in_tree:
                if line.startswith("====="):
                    in_tree = False
                    continue
                tree_lines.append(line)
    return tree_lines


def parse_tree(tree):
    """Parse a parameter tree from besmarts into a sequence of SMARTS, mean
    pairs.

    """
    pairs = []
    # looking for untagged SMARTS like "[!r5]~[*]"
    pat = re.compile(r"\[([^\]:]+?)\]")
    for line in tree:
        sp = line.split()
        # this is the mean bond length
        mean = float(sp[5])
        smarts = sp[14]
        c = 1
        while re.search(pat, smarts):
            smarts = re.sub(pat, f"[\\1:{c}]", smarts, count=1)
            c += 1

        pairs.append((smarts, mean))
    return pairs


tree = get_last_tree("espaloma_bonds.log")
pairs = parse_tree(tree)

ff = ForceField("openff-2.1.0.offxml")
bh = ff.get_parameter_handler("Bonds")
bh.parameters.clear()

for i, (smirks, mean) in enumerate(pairs):
    kcal = unit.kilocalorie / unit.mole / unit.angstrom**2
    bh.add_parameter(
        dict(
            smirks=smirks,
            length=mean * unit.angstrom,
            k=1.0 * kcal,
            id=f"b{i}",
        )
    )

print(len(ff.get_parameter_handler("Bonds").parameters))
print(ff.to_string())

# okay I get it now, Trevor said to "[label] each bond/angle, then [use] the
# average of what matched to each parameter." so I first need to generate some
# pseudo-parameters in a force field, use that force field to label_molecules,
# and then take the average back from my espaloma parameters? I guess I should
# be able to look up in the espaloma "table" a value for a given (molecule,
# atom_i, atom_j) key. I might still not get it

# after having to look at the msm script, I think it actually handles this and
# sounds like what Trevor was saying, so I can safely plop these into a force
# field with a dummy value and run msm on it. just double check that it comes
# out okay
