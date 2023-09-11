# loading raw output from besmarts and turning it into a force field


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
    for line in tree:
        print(line, end="")
        sp = line.split()
        # this is the mean bond length
        mean = float(sp[5])
        print(f"\t{mean}")


tree = get_last_tree("espaloma_bonds.log")
parse_tree(tree)
