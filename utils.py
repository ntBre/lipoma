import re
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from rdkit.Chem.Draw import MolsToGridImage, rdDepictor, rdMolDraw2D


@dataclass
class Record:
    """Each record has a list of equilibrium values and force constants for
    plotting, a smirks pattern, a list of molecules, and a list of
    environments. All of the lists are parallel to each other"""

    eqs: List[float]
    fcs: List[float]
    mols: List[str]
    envs: List[Tuple[int, int]]
    smirks: str
    ident: str

    @staticmethod
    def default(smirks, ident):
        return Record([], [], [], [], smirks, ident)


# adapted from ligand Molecule::to_svg
def draw_rdkit(mol, smirks, matches):
    rdmol = mol.to_rdkit()
    rdDepictor.SetPreferCoordGen(True)
    rdDepictor.Compute2DCoords(rdmol)
    rdmol = rdMolDraw2D.PrepareMolForDrawing(rdmol)
    return MolsToGridImage(
        [rdmol],
        useSVG=True,
        highlightAtomLists=[matches],
        subImgSize=(300, 300),
        molsPerRow=1,
    )


def unit(vec):
    v = np.array(vec)
    return v / np.linalg.norm(v)


def position_if(lst, f):
    "Return the first index in `lst` satisfying `f`"
    return next(i for i, x in enumerate(lst) if f(x))


def close(x, y, eps=1e-16):
    return abs(x - y) < eps


# pasted from benchmarking/parse_hist
LABEL = re.compile(r"([bati])(\d+)([a-z]*)")


def sort_label(key):
    t, n, tail = LABEL.match(key).groups()
    return (t, int(n), tail)


def make_smirks(records):
    pairs = [(smirks, r.ident) for smirks, r in records.items()]
    pairs = sorted(pairs, key=lambda pair: sort_label(pair[1]))
    return [smirks for smirks, _ in pairs]
