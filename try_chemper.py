from collections import defaultdict

import numpy as np
from chemper.mol_toolkits import mol_toolkit
from chemper.smirksify import SMIRKSifier, print_smirks
from sklearn.mixture import GaussianMixture as model
from tqdm import tqdm

from twod import make_records, make_smirks

RECORDS = make_records("msm")
TYPE = "msm"
RECORDS = make_records(TYPE)
SMIRKS = make_smirks(RECORDS)
CUR_SMIRK = 0
NCLUSTERS = 1


def unit(vec):
    v = np.array(vec)
    return v / np.linalg.norm(v)


def make_fig(record, nclusters):
    mat = np.column_stack((unit(record.eqs), unit(record.fcs)))
    if nclusters > 1 and len(mat) > nclusters:
        m = model(n_components=nclusters).fit(mat)
        kmeans = m.predict(mat)
        colors = kmeans.astype(str)
        envs = [[env] for i, env in enumerate(record.envs)]
        mols = [
            mol_toolkit.Mol.from_smiles(s) for i, s in enumerate(record.mols)
        ]
        # according to the chemper examples, I'm trying to assemble a list of
        # tuples where the first element is a name (in this case our "color"),
        # and the second element is a list of lists of envs, one for each
        # molecule
        ucolors = set(colors)
        work = {c: [] for c in ucolors}

        for i, mol in enumerate(mols):
            work[colors[i]].append(envs[i])
            for c in ucolors:
                if c != colors[i]:
                    work[c].append([])

        work = list(work.items())
        fier = SMIRKSifier(mols, work)
        print_smirks(fier.current_smirks)
        fier.reduce()
        print_smirks(fier.current_smirks)


failed, worked = 0, 0
for record in tqdm(RECORDS.values()):
    try:
        make_fig(record, 2)
    except Exception:
        failed += 1
    else:
        worked += 1

print(failed, worked)
