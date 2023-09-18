# like cluster, but instead of focusing on the "outliers" also check the main
# group

from openff.qcsubmit.results import OptimizationResultCollection
from openff.toolkit import Molecule
from tqdm import tqdm

from cluster import deduplicate_by, load_thresh
from main import espaloma_label


def partition():
    smirkss = load_thresh("data/labeled_bonds.dat")

    opt = OptimizationResultCollection.parse_file("datasets/filtered-opt.json")

    # demand execution here so I can deduplicate and reuse molecules
    molecules = [
        m for m in tqdm(opt.to_molecules(), desc="Converting molecules")
    ]
    print(len(molecules), " initially")
    molecules = deduplicate_by(molecules, Molecule.to_inchikey)
    print(len(molecules), " after dedup")

    # TODO make these entries in a dict over smirks

    # molecules matching the fn returned by load_thresh
    in_thresh = []
    # molecules not matching the fn returned by load_thresh
    out_thresh = []
    for s, (smirks, fn) in tqdm(
        enumerate(smirkss), total=len(smirkss), desc="Processing smirks"
    ):
        for m, molecule in enumerate(molecules):
            matches = molecule.chemical_environment_matches(smirks)
            if matches:
                _, esp = espaloma_label(molecule)
                esp = [
                    (b.atom1 - 1, b.atom2 - 1)
                    for b in esp["bonds"]
                    if (b.atom1 - 1, b.atom2 - 1) in matches and fn(b.k)
                ]
                if esp:
                    in_thresh.append(molecule)
                else:
                    out_thresh.append(molecule)
        break  # only checking b84 for now

    return in_thresh, out_thresh


if __name__ == "__main__":
    partition()
