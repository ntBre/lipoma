# like cluster, but instead of focusing on the "outliers" also check the main
# group

from collections import defaultdict

from openff.qcsubmit.results import OptimizationResultCollection
from openff.toolkit import Molecule
from tqdm import tqdm

from cluster import deduplicate_by, load_thresh
from main import espaloma_label


def partition():
    smirkss = load_thresh("data/labeled_bonds.dat")

    opt = OptimizationResultCollection.parse_file("datasets/filtered-opt.json")

    molecules = deduplicate_by(
        tqdm(opt.to_molecules(), desc="Converting molecules"),
        Molecule.to_inchikey,
    )

    # molecules matching and not matching the fn returned by load_thresh
    in_thresh = defaultdict(list)
    out_thresh = defaultdict(list)
    for s, (smirks, fn) in enumerate(smirkss):
        for m, molecule in tqdm(
            enumerate(molecules),
            total=len(molecules),
            desc=f"Processing smirks {s}",
        ):
            matches = molecule.chemical_environment_matches(smirks)
            if matches:
                _, esp = espaloma_label(molecule)
                esp = [
                    (b.atom1 - 1, b.atom2 - 1)
                    for b in esp["bonds"]
                    if (b.atom1 - 1, b.atom2 - 1) in matches and fn(b.k)
                ]
                if esp:
                    in_thresh[smirks].append(molecule)
                else:
                    out_thresh[smirks].append(molecule)
        break  # only checking b84 for now

    return in_thresh, out_thresh


if __name__ == "__main__":
    partition()
