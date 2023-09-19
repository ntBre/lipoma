# like cluster, but instead of focusing on the "outliers" also check the main
# group

from collections import defaultdict, namedtuple

from openff.qcsubmit.results import OptimizationResultCollection
from openff.toolkit import Molecule
from tqdm import tqdm

from cluster import deduplicate_by, load_thresh
from main import espaloma_label

Data = namedtuple("Data", "molecule matches")


def inner(molecules, smirks, fn, s=None):
    msg = f"Processing smirks {s}"
    if s is None:
        msg = "Processing smirks"
    in_thresh = []
    out_thresh = []
    for m, molecule in tqdm(
        enumerate(molecules),
        total=len(molecules),
        desc=msg,
    ):
        matches = molecule.chemical_environment_matches(smirks)
        if matches:
            _, esp = espaloma_label(molecule, types=["bonds"])
            esp_in, esp_out = [], []
            for b in esp["bonds"]:
                idx = (b.atom1 - 1, b.atom2 - 1)
                if idx in matches:
                    if fn(b.k):
                        esp_in.append(idx)
                    else:
                        esp_out.append(idx)
            if esp_in:
                in_thresh.append(Data(molecule, esp_in))
            if esp_out:
                out_thresh.append(Data(molecule, esp_out))
    return in_thresh, out_thresh


def partition():
    """Partition the input molecules into sets matching the threshold function
    and not matching the function for each smirks pattern

    """
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
        i, o = inner(molecules, smirks, fn, s)
        in_thresh[smirks] = i
        out_thresh[smirks] = o
        break  # only checking b84 for now

    return in_thresh, out_thresh


def count(set_, pat):
    """Count the number of chemical environment matches in the molecule `set_`
    for the given smirks `pat`"""
    tot = 0
    for molecule, matches in set_:
        mat = molecule.chemical_environment_matches(pat)
        smat, smatches = set(mat), set(matches)
        # check that the very specific matches detected above are a subset of
        # the current matches
        if mat and (smatches <= smat):
            tot += 1
    return tot


# okay, this isn't enough. all I can do with this is check that the chemical
# environment of the partitioned molecules matches or doesn't match some
# pattern, but I need to know what I'm checking in the loop: it matches the
# pattern AND the specific pairs that are above/below the threshold

if __name__ == "__main__":
    partition()
