import json
import logging
import warnings
from collections import defaultdict
from multiprocessing import Pool
from typing import Union

from openff.qcsubmit.results import (
    OptimizationResultCollection,
    TorsionDriveResultCollection,
)
from openff.toolkit.topology import Molecule
from openff.units.openmm import from_openmm
from openmm.openmm import HarmonicBondForce
from tqdm import tqdm

import espaloma as esp

warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("openff").setLevel(logging.ERROR)


# copy pasta from known-issues main.py
def load_dataset(
    dataset: str,
    typ: str = None,
) -> Union[OptimizationResultCollection, TorsionDriveResultCollection]:
    """Peeks at the first entry of `dataset` to determine its type and
    then loads it appropriately. If the `typ` argument is supplied,
    treat that as the type instead.

    Raises a `TypeError` if the first entry is neither a `torsion`
    record nor an `optimization` record.
    """
    if typ is None:
        with open(dataset, "r") as f:
            j = json.load(f)
        entries = j["entries"]
        keys = entries.keys()
        assert len(keys) == 1  # only handling this case for now
        key = list(keys)[0]
        typ = j["entries"][key][0]["type"]
    match typ:
        case "torsion":
            return TorsionDriveResultCollection.parse_file(dataset)
        case "optimization":
            return OptimizationResultCollection.parse_file(dataset)
        case t:
            raise TypeError(f"Unknown result collection type: {t}")


def inner(molecule):
    mapped_smiles = molecule.to_smiles(mapped=True)

    # create an Espaloma Graph object to represent the molecule of interest
    molecule_graph = esp.Graph(molecule)

    # load pretrained model
    espaloma_model = esp.get_model("latest")

    # apply a trained espaloma model to assign parameters
    espaloma_model(molecule_graph.heterograph)

    # create an OpenMM System for the specified molecule
    forcefield = "openff_unconstrained-2.1.0"
    openmm_system = esp.graphs.deploy.openmm_system_from_graph(
        molecule_graph, forcefield=forcefield
    )

    d = list()
    # hopefully these indices match the mapped_smiles...
    for force in openmm_system.getForces():
        if isinstance(force, HarmonicBondForce):
            for b in range(force.getNumBonds()):
                # ignore the force constant for now
                i, j, eq, _k = force.getBondParameters(b)
                # convert from openmm nanometers to just the value in
                # angstroms
                d.append(
                    (i + 1, j + 1, from_openmm(eq).to("angstrom").magnitude)
                )

    return mapped_smiles, d


def to_besmarts(
    molecules: list[Molecule],
    procs: int = 8,
    chunksize: int = 8,
) -> dict[str, dict[tuple[int, int], list[float]]]:
    results = defaultdict(dict)
    with Pool(processes=procs) as pool:
        for mapped_smiles, d in tqdm(
            pool.imap(
                inner,
                molecules,
                chunksize=chunksize,
            ),
            desc="Converting to besmarts",
            total=len(molecules),
        ):
            results[mapped_smiles] = d

        # TODO angles and torsions too

    return results


def main():
    ds = load_dataset("filtered-opt.json", typ="optimization")
    data = [v for value in ds.entries.values() for v in value]
    # a little dumb to `from_mapped_smiles` here and then `to_mapped_smiles`
    # above, but I guess I do want the Molecule eventually
    molecules = [
        Molecule.from_mapped_smiles(r.cmiles, allow_undefined_stereo=True)
        for r in data
    ]
    # molecules = [Molecule.from_smiles("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")]
    besmarts = to_besmarts(molecules)

    with open("out.json", "w") as out:
        json.dump(besmarts, out)


if __name__ == "__main__":
    main()
