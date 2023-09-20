import json
import logging
import warnings
from collections import defaultdict
from multiprocessing import Pool

from openff.toolkit import ForceField, Molecule
from openff.units.openmm import from_openmm
from openmm.openmm import (
    HarmonicAngleForce,
    HarmonicBondForce,
    NonbondedForce,
    PeriodicTorsionForce,
)
from tqdm import tqdm
from vflib import load_dataset

import espaloma as esp
from wrapper import openmm_system_from_graph

warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("openff").setLevel(logging.ERROR)


FORCEFIELD = ForceField("openff_unconstrained-2.1.0.offxml")


def espaloma_label(molecule, types=["bonds", "angles", "torsions"]):
    """Takes a `Molecule`, constructs an espaloma Graph object, assigns the
    molecule parameters based on that graph, constructs an OpenMM system from
    the graph, and extracts the force field parameters from the OpenMM system.

    Returns the molecule's mapped SMILES string and dict of bonds->[Bond],
    angles->[Angle], and torsions->[Torsion]

    """
    mapped_smiles = molecule.to_smiles(mapped=True)

    # create an Espaloma Graph object to represent the molecule of interest
    molecule_graph = esp.Graph(molecule)

    # load pretrained model
    espaloma_model = esp.get_model("latest")

    # apply a trained espaloma model to assign parameters
    espaloma_model(molecule_graph.heterograph)

    # create an OpenMM System for the specified molecule
    d = openmm_system_from_graph(molecule_graph, forcefield=FORCEFIELD)

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
                espaloma_label,
                molecules,
                chunksize=chunksize,
            ),
            desc="Converting to besmarts",
            total=len(molecules),
        ):
            results[mapped_smiles] = d

    return results


def main():
    ds = load_dataset("datasets/filtered-opt.json", type_="optimization")
    molecules = ds.to_molecules()

    besmarts = to_besmarts(molecules)

    with open("out.json", "w") as out:
        json.dump(besmarts, out, default=lambda x: x.to_json())


if __name__ == "__main__":
    main()
