import json
import warnings
from collections import defaultdict
from typing import Union

from openff.qcsubmit.results import (
    OptimizationResultCollection,
    TorsionDriveResultCollection,
)
from openff.toolkit.topology import Molecule
from openff.units.openmm import from_openmm
from openmm.openmm import HarmonicBondForce

import espaloma as esp

warnings.filterwarnings("ignore", category=UserWarning)


# copy pasta from known-issues main.py
def load_dataset(
    dataset: str,
) -> Union[OptimizationResultCollection, TorsionDriveResultCollection]:
    """Peeks at the first entry of `dataset` to determine its type and
    then loads it appropriately.

    Raises a `TypeError` if the first entry is neither a `torsion`
    record nor an `optimization` record.
    """
    with open(dataset, "r") as f:
        j = json.load(f)
    entries = j["entries"]
    keys = entries.keys()
    assert len(keys) == 1  # only handling this case for now
    key = list(keys)[0]
    match j["entries"][key][0]["type"]:
        case "torsion":
            return TorsionDriveResultCollection.parse_file(dataset)
        case "optimization":
            return OptimizationResultCollection.parse_file(dataset)
        case t:
            raise TypeError(f"Unknown result collection type: {t}")


def to_besmarts(
    molecules: list[Molecule],
) -> dict[str, dict[tuple[int, int], list[float]]]:
    results = defaultdict(dict)
    for molecule in molecules:
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

        # hopefully these indices match the mapped_smiles...
        for force in openmm_system.getForces():
            if isinstance(force, HarmonicBondForce):
                for b in range(force.getNumBonds()):
                    # ignore the force constant for now
                    i, j, eq, _k = force.getBondParameters(b)
                    # convert from openmm nanometers to just the value in
                    # angstroms
                    results[mapped_smiles][(i, j)] = [
                        from_openmm(eq).to("angstrom").magnitude
                    ]

        # TODO angles and torsions too

    return results


def main():
    molecule = Molecule.from_smiles("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
    print(to_besmarts([molecule]))


if __name__ == "__main__":
    main()
