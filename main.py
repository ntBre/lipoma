import json
import logging
import warnings
from collections import defaultdict
from dataclasses import dataclass
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
from patches import openmm_system_from_graph

warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("openff").setLevel(logging.ERROR)


@dataclass
class Bond:
    """
    A bond between two atoms, with equilibrium distance `eq` in Å and a
    force constant `k` in kcal/mol/Å²"""

    atom1: int
    atom2: int
    eq: float
    k: float

    def to_json(self):
        return [self.atom1, self.atom2, self.eq, self.k]

    def from_zero(self):
        "Returns `self` with atoms indexed from 0 instead of 1"
        return Bond(self.atom1 - 1, self.atom2 - 1, self.eq, self.k)

    def as_tuple(self):
        return self.atom1, self.atom2, self.eq, self.k


@dataclass
class Angle:
    """
    An angle between three atoms, with equilibrium value `eq` in radians
    and a force constant `k` in kcal/mol/rad²"""

    atom1: int
    atom2: int
    atom3: int
    eq: float
    k: float

    def to_json(self):
        return [self.atom1, self.atom2, self.atom3, self.eq, self.k]

    def from_zero(self):
        "Returns `self` with atoms indexed from 0 instead of 1"
        return Angle(
            self.atom1 - 1, self.atom2 - 1, self.atom3 - 1, self.eq, self.k
        )

    def as_tuple(self):
        return self.atom1, self.atom2, self.atom3, self.eq, self.k


@dataclass
class Torsion:
    """
    A torsion between four atoms, with periodicity `per`, phase offset
    `phase` in radians and force constant `k` in kcal/mol"""

    atom1: int
    atom2: int
    atom3: int
    atom4: int
    per: int
    phase: float
    k: float

    def to_json(self):
        return [
            self.atom1,
            self.atom2,
            self.atom3,
            self.atom4,
            self.per,
            self.phase,
            self.k,
        ]

    def from_zero(self):
        "Returns `self` with atoms indexed from 0 instead of 1"
        return Torsion(
            self.atom1 - 1,
            self.atom2 - 1,
            self.atom3 - 1,
            self.atom4 - 1,
            self.per,
            self.phase,
            self.k,
        )

    def as_tuple(self):
        return (
            self.atom1,
            self.atom2,
            self.atom3,
            self.atom4,
            self.per,
            self.phase,
            self.k,
        )


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
    openmm_system = openmm_system_from_graph(
        molecule_graph, forcefield=FORCEFIELD
    )

    d = dict(bonds=[], angles=[], torsions=[])
    # hopefully these indices match the mapped_smiles...
    for force in openmm_system.getForces():
        if isinstance(force, HarmonicBondForce) and "bonds" in types:
            for b in range(force.getNumBonds()):
                # ignore the force constant for now
                i, j, eq, f = force.getBondParameters(b)
                # convert from openmm nanometers to just the value in
                # angstroms
                d["bonds"].append(
                    Bond(
                        i + 1,
                        j + 1,
                        from_openmm(eq).to("angstrom").magnitude,
                        from_openmm(f)
                        .to("kcal / (mol angstrom**2)")
                        .magnitude,
                    )
                )
        elif isinstance(force, HarmonicAngleForce) and "angles" in types:
            for a in range(force.getNumAngles()):
                i, j, k, eq, f = force.getAngleParameters(a)
                d["angles"].append(
                    Angle(
                        i + 1,
                        j + 1,
                        k + 1,
                        from_openmm(eq).magnitude,
                        from_openmm(f).to("kcal / (mol rad**2)").magnitude,
                    )
                )
        elif isinstance(force, PeriodicTorsionForce) and "torsions" in types:
            for t in range(force.getNumTorsions()):
                i, j, k, l, per, phase, f = force.getTorsionParameters(t)
                d["torsions"].append(
                    Torsion(
                        i + 1,
                        j + 1,
                        k + 1,
                        l + 1,
                        per,
                        from_openmm(phase).magnitude,
                        from_openmm(f).to("kcal/mol").magnitude,
                    )
                )
        elif isinstance(force, NonbondedForce):
            pass
        else:
            pass

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
