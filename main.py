import json
import warnings
from collections import defaultdict

from openff.toolkit.topology import Molecule
from openff.units.openmm import from_openmm
from openmm.openmm import HarmonicBondForce

import espaloma as esp

warnings.filterwarnings("ignore", category=UserWarning)

molecule = Molecule.from_smiles("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
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


# get the mapping between position and indices, from espaloma/graphs/deploy.py
bond_lookup = {
    tuple(idxs.detach().numpy()): position
    for position, idxs in enumerate(molecule_graph.nodes["n2"].data["idxs"])
}

# print(bond_lookup)

# I guess what I would like to do here is map these forces back to actual atom
# labels. I'm not going to be able to map these to smirks patterns in any
# direct sense, but I'm curious to see the elements involved. are these indices
# into the original Molecule?
result = defaultdict(dict)
for force in openmm_system.getForces():
    if isinstance(force, HarmonicBondForce):
        for b in range(force.getNumBonds()):
            # ignore the force constant for now
            i, j, eq, _k = force.getBondParameters(b)
            # convert from openmm nanometers to just the value in angstroms
            result[mapped_smiles][(i, j)] = [
                from_openmm(eq).to("angstrom").magnitude
            ]


print(result)
# json.dumps(result)
# for k, v in result.items():
#     print(k)
#     for idxs, eqs in v.items():
#         print(f"\t{idxs} => {eqs}")

# my idea here is to take a molecule, use espaloma to generate the parameters
# for it, and just print them out. eventually, hopefully I can aggregate and
# deduplicate these parameters across a whole training set of molecules to get
# a new SMIRNOFF style force field that I can work with in the normal way:
# optimize with forcebalance and then benchmark with ibstore
#
# First, I need to figure out how to get a SMIRNOFF force field back from an
# openmm system. This `openmm_system_from_graph` call uses
# ForceField.create_openmm_system internally, which is good, but it overwrites
# the parameters on the openmm system, not directly in the force field
