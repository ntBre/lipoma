import warnings

from chemper.graphs.single_graph import SingleGraph
from chemper.mol_toolkits import mol_toolkit
from chemper.smirksify import SMIRKSifier, print_smirks
from openff.toolkit.topology import Molecule
from openmm.openmm import HarmonicBondForce
from rdkit.Chem import rdmolfiles

import espaloma as esp

warnings.filterwarnings("ignore", category=UserWarning)

molecule = Molecule.from_smiles("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
# print(molecule.bonds)
rdmol = molecule.to_rdkit()

chemper_mol = mol_toolkit.Mol.from_smiles("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")

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
for force in openmm_system.getForces():
    if isinstance(force, HarmonicBondForce):
        for b in range(force.getNumBonds()):
            i, j, eq, k = force.getBondParameters(b)

            atoms = set()
            for bond in molecule.bonds:
                idxs = [bond.atom1_index, bond.atom2_index]
                if i in idxs or j in idxs:
                    atoms.add(idxs[0])
                    atoms.add(idxs[1])
            print(atoms)
            graph = SingleGraph(chemper_mol, (i, j), layers=1)

            cluster_list = [(f"label {b}", [[(i, j)]])]
            fier = SMIRKSifier([chemper_mol], cluster_list, verbose=False)
            res = fier.reduce(max_its=100, verbose=False)
            print(res)

            print(
                i,
                j,
                rdmolfiles.MolFragmentToSmarts(rdmol, atomsToUse=atoms),
            )
        print()

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
