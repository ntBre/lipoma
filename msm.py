# comparing msm parameters to the Sage values. mostly copied from
# vflib/bin/create_msm.py

import faulthandler
import logging
import os
from collections import defaultdict

import click
import numpy as np
from openff.qcsubmit.results import OptimizationResultCollection
from openff.qcsubmit.results.filters import LowestEnergyFilter
from openff.toolkit import ForceField, Molecule
from openff.units import unit
from qcportal.models import ResultRecord
from qubekit.bonded.mod_seminario import ModSeminario, ModSemMaths
from qubekit.molecules import Ligand
from qubekit.utils.exceptions import StereoChemistryError
from tqdm import tqdm
from vflib.utils import Timer

from query import Records


def force_constant_bond(bond, eigenvals, eigenvecs, coords):
    atom_a, atom_b = bond
    eigenvals_ab = eigenvals[atom_a, atom_b, :]
    eigenvecs_ab = eigenvecs[:, :, atom_a, atom_b]

    unit_vectors_ab = ModSemMaths.unit_vector_along_bond(coords, bond)

    lst = [
        eigenvals_ab[i] * abs(np.dot(unit_vectors_ab, eigenvecs_ab[:, i]))
        for i in range(3)
    ]
    return -0.5 * sum(lst)


ModSemMaths.force_constant_bond = force_constant_bond


class Params:
    def __init__(self):
        self.bond_eq = defaultdict(list)
        self.bond_k = defaultdict(list)
        self.angle_eq = defaultdict(list)
        self.angle_k = defaultdict(list)

    def update(self, other):
        self.bond_eq.update(other.bond_eq)
        self.bond_k.update(other.bond_k)
        self.angle_eq.update(other.angle_eq)
        self.angle_k.update(other.angle_k)


def calculate_parameters(
    records,
    qc_record: ResultRecord,
    molecule: Molecule,
    forcefield: ForceField,
):
    """Calculate the modified seminario parameters for the given input
    molecule and store them by OFF SMIRKS. `records` is an out param for
    storing the results

    """
    mod_sem = ModSeminario()

    qube_mol = Ligand.from_rdkit(molecule.to_rdkit(), name="offmol")
    qube_mol.hessian = qc_record.return_result
    qube_mol = mod_sem.run(qube_mol)
    labels = forcefield.label_molecules(molecule.to_topology())[0]

    # bond units
    kj_per_mol_per_nm2 = unit.kilojoule_per_mole / unit.nanometer**2
    kcal_per_mol_per_ang = unit.kilocalorie_per_mole / (unit.angstrom**2)

    # angle units
    kj_per_mol_per_rad2 = unit.kilojoule_per_mole / (unit.radian**2)
    kcal_per_mol_per_rad = unit.kilocalorie_per_mole / unit.radian**2

    smiles = molecule.to_smiles(mapped=True)

    for bond, p in labels["Bonds"].items():
        qube_param = qube_mol.BondForce[bond]
        smirks = p.smirks

        length = qube_param.length * unit.nanometer
        length = length.to(unit.angstrom).magnitude
        records["bonds_eq"][smirks].molecules.append(smiles)
        records["bonds_eq"][smirks].espaloma_values.append(length)
        records["bonds_eq"][smirks].envs.append(bond)
        records["bonds_eq"][smirks].sage_value = p.length.magnitude
        records["bonds_eq"][smirks].ident = p.id

        k = qube_param.k * kj_per_mol_per_nm2
        k = k.to(kcal_per_mol_per_ang).magnitude
        records["bonds_dedup"][smirks].molecules.append(smiles)
        records["bonds_dedup"][smirks].espaloma_values.append(k)
        records["bonds_dedup"][smirks].envs.append(bond)
        records["bonds_dedup"][smirks].sage_value = p.k.magnitude
        records["bonds_dedup"][smirks].ident = p.id

    for angle, p in labels["Angles"].items():
        qube_param = qube_mol.AngleForce[angle]

        angle = qube_param.angle * unit.radian
        angle = angle.to(unit.degree).magnitude
        records["angles_eq"][smirks].molecules.append(smiles)
        records["angles_eq"][smirks].espaloma_values.append(angle)
        records["angles_eq"][smirks].envs.append(angle)
        records["angles_eq"][smirks].sage_value = p.angle.magnitude
        records["angles_eq"][smirks].ident = p.id

        k = qube_param.k * kj_per_mol_per_rad2
        k = k.to(kcal_per_mol_per_rad).magnitude
        records["angles_dedup"][smirks].molecules.append(smiles)
        records["angles_dedup"][smirks].espaloma_values.append(k)
        records["angles_dedup"][smirks].envs.append(angle)
        records["angles_dedup"][smirks].sage_value = p.k.magnitude
        records["angles_dedup"][smirks].ident = p.id


@click.command()
@click.option("--forcefield", "-f", default="openff-2.1.0.offxml")
@click.option("--dataset", "-d", default="datasets/filtered-opt.json")
@click.option("--out-dir", "-o", default="data")
def main(forcefield, dataset, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    timer = Timer()
    dataset = OptimizationResultCollection.parse_file(dataset)

    timer.say("finished loading dataset")

    filtered = dataset.filter(LowestEnergyFilter())

    timer.say("finished filtering dataset")

    hessian_set = filtered.to_basic_result_collection(driver="hessian")

    timer.say("finished converting dataset")

    print(f"Found {hessian_set.n_results} hessian calculations")
    print(f"Found {hessian_set.n_molecules} hessian molecules")

    ff = ForceField(forcefield, allow_cosmetic_attributes=True)

    records_and_molecules = hessian_set.to_records()

    records = defaultdict(Records)
    errors = 0
    for record, molecule in tqdm(records_and_molecules, desc="Computing msm"):
        try:
            calculate_parameters(records, record, molecule, ff)
        except (KeyError, StereoChemistryError):
            errors += 1

    timer.say(f"finished labeling with {errors} errors")

    for param, record in records.items():
        record.to_json(f"{out_dir}/{param}.json")


if __name__ == "__main__":
    logging.getLogger("openff").setLevel(logging.ERROR)
    with open("fault_handler.log", "w") as fobj:
        faulthandler.enable(fobj)
        main()
