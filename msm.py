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


def dot(v, w):
    ret = 0.0
    for i in range(3):
        ret += v[i] * w[i]
    return ret


def force_constant_bond(bond, eigenvals, eigenvecs, coords):
    atom_a, atom_b = bond
    eigenvals_ab = eigenvals[atom_a, atom_b, :]
    eigenvecs_ab = eigenvecs[:, :, atom_a, atom_b]

    unit_vectors_ab = ModSemMaths.unit_vector_along_bond(coords, bond)

    s = 0.0
    for i in range(3):
        v = unit_vectors_ab
        w = eigenvecs_ab[:, i]
        d = dot(v, w)
        s += eigenvals_ab[i] * abs(d)

    return -0.5 * s


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

        length = qube_param.length * unit.nanometer
        length = length.to(unit.angstrom).magnitude
        records["bonds_eq"][p.smirks].molecules.append(smiles)
        records["bonds_eq"][p.smirks].espaloma_values.append(length)
        records["bonds_eq"][p.smirks].envs.append(bond)
        records["bonds_eq"][p.smirks].sage_value = p.length.magnitude
        records["bonds_eq"][p.smirks].ident = p.id

        k = qube_param.k * kj_per_mol_per_nm2
        k = k.to(kcal_per_mol_per_ang).magnitude
        records["bonds_dedup"][p.smirks].molecules.append(smiles)
        records["bonds_dedup"][p.smirks].espaloma_values.append(k)
        records["bonds_dedup"][p.smirks].envs.append(bond)
        records["bonds_dedup"][p.smirks].sage_value = p.k.magnitude
        records["bonds_dedup"][p.smirks].ident = p.id

    for key, p in labels["Angles"].items():
        qube_param = qube_mol.AngleForce[key]

        angle = qube_param.angle * unit.radian
        angle = angle.to(unit.degree).magnitude
        records["angles_eq"][p.smirks].molecules.append(smiles)
        records["angles_eq"][p.smirks].espaloma_values.append(angle)
        records["angles_eq"][p.smirks].envs.append(key)
        records["angles_eq"][p.smirks].sage_value = p.angle.magnitude
        records["angles_eq"][p.smirks].ident = p.id

        k = qube_param.k * kj_per_mol_per_rad2
        k = k.to(kcal_per_mol_per_rad).magnitude
        records["angles_dedup"][p.smirks].molecules.append(smiles)
        records["angles_dedup"][p.smirks].espaloma_values.append(k)
        records["angles_dedup"][p.smirks].envs.append(key)
        records["angles_dedup"][p.smirks].sage_value = p.k.magnitude
        records["angles_dedup"][p.smirks].ident = p.id


def distance(record) -> float:
    """Compute the mean absolute difference between record.espaloma_values and
    record.sage_value"""
    lst = [abs(e - record.sage_value) for e in record.espaloma_values]
    return sum(lst) / len(lst)


def summary(records):
    """Print the distance summary for all of records"""
    ptypes = ["bonds_eq", "bonds_dedup", "angles_eq", "angles_dedup"]
    width = 14
    print("".join([f"{p:>{width}}" for p in ptypes]))
    for ptype in ptypes:
        ds = [distance(rec) for rec in records[ptype].values()]
        avg = sum(ds) / len(ds)
        print(f"{avg:{width}.6f}", end="")
    print()


class MSM:
    def __init__(self, dataset):
        self.timer = Timer()
        dataset = OptimizationResultCollection.parse_file(dataset)

        self.timer.say("finished loading dataset")

        filtered = dataset.filter(LowestEnergyFilter())

        self.timer.say("finished filtering dataset")

        hessian_set = filtered.to_basic_result_collection(driver="hessian")

        self.timer.say("finished converting dataset")

        print(f"Found {hessian_set.n_results} hessian calculations")
        print(f"Found {hessian_set.n_molecules} hessian molecules")

        self.records_and_molecules = hessian_set.to_records()

    def compute_msm(self, forcefield):
        ff = ForceField(forcefield, allow_cosmetic_attributes=True)

        records = defaultdict(Records)
        errors = 0
        for record, molecule in tqdm(
            self.records_and_molecules, desc="Computing msm"
        ):
            try:
                calculate_parameters(records, record, molecule, ff)
            except (KeyError, StereoChemistryError):
                errors += 1

        self.timer.say(f"finished labeling with {errors} errors")

        return records

    def score(self, forcefield):
        "Print a distance summary for `forcefield`"
        records = self.compute_msm(forcefield)
        summary(records)

    def update_forcefield(self, forcefield) -> ForceField:
        """Update the parameters in `forcefield` with the MSM guess and return
        the new ForceField"""
        ff = ForceField(forcefield, allow_cosmetic_attributes=True)
        records = self.compute_msm(forcefield)
        bh = ff.get_parameter_handler("Bonds")
        for smirks, record in records["bonds_eq"].items():
            bond = bh.parameters[smirks]
            bond.length = np.mean(record.espaloma_values) * unit.angstrom
            bond.k = (
                np.mean(records["bonds_dedup"][smirks].espaloma_values)
                * unit.kilocalorie_per_mole
                / (unit.angstrom**2)
            )

        ah = ff.get_parameter_handler("Angles")
        for smirks, record in records["angles_eq"].items():
            angle = ah.parameters[smirks]
            angle.angle = np.mean(record.espaloma_values) * unit.degrees
            angle.k = (
                np.mean(records["angles_dedup"][smirks].espaloma_values)
                * unit.kilocalorie_per_mole
                / unit.radian**2
            )
        return ff


@click.command()
@click.option("--forcefield", "-f", default="openff-2.1.0.offxml")
@click.option("--dataset", "-d", default="datasets/filtered-opt.json")
@click.option("--out-dir", "-o", default="data")
def main(forcefield, dataset, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    msm = MSM(dataset)

    records = msm.compute_msm(forcefield)

    for param, record in records.items():
        record.to_json(f"{out_dir}/{param}.json")


if __name__ == "__main__":
    logging.getLogger("openff").setLevel(logging.ERROR)
    with open("fault_handler.log", "w") as fobj:
        faulthandler.enable(fobj)
        main()
