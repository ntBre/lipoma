import math
from dataclasses import dataclass

import numpy as np
import torch
from openff.toolkit import ForceField
from openmm import unit
from openmm.unit import Quantity

import espaloma as esp


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


def handle_bonds(force, g, bond_lookup, d):
    assert force.getNumBonds() * 2 == g.heterograph.number_of_nodes("n2")

    for idx in range(force.getNumBonds()):
        idx0, idx1, eq, k = force.getBondParameters(idx)
        position = bond_lookup[(idx0, idx1)]
        _eq = g.nodes["n2"].data["eq"][position].detach().numpy().item()
        _k = g.nodes["n2"].data["k"][position].detach().numpy().item()

        _eq = Quantity(
            _eq,
            esp.units.DISTANCE_UNIT,
        ).value_in_unit(unit.angstroms)

        _k = Quantity(
            _k,
            esp.units.FORCE_CONSTANT_UNIT,
        ).value_in_unit(unit.kilocalories_per_mole / unit.angstroms**2)

        d["bonds"].append(Bond(idx0 + 1, idx1 + 1, _eq, _k))


def handle_angles(force, g, angle_lookup, d):
    assert force.getNumAngles() * 2 == g.heterograph.number_of_nodes("n3")

    for idx in range(force.getNumAngles()):
        idx0, idx1, idx2, eq, k = force.getAngleParameters(idx)
        position = angle_lookup[(idx0, idx1, idx2)]
        _eq = g.nodes["n3"].data["eq"][position].detach().numpy().item()
        _k = g.nodes["n3"].data["k"][position].detach().numpy().item()

        _eq = Quantity(
            _eq,
            esp.units.ANGLE_UNIT,
        ).value_in_unit(unit.radians)

        _k = Quantity(
            _k,
            esp.units.ANGLE_FORCE_CONSTANT_UNIT,
        ).value_in_unit(unit.kilocalories_per_mole / unit.radian**2)

        d["angles"].append(Angle(idx0 + 1, idx1 + 1, idx2 + 1, _eq, _k))


def handle_torsions(force, g, d):
    if (
        "periodicity" not in g.nodes["n4"].data
        or "phase" not in g.nodes["n4"].data
    ):
        g.nodes["n4"].data["periodicity"] = torch.arange(1, 7)[None, :].repeat(
            g.heterograph.number_of_nodes("n4"), 1
        )

        g.nodes["n4"].data["phases"] = torch.zeros(
            g.heterograph.number_of_nodes("n4"), 6
        )

        g.nodes["n4_improper"].data["periodicity"] = torch.arange(1, 7)[
            None, :
        ].repeat(g.heterograph.number_of_nodes("n4_improper"), 1)

        g.nodes["n4_improper"].data["phases"] = torch.zeros(
            g.heterograph.number_of_nodes("n4_improper"), 6
        )

    for idx in range(g.heterograph.number_of_nodes("n4")):
        idx0 = g.nodes["n4"].data["idxs"][idx, 0].item()
        idx1 = g.nodes["n4"].data["idxs"][idx, 1].item()
        idx2 = g.nodes["n4"].data["idxs"][idx, 2].item()
        idx3 = g.nodes["n4"].data["idxs"][idx, 3].item()

        # assuming both (a,b,c,d) and (d,c,b,a) are listed for every
        # torsion, only pick one of the orderings
        if idx0 < idx3:
            periodicities = g.nodes["n4"].data["periodicity"][idx]
            phases = g.nodes["n4"].data["phases"][idx]
            ks = g.nodes["n4"].data["k"][idx]
            for sub_idx in range(ks.flatten().shape[0]):
                k = ks[sub_idx].item()
                if k != 0.0:
                    _periodicity = periodicities[sub_idx].item()
                    _phase = phases[sub_idx].item()

                    if k < 0:
                        k = -k
                        _phase = math.pi - _phase

                    k = Quantity(
                        k,
                        esp.units.ENERGY_UNIT,
                    ).value_in_unit(unit.kilocalories_per_mole)

                    d["torsions"].append(
                        Torsion(
                            idx0 + 1,
                            idx1 + 1,
                            idx2 + 1,
                            idx3 + 1,
                            _periodicity,
                            _phase,
                            k,
                        )
                    )

    if "k" in g.nodes["n4_improper"].data:
        for idx in range(g.heterograph.number_of_nodes("n4_improper")):
            idx0 = g.nodes["n4_improper"].data["idxs"][idx, 0].item()
            idx1 = g.nodes["n4_improper"].data["idxs"][idx, 1].item()
            idx2 = g.nodes["n4_improper"].data["idxs"][idx, 2].item()
            idx3 = g.nodes["n4_improper"].data["idxs"][idx, 3].item()

            periodicities = g.nodes["n4_improper"].data["periodicity"][idx]
            phases = g.nodes["n4_improper"].data["phases"][idx]
            ks = g.nodes["n4_improper"].data["k"][idx]
            for sub_idx in range(ks.flatten().shape[0]):
                k = ks[sub_idx].item()
                if k != 0.0:
                    _periodicity = periodicities[sub_idx].item()
                    _phase = phases[sub_idx].item()

                    if k < 0:
                        k = -k
                        _phase = math.pi - _phase

                    k = Quantity(
                        k,
                        esp.units.ENERGY_UNIT,
                    ).value_in_unit(unit.kilocalories_per_mole)

                    d["torsions"].append(
                        Torsion(
                            idx0 + 1,
                            idx1 + 1,
                            idx2 + 1,
                            idx3 + 1,
                            _periodicity,
                            _phase,
                            0.5 * k,
                        )
                    )


def openmm_system_from_graph(g, forcefield: ForceField):
    # get the mapping between position and indices
    bond_lookup = {
        tuple(idxs.detach().numpy()): position
        for position, idxs in enumerate(g.nodes["n2"].data["idxs"])
    }

    angle_lookup = {
        tuple(idxs.detach().numpy()): position
        for position, idxs in enumerate(g.nodes["n3"].data["idxs"])
    }

    # charge method always == "nn"
    g.mol.partial_charges = unit.elementary_charge * g.nodes["n1"].data[
        "q"
    ].flatten().detach().cpu().numpy().astype(
        np.float64,
    )
    sys = forcefield.create_openmm_system(
        g.mol.to_topology(),
        charge_from_molecules=[g.mol],
        allow_nonintegral_charges=True,
    )

    d = dict(bonds=[], angles=[], torsions=[])
    for force in sys.getForces():
        match force.__class__.__name__:
            case "HarmonicBondForce":
                handle_bonds(force, g, bond_lookup, d)
            case "HarmonicAngleForce":
                handle_angles(force, g, angle_lookup, d)
            case "PeriodicTorsionForce":
                handle_torsions(force, g, d)

    return d
