# remove parameterize from some parameters that we shouldn't touch and move
# angles very close to 180.0 to 180.0 exactly

from openff.toolkit import ForceField
from openff.units import unit

ff = ForceField(
    "fb-fit/forcefield/force-field.offxml", allow_cosmetic_attributes=True
)

angles = ff.get_parameter_handler("Angles")

for angle in angles:
    if angle.angle.to(unit.degrees) > 170.0 * unit.degrees:
        if angle.angle.to(unit.degrees) > 178.0 * unit.degrees:
            angle.angle = 180.0 * unit.degrees
        if 'parameterize' in angle._cosmetic_attribs:
            assert "angle, k" == angle._parameterize
            angle._parameterize = "k"

ff.to_file("fb-fit/forcefield/force-field.offxml")
