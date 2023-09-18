# filter the esp-tors-10 benchmark results to consider only the molecules
# associated with the two torsions

import logging

import numpy as np
from vflib.coverage import ParameterType, check_record_coverage

logging.getLogger("openff").setLevel(logging.ERROR)


# having trouble loading esp-tors-10 force field again (version mismatch in
# ib-dev env vs espaloma env), so just get the coverage from the Sage force
# field it's derived from
coverage = check_record_coverage(
    "openff-2.1.0.offxml", "datasets/industry.json", ParameterType.Torsions
)

t129_record_ids = coverage["t129"]
t140_record_ids = coverage["t140"]

np.savetxt("output/esp-tors-10/t129.records", t129_record_ids, fmt="%s")
np.savetxt("output/esp-tors-10/t140.records", t140_record_ids, fmt="%s")
