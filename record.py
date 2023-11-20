from dataclasses import asdict, dataclass
from typing import List


@dataclass
class Record:
    # parallel to espaloma_values, matching molecules to espaloma values
    molecules: List[str]
    espaloma_values: List[float]
    sage_value: float
    ident: str
    envs: List[List[int]]

    def __init__(
        self,
        molecules=None,
        espaloma_values=None,
        sage_value=None,
        ident=None,
        envs=None,
    ):
        # these three lists are parallel to each other
        if molecules is None:
            molecules = []
        if espaloma_values is None:
            espaloma_values = []
        if envs is None:
            envs = []

        self.molecules = molecules
        self.espaloma_values = espaloma_values
        self.sage_value = sage_value
        self.ident = ident
        self.envs = envs  # chemical environments from espaloma

    def asdict(self):
        return asdict(self)

    def to_dict(self):
        return {
            (m, tuple(e)): v
            for m, e, v in zip(self.molecules, self.envs, self.espaloma_values)
        }
