# plotting k and eq at the same time

import base64
import re
import warnings
from dataclasses import dataclass
from functools import cache
from typing import List, Tuple

import numpy as np
from sklearn.cluster import KMeans as model

from query import Records

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")

    import plotly.express as px
    from chemper.mol_toolkits import mol_toolkit
    from chemper.smirksify import SMIRKSifier, print_smirks
    from dash import Dash, Input, Output, callback, dcc, html
    from openff.toolkit import Molecule
    from rdkit.Chem.Draw import MolsToGridImage, rdDepictor, rdMolDraw2D


def unit(vec):
    v = np.array(vec)
    return v / np.linalg.norm(v)


def make_fig(record, nclusters):
    mat = np.column_stack((unit(record.eqs), unit(record.fcs)))
    # kmeans seems to cluster too simply to be of much use.
    # AffinityPropagation, on the other hand, yields too many clusters to be of
    # much use. it's also quite slow to run. I skipped Mean-shift because it
    # has similar characteristics to AffinityPropagation. SpectralClustering is
    # nice because it finally gives non-vertical splits, but it becomes
    # intractable above ~3 clusters. AgglomerativeClustering with the Ward
    # linkage looks the same as kmeans but runs much slower. OPTICS produces
    # too many clusters again. KMeans it is

    if nclusters > 1 and len(mat) > nclusters:
        kmeans = model(n_clusters=nclusters, n_init="auto").fit(mat)
        colors = kmeans.labels_.astype(str)
    else:
        colors = ["black"] * len(mat)
    fig = px.scatter(
        x=record.eqs,
        y=record.fcs,
        title=f"{record.ident} {record.smirks}",
        width=800,
        height=600,
        color=colors,  # for discrete types
    )
    fig.update_layout(xaxis_title="eq", yaxis_title="k")
    return dcc.Graph(figure=fig, id="graph")


# adapted from ligand Molecule::to_svg
def draw_rdkit(mol, smirks, matches):
    rdmol = mol.to_rdkit()
    rdDepictor.SetPreferCoordGen(True)
    rdDepictor.Compute2DCoords(rdmol)
    rdmol = rdMolDraw2D.PrepareMolForDrawing(rdmol)
    return MolsToGridImage(
        [rdmol],
        useSVG=True,
        highlightAtomLists=[matches],
        subImgSize=(300, 300),
        molsPerRow=1,
    )


def position_if(lst, f):
    "Return the first index in `lst` satisfying `f`"
    return next(i for i, x in enumerate(lst) if f(x))


def close(x, y, eps=1e-16):
    return abs(x - y) < eps


@callback(Output("click-output", "children"), Input("graph", "clickData"))
def display_click_data(clickData):
    if clickData:
        data = clickData["points"][0]
        dx, dy = data["x"], data["y"]
        record = RECORDS[SMIRKS[CUR_SMIRK]]
        find = zip(record.eqs, record.fcs)
        # have to search directly because the multiple curves when clustering
        # ruins using the data index directly
        p = position_if(find, lambda px: close(px[0], dx) and close(px[1], dy))
        mol, env = record.mols[p], record.envs[p]
        pics = []
        mol = Molecule.from_mapped_smiles(mol, allow_undefined_stereo=True)
        svg = draw_rdkit(mol, SMIRKS[CUR_SMIRK], env)
        try:
            encoded = base64.b64encode(bytes(svg, "utf-8"))
        except Exception as e:
            print("error: ", e)
        pics.append(
            html.Img(src=f"data:image/svg+xml;base64,{encoded.decode()}")
        )
        return pics


@callback(Output("select-output", "children"), Input("graph", "selectedData"))
def display_select_data(selectData):
    if selectData:
        data = [x["pointNumber"] for x in selectData["points"]]
        record = RECORDS[SMIRKS[CUR_SMIRK]]
        mols = [s for i, s in enumerate(record.mols) if i in data]
        print(mols[:5])
        envs_in = [env for i, env in enumerate(record.envs) if i in data]
        print(envs_in[:5])
        x = []
        y = []
        for i, env in enumerate(record.envs):
            if i in data:
                x.append([env])
                y.append([])
            else:
                x.append([])
                y.append([env])
        atom_index_list = [("x", x), ("y", y)]
        print("calling smirksifier")
        graph = SMIRKSifier(mols, atom_index_list, verbose=True, max_layers=1)
        print("done with that")
        print("the graph", graph)
        red = graph.reduce(max_its=10)
        print("the red", red)
        print_smirks(red)
        return f"{None}"


@callback(
    Output("graph-container", "children", allow_duplicate=True),
    Input("previous", "n_clicks"),
    prevent_initial_call=True,
)
def previous_button(_):
    global CUR_SMIRK
    if CUR_SMIRK >= 1:
        CUR_SMIRK -= 1
    return make_fig(RECORDS[SMIRKS[CUR_SMIRK]], NCLUSTERS)


@callback(
    Output("graph-container", "children", allow_duplicate=True),
    Input("next", "n_clicks"),
    prevent_initial_call=True,
)
def next_button(_):
    global CUR_SMIRK
    if CUR_SMIRK < len(SMIRKS) - 1:
        CUR_SMIRK += 1
    return make_fig(RECORDS[SMIRKS[CUR_SMIRK]], NCLUSTERS)


@callback(
    Output("graph-container", "children", allow_duplicate=True),
    Input("clusters", "value"),
    prevent_initial_call=True,
)
def choose_clusters(value):
    global RECORDS, SMIRKS, CUR_SMIRK, TYPE, NCLUSTERS
    NCLUSTERS = value
    fig = make_fig(RECORDS[SMIRKS[CUR_SMIRK]], NCLUSTERS)
    return fig


@callback(
    Output("graph-container", "children", allow_duplicate=True),
    Input("radio3", "value"),
    prevent_initial_call=True,
)
def choose_data(value):
    global RECORDS, SMIRKS, CUR_SMIRK, TYPE
    TYPE = value
    RECORDS = make_records(TYPE)
    SMIRKS = make_smirks(RECORDS)
    fig = make_fig(RECORDS[SMIRKS[CUR_SMIRK]], NCLUSTERS)
    return fig


@callback(
    Output("graph-container", "children", allow_duplicate=True),
    Input("radio", "value"),
    prevent_initial_call=True,
)
def choose_parameter(value):
    global RECORDS, SMIRKS, CUR_SMIRK
    match value:
        case "Bonds":
            param = "bonds"
        case "Angles":
            param = "angles"
        case e:
            raise ValueError(f"choose_parameter got {e}")
    RECORDS = make_records(TYPE, param)
    SMIRKS = make_smirks(RECORDS)
    CUR_SMIRK = 0
    return make_fig(RECORDS[SMIRKS[CUR_SMIRK]], NCLUSTERS)


# pasted from benchmarking/parse_hist
LABEL = re.compile(r"([bati])(\d+)([a-z]*)")


def sort_label(key):
    t, n, tail = LABEL.match(key).groups()
    return (t, int(n), tail)


def make_smirks(records):
    pairs = [(smirks, r.ident) for smirks, r in records.items()]
    pairs = sorted(pairs, key=lambda pair: sort_label(pair[1]))
    return [smirks for smirks, _ in pairs]


def to_dict(record):
    return {
        (m, tuple(e)): v
        for m, e, v in zip(
            record.molecules, record.envs, record.espaloma_values
        )
    }


@dataclass
class Record:
    """Each record has a list of equilibrium values and force constants for
    plotting, a smirks pattern, a list of molecules, and a list of
    environments. All of the lists are parallel to each other"""

    eqs: List[float]
    fcs: List[float]
    mols: List[str]
    envs: List[Tuple[int, int]]
    smirks: str
    ident: str

    @staticmethod
    def default(smirks, ident):
        return Record([], [], [], [], smirks, ident)


@cache
def make_records(method, param="bonds"):
    match method:
        case "esp":
            dir_ = "data/industry"
        case "msm":
            dir_ = "data/msm"
    base = f"{dir_}/{param}"
    bk = Records.from_file(f"{base}_dedup.json")
    be = Records.from_file(f"{base}_eq.json")

    # a Records is a dict of smirks -> Record and a Record contains three
    # parallel arrays I'm interested in: molecules, espaloma_values, and envs.

    # first make sure they have the same smirks patterns. probably I should
    # convert to a set to ignore order, but this is working for now
    assert list(bk.keys()) == list(be.keys())

    rets = dict()
    for smirks, record in bk.items():
        dk = to_dict(record)
        de = to_dict(be[smirks])
        ret = Record.default(smirks, record.ident)
        for k, v in dk.items():
            mol, env = k
            ret.eqs.append(de[k])
            ret.fcs.append(v)
            ret.mols.append(mol)
            ret.envs.append(env)
        rets[smirks] = ret
    return rets


TYPE = "msm"
RECORDS = make_records(TYPE)
SMIRKS = make_smirks(RECORDS)
CUR_SMIRK = 0
NCLUSTERS = 1

app = Dash(__name__)

colors = {"background": "white", "text": "black"}

app.layout = html.Div(
    style={"backgroundColor": colors["background"]},
    children=[
        dcc.RadioItems(["msm", "esp"], TYPE, inline=True, id="radio3"),
        dcc.RadioItems(["Bonds", "Angles"], "Bonds", inline=True, id="radio"),
        html.Button("Previous", id="previous", n_clicks=0),
        html.Button("Next", id="next", n_clicks=0),
        dcc.Slider(1, 10, 1, value=NCLUSTERS, id="clusters"),
        html.Div(
            [
                html.Div(
                    [make_fig(RECORDS[SMIRKS[CUR_SMIRK]], NCLUSTERS)],
                    id="graph-container",
                    style=dict(display="inline-block"),
                ),
                html.Div(
                    [],
                    id="click-output",
                    style={
                        "display": "inline-block",
                        "max-height": "90vh",
                        "overflow": "hidden",
                        "overflow-y": "scroll",
                    },
                ),
            ],
            style=dict(display="flex"),
        ),
        html.Div(
            [],
            id="select-output",
        ),
    ],
)


if __name__ == "__main__":
    app.run(debug=True)
