import base64
import re
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")

    import numpy as np
    import plotly.express as px
    from dash import Dash, Input, Output, callback, dcc, html
    from openff.toolkit import Molecule
    from rdkit.Chem.Draw import MolsToGridImage, rdDepictor, rdMolDraw2D

    from query import Records


def make_fig(smirk, record, title):
    fig = px.histogram(
        record.espaloma_values,
        title=f"{record.ident} {smirk}",
        labels=title,
        width=800,
        height=600,
    )
    fig.add_vline(
        x=record.sage_value,
        annotation_text="Sage",
        line_dash="dash",
        line_color="green",
    )
    fig.add_vline(
        x=np.average(record.espaloma_values),
        annotation_text=f"{title} Avg.",
        line_dash="dash",
    )
    fig.update_traces(marker_line_width=1, name=title)
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


MAX = 100


@callback(Output("click-output", "children"), Input("graph", "clickData"))
def display_click_data(clickData):
    if clickData:
        points = clickData["points"][0]["pointNumbers"]
        record = RECORDS[SMIRKS[CUR_SMIRK]]
        mols = {record.molecules[p]: record.envs[p] for p in points}
        pics = []
        count = 0
        for mol, env in mols.items():
            if count > MAX:
                break
            mol = Molecule.from_mapped_smiles(mol, allow_undefined_stereo=True)
            svg = draw_rdkit(mol, SMIRKS[CUR_SMIRK], env)
            try:
                encoded = base64.b64encode(bytes(svg, "utf-8"))
            except Exception as e:
                print("error: ", e)
            pics.append(
                html.Img(src=f"data:image/svg+xml;base64,{encoded.decode()}")
            )
            count += 1
        return pics


@callback(
    Output("graph-container", "children", allow_duplicate=True),
    Input("previous", "n_clicks"),
    prevent_initial_call=True,
)
def previous_button(_):
    global CUR_SMIRK
    if CUR_SMIRK >= 1:
        CUR_SMIRK -= 1
    return make_fig(SMIRKS[CUR_SMIRK], RECORDS[SMIRKS[CUR_SMIRK]], TITLE)


@callback(
    Output("graph-container", "children", allow_duplicate=True),
    Input("next", "n_clicks"),
    prevent_initial_call=True,
)
def next_button(_):
    global CUR_SMIRK
    if CUR_SMIRK < len(SMIRKS) - 1:
        CUR_SMIRK += 1
    return make_fig(SMIRKS[CUR_SMIRK], RECORDS[SMIRKS[CUR_SMIRK]], TITLE)


def make_radio(k):
    match k:
        case "k" | "esp":
            radio = (
                dcc.RadioItems(
                    ["Bonds", "Angles", "Torsions", "Impropers"],
                    "Bonds",
                    inline=True,
                    id="radio",
                ),
            )
        case "eq" | "msm":
            radio = (
                dcc.RadioItems(
                    ["Bonds", "Angles"],
                    "Bonds",
                    inline=True,
                    id="radio",
                ),
            )
    return radio


@callback(
    Output("radio-parent", "children", allow_duplicate=True),
    Output("graph-container", "children", allow_duplicate=True),
    Input("radio2", "value"),
    prevent_initial_call=True,
)
def choose_constant(value):
    global RECORDS, SMIRKS, CUR_SMIRK, TYPE
    TYPE = value
    RECORDS = make_records(value)
    SMIRKS = make_smirks(RECORDS)
    CUR_SMIRK = 0
    fig = make_fig(SMIRKS[CUR_SMIRK], RECORDS[SMIRKS[CUR_SMIRK]], TITLE)
    radio = make_radio(value)
    return radio, fig


@callback(
    Output("radio-parent", "children", allow_duplicate=True),
    Output("graph-container", "children", allow_duplicate=True),
    Input("radio3", "value"),
    prevent_initial_call=True,
)
def choose_data(value):
    global RECORDS, SMIRKS, CUR_SMIRK, TYPE, DIR, TITLE
    match value:
        case "esp":
            TITLE = "Espaloma"
            DIR = "data/industry"
        case "msm":
            TITLE = "MSM"
            DIR = "data/msm"
    RECORDS = make_records(TYPE)
    SMIRKS = make_smirks(RECORDS)
    CUR_SMIRK = 0
    fig = make_fig(SMIRKS[CUR_SMIRK], RECORDS[SMIRKS[CUR_SMIRK]], TITLE)
    radio = make_radio(value)
    return radio, fig


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
        case "Torsions":
            param = "torsions"
        case "Impropers":
            param = "impropers"
    RECORDS = make_records(TYPE, param)
    SMIRKS = make_smirks(RECORDS)
    CUR_SMIRK = 0
    return make_fig(SMIRKS[CUR_SMIRK], RECORDS[SMIRKS[CUR_SMIRK]], TITLE)


# pasted from benchmarking/parse_hist
LABEL = re.compile(r"([bati])(\d+)([a-z]*)")


def sort_label(key):
    t, n, tail = LABEL.match(key).groups()
    return (t, int(n), tail)


def make_smirks(records):
    pairs = [(smirks, record) for smirks, record in records.items()]
    pairs = sorted(pairs, key=lambda pair: sort_label(pair[1].ident))
    return [smirks for smirks, record in pairs]


def make_records(typ, param="bonds"):
    match typ:
        case "k":
            suff = "_dedup"
        case "eq":
            suff = "_eq"

    return Records.from_file(f"{DIR}/{param}{suff}.json")


TITLE = "Espaloma"
DIR = "data/industry"
TYPE = "k"
RECORDS = make_records(TYPE)
SMIRKS = make_smirks(RECORDS)
CUR_SMIRK = 0

app = Dash(__name__)

colors = {"background": "white", "text": "black"}

app.layout = html.Div(
    style={"backgroundColor": colors["background"]},
    children=[
        dcc.RadioItems(["esp", "msm"], "esp", inline=True, id="radio3"),
        dcc.RadioItems(["k", "eq"], "k", inline=True, id="radio2"),
        html.Div(make_radio("k"), id="radio-parent"),
        html.Button("Previous", id="previous", n_clicks=0),
        html.Button("Next", id="next", n_clicks=0),
        html.Div(
            [
                html.Div(
                    [
                        make_fig(
                            SMIRKS[CUR_SMIRK],
                            RECORDS[SMIRKS[CUR_SMIRK]],
                            TITLE,
                        )
                    ],
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
        html.Div([], id="radio-output"),
    ],
)


if __name__ == "__main__":
    app.run(debug=True)
