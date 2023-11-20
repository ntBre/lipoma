import base64
import warnings

from utils import draw_rdkit, make_smirks

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")

    import numpy as np
    import pandas as pd
    import plotly.express as px
    from dash import Dash, Input, Output, callback, ctx, dcc, exceptions, html
    from openff.toolkit import Molecule

    from query import Records


def make_fig(smirk, record, title, colors=None):
    if colors is None:
        df = pd.DataFrame(
            dict(
                values=record.espaloma_values,
                labels=[title] * len(record.espaloma_values),
            )
        )
    else:
        df = pd.DataFrame(colors)

    fig = px.histogram(
        df,
        title=f"{record.ident} {smirk}",
        x="values",
        color="labels",
        labels="labels",
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
    fig.update_traces(marker_line_width=1)
    return dcc.Graph(figure=fig, id="graph")


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
    [
        Output("graph-container", "children", allow_duplicate=True),
        Output("smirks_input", "value", allow_duplicate=True),
    ],
    Input("previous", "n_clicks"),
    prevent_initial_call=True,
)
def previous_button(_):
    global CUR_SMIRK
    if CUR_SMIRK >= 1:
        CUR_SMIRK -= 1
    return (
        make_fig(SMIRKS[CUR_SMIRK], RECORDS[SMIRKS[CUR_SMIRK]], TITLE),
        SMIRKS[CUR_SMIRK],
    )


@callback(
    [
        Output("graph-container", "children", allow_duplicate=True),
        Output("smirks_input", "value", allow_duplicate=True),
    ],
    Input("next", "n_clicks"),
    prevent_initial_call=True,
)
def next_button(_):
    global CUR_SMIRK
    if CUR_SMIRK < len(SMIRKS) - 1:
        CUR_SMIRK += 1
    return (
        make_fig(SMIRKS[CUR_SMIRK], RECORDS[SMIRKS[CUR_SMIRK]], TITLE),
        SMIRKS[CUR_SMIRK],
    )


@callback(
    Output("graph-container", "children", allow_duplicate=True),
    [Input("submit", "n_clicks"), Input("smirks_input", "value")],
    prevent_initial_call=True,
)
def submit_smirks(_, smirks):
    if ctx.triggered_id == "submit":
        rec = cur_record()
        colors = dict(labels=[], values=[])
        for m, e, v in zip(rec.molecules, rec.envs, rec.espaloma_values):
            mol = Molecule.from_mapped_smiles(m, allow_undefined_stereo=True)
            # in this class the envs are lists!!! I hate python so much, why
            # didn't this give a type error
            if (env := mol.chemical_environment_matches(smirks)) and (
                tuple(e) in env or tuple(e[::-1]) in env
            ):
                colors["labels"].append("matched")
            else:
                colors["labels"].append("unmatched")
            colors["values"].append(v)

        return make_fig(
            SMIRKS[CUR_SMIRK], RECORDS[SMIRKS[CUR_SMIRK]], TITLE, colors=colors
        )
    raise exceptions.PreventUpdate()


def cur_record():
    return RECORDS[SMIRKS[CUR_SMIRK]]


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
        case e:
            raise ValueError(e)

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


def make_records(typ, param="bonds"):
    match typ:
        case "k":
            suff = "_dedup"
        case "eq":
            suff = "_eq"

    return Records.from_file(f"{DIR}/{param}{suff}.json")


def make_input(smirks):
    return dcc.Input(id="smirks_input", value=smirks, style=dict(width="30vw"))


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
                make_input(SMIRKS[CUR_SMIRK]),
                html.Button("Submit", id="submit", n_clicks=0),
            ]
        ),
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
