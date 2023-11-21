# plotting k and eq at the same time

import base64
import warnings
from functools import cache

import click
import numpy as np
from sklearn.mixture import GaussianMixture as model

from query import Records
from utils import Record, close, draw_rdkit, make_smirks, position_if, unit

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")

    import plotly.express as px
    from dash import Dash, Input, Output, callback, ctx, dcc, exceptions, html
    from openff.toolkit import Molecule


def cluster(record, nclusters):
    mat = np.column_stack((unit(record.eqs), unit(record.fcs)))
    if nclusters > 1 and len(mat) > nclusters:
        m = model(n_components=nclusters).fit(mat)
        kmeans = m.predict(mat)
        colors = kmeans.astype(str)
    else:
        colors = ["black"] * len(mat)
    return colors


def make_fig(record, nclusters=None, colors=None):
    if nclusters is not None:
        colors = cluster(record, nclusters)
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


def make_pics(pairs):
    record = cur_record()
    # have to search directly because the multiple curves when clustering
    # ruins using the data index directly
    pics = []
    for dx, dy in pairs:
        find = zip(record.eqs, record.fcs)
        p = position_if(find, lambda px: close(px[0], dx) and close(px[1], dy))
        mol, env = record.mols[p], record.envs[p]
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


@callback(
    Output("click-output", "children", allow_duplicate=True),
    Input("graph", "clickData"),
    prevent_initial_call=True,
)
def display_click_data(clickData):
    if clickData:
        data = clickData["points"][0]
        dx, dy = data["x"], data["y"]
        return make_pics([(dx, dy)])


@callback(
    Output("click-output", "children", allow_duplicate=True),
    Input("graph", "selectedData"),
    prevent_initial_call=True,
)
def display_select_data(selectData):
    if selectData:
        vals = [(p["x"], p["y"]) for p in selectData["points"]]
        return make_pics(vals)


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
    return make_fig(cur_record(), NCLUSTERS), SMIRKS[CUR_SMIRK]


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
    return make_fig(cur_record(), NCLUSTERS), SMIRKS[CUR_SMIRK]


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


@callback(
    Output("graph-container", "children", allow_duplicate=True),
    [Input("submit", "n_clicks"), Input("smirks_input", "value")],
    prevent_initial_call=True,
)
def submit_smirks(_, smirks):
    if ctx.triggered_id == "submit":
        rec = cur_record()
        colors = []
        for m, e in zip(rec.mols, rec.envs):
            mol = Molecule.from_mapped_smiles(m, allow_undefined_stereo=True)
            if (env := mol.chemical_environment_matches(smirks)) and (
                e in env or e[::-1] in env
            ):
                colors.append("matched")
            else:
                colors.append("unmatched")
        return make_fig(RECORDS[SMIRKS[CUR_SMIRK]], colors=colors)
    raise exceptions.PreventUpdate()


@cache
def make_records(method, param="bonds"):
    match method:
        case "esp":
            dir_ = "data/esp"
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
        dk = record.to_dict()
        de = be[smirks].to_dict()
        ret = Record.default(smirks, record.ident)
        for k, v in dk.items():
            mol, env = k
            ret.eqs.append(de[k])
            ret.fcs.append(v)
            ret.mols.append(mol)
            ret.envs.append(env)
        rets[smirks] = ret
    return rets


def make_input(smirks):
    return dcc.Input(id="smirks_input", value=smirks, style=dict(width="30vw"))


def cur_record():
    return RECORDS[SMIRKS[CUR_SMIRK]]


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
        html.H1("k vs eq"),
        dcc.RadioItems(["msm", "esp"], TYPE, inline=True, id="radio3"),
        dcc.RadioItems(["Bonds", "Angles"], "Bonds", inline=True, id="radio"),
        html.Button("Previous", id="previous", n_clicks=0),
        html.Button("Next", id="next", n_clicks=0),
        dcc.Slider(1, 10, 1, value=NCLUSTERS, id="clusters"),
        make_input(SMIRKS[CUR_SMIRK]),
        html.Button("Submit", id="submit", n_clicks=0),
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


@click.command()
@click.option("--port", "-p", default=8060)
def main(port):
    app.run(debug=True, port=port)


if __name__ == "__main__":
    main()
