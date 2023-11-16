# plotting k and eq at the same time

import base64
import warnings
from functools import cache

import numpy as np
from sklearn.mixture import GaussianMixture as model

from query import Records
from utils import Record, close, draw_rdkit, make_smirks, position_if, unit

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")

    import plotly.express as px
    from dash import Dash, Input, Output, callback, dcc, html
    from openff.toolkit import Molecule


def make_fig(record, nclusters):
    mat = np.column_stack((unit(record.eqs), unit(record.fcs)))
    if nclusters > 1 and len(mat) > nclusters:
        m = model(n_components=nclusters).fit(mat)
        kmeans = m.predict(mat)
        colors = kmeans.astype(str)
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
    # diagonal line if they agreed
    fig.add_shape(
        type="line",
        x0=min(record.eqs),
        x1=max(record.eqs),
        y0=min(record.fcs),
        y1=max(record.fcs),
    )
    fig.update_layout(xaxis_title="msm", yaxis_title="esp")
    return dcc.Graph(figure=fig, id="graph")


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
        return f"{data}"


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


@cache
def make_records(method, param="bonds"):
    match method:
        case "k":
            suff = "_dedup"
        case "eq":
            suff = "_eq"
    # k -> espaloma, eq -> msm to reuse structure from twod
    bk = Records.from_file(f"data/esp/{param}{suff}.json")
    be = Records.from_file(f"data/msm/{param}{suff}.json")

    # a Records is a dict of smirks -> Record and a Record contains three
    # parallel arrays I'm interested in: molecules, espaloma_values, and envs.

    keys = set(bk.keys()) & set(be.keys())

    rets = dict()
    for smirks in keys:
        record = bk[smirks]
        dk = record.to_dict()
        de = be[smirks].to_dict()
        ret = Record.default(smirks, record.ident)
        for k, v in dk.items():
            mol, env = k
            if k in de:
                ret.eqs.append(de[k])
                ret.fcs.append(v)
                ret.mols.append(mol)
                ret.envs.append(env)
        rets[smirks] = ret
    return rets


TYPE = "k"
RECORDS = make_records(TYPE)
SMIRKS = make_smirks(RECORDS)
CUR_SMIRK = 0
NCLUSTERS = 1

app = Dash(__name__)

colors = {"background": "white", "text": "black"}

app.layout = html.Div(
    style={"backgroundColor": colors["background"]},
    children=[
        html.H1("Espaloma vs MSM"),
        dcc.RadioItems(["k", "eq"], TYPE, inline=True, id="radio3"),
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
    app.run(debug=True, port=8070)
