import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")

    import base64

    import numpy as np
    import plotly.express as px
    from dash import Dash, Input, Output, callback, dcc, html
    from openff.toolkit import Molecule
    from rdkit.Chem.Draw import MolsToGridImage, rdDepictor, rdMolDraw2D

    from query import Records


def make_fig(smirk, record):
    fig = px.histogram(
        record.espaloma_values,
        title=f"{record.ident} {smirk}",
        labels="Espaloma",
    )
    fig.add_vline(
        x=record.sage_value, annotation_text="Sage Avg.", line_dash="dash"
    )
    fig.add_vline(
        x=np.average(record.espaloma_values),
        annotation_text="Espaloma Avg.",
        line_dash="dash",
    )
    fig.update_traces(marker_line_width=1, name="Espaloma")
    return dcc.Graph(figure=fig, id="graph")


# adapted from ligand Molecule::to_svg
def draw_rdkit(mol, smirks):
    matches = mol.chemical_environment_matches(smirks)
    rdmol = mol.to_rdkit()
    rdDepictor.SetPreferCoordGen(True)
    rdDepictor.Compute2DCoords(rdmol)
    rdmol = rdMolDraw2D.PrepareMolForDrawing(rdmol)
    return MolsToGridImage(
        [rdmol],
        useSVG=True,
        highlightAtomLists=matches,
        subImgSize=(300, 300),
        molsPerRow=1,
    )


@callback(Output("click-output", "children"), Input("graph", "clickData"))
def display_click_data(clickData):
    if clickData:
        points = clickData["points"][0]["pointNumbers"]
        mols = {RECORDS[SMIRKS[CUR_SMIRK]].molecules[p] for p in points}
        pics = []
        for mol in mols:
            svg = draw_rdkit(
                Molecule.from_smiles(mol, allow_undefined_stereo=True),
                SMIRKS[CUR_SMIRK],
            )
            try:
                encoded = base64.b64encode(bytes(svg, "utf-8"))
            except Exception as e:
                print("error: ", e)
            pics.append(
                html.Img(src=f"data:image/svg+xml;base64,{encoded.decode()}")
            )
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
    return make_fig(SMIRKS[CUR_SMIRK], RECORDS[SMIRKS[CUR_SMIRK]])


@callback(
    Output("graph-container", "children", allow_duplicate=True),
    Input("next", "n_clicks"),
    prevent_initial_call=True,
)
def next_button(_):
    global CUR_SMIRK
    if CUR_SMIRK < len(SMIRKS) - 1:
        CUR_SMIRK += 1
    return make_fig(SMIRKS[CUR_SMIRK], RECORDS[SMIRKS[CUR_SMIRK]])


RECORDS = Records.from_file("data/bonds_dedup.json")

SMIRKS = list(RECORDS.keys())

CUR_SMIRK = 0

app = Dash(__name__)

colors = {"background": "white", "text": "black"}

app.layout = html.Div(
    style={"backgroundColor": colors["background"]},
    children=[
        html.Button("Previous", id="previous", n_clicks=0),
        html.Button("Next", id="next", n_clicks=0),
        html.Div(
            [make_fig(SMIRKS[CUR_SMIRK], RECORDS[SMIRKS[CUR_SMIRK]])],
            id="graph-container",
        ),
        html.Div([], id="click-output"),
    ],
)


if __name__ == "__main__":
    app.run(debug=True)
