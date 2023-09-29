import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")

    import base64

    import plotly.express as px
    from dash import Dash, Input, Output, callback, dcc, html
    from openff.toolkit import Molecule
    from rdkit.Chem.Draw import rdDepictor, rdMolDraw2D

    from query import Records

records = Records.from_file("data/bonds_dedup.json")

smirks = next(iter(records.keys()))
record = records[smirks]

app = Dash(__name__)

colors = {"background": "white", "text": "black"}

fig = px.histogram(
    record.espaloma_values, title=f"{record.ident} {smirks}", labels="Espaloma"
)
fig.add_vline(x=record.sage_value, annotation_text="Sage", line_dash="dash")
fig.update_traces(marker_line_width=1, name="Espaloma")

app.layout = html.Div(
    style={"backgroundColor": colors["background"]},
    children=[
        dcc.Graph(id="graph", figure=fig),
        html.Div(
            [
                dcc.Markdown("hello world"),
            ],
            id="click-output",
        ),
    ],
)


# pasted in from ligand Molecule::to_svg
def draw_rdkit(mol):
    rdmol = mol.to_rdkit()
    rdDepictor.SetPreferCoordGen(True)
    rdDepictor.Compute2DCoords(rdmol)
    rdmol = rdMolDraw2D.PrepareMolForDrawing(rdmol)
    drawer = rdMolDraw2D.MolDraw2DSVG(300, 300)
    drawer.DrawMolecule(rdmol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


@callback(Output("click-output", "children"), Input("graph", "clickData"))
def display_click_data(clickData):
    if clickData:
        points = clickData["points"][0]["pointNumbers"]
        mols = {record.molecules[p] for p in points}
        pics = []
        for mol in mols:
            svg = draw_rdkit(
                Molecule.from_smiles(mol, allow_undefined_stereo=True)
            )
            try:
                encoded = base64.b64encode(bytes(svg, "utf-8"))
            except Exception as e:
                print("error: ", e)
            pics.append(
                html.Img(src=f"data:image/svg+xml;base64,{encoded.decode()}")
            )
        return pics


if __name__ == "__main__":
    app.run(debug=True)
