# read the results produced by query.py and generate useful graphics

import os
import shutil
from io import StringIO

import click
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sea
from openff.toolkit import ForceField

# Usage:
# python parse_query.py -i data/bonds_dedup.dat -o /tmp/bonds [-p]


@click.command()
@click.option("--infile", "-i")
@click.option("--outdir", "-o")
@click.option("--plot", "-p", default=True, is_flag=True)
def main(infile, outdir, plot):
    if os.path.exists(outdir):
        shutil.rmtree(outdir)
    os.makedirs(outdir)

    ff = ForceField("openff-2.1.0.offxml")
    bh = ff.get_parameter_handler("Bonds")
    ah = ff.get_parameter_handler("Angles")
    th = ff.get_parameter_handler("ProperTorsions")

    pmap = {h.smirks: h.id for h in bh}
    pmap.update({h.smirks: h.id for h in ah})
    pmap.update({h.smirks: h.id for h in th})

    buf = StringIO()
    with open(infile, "r") as inp:
        for i, line in enumerate(inp):
            if line.startswith("#"):
                continue

            sp = line.split()
            [smirks, count, sage, *rest] = sp
            assert int(count) == len(rest)
            data = [float(r) for r in rest]
            esp_avg = np.average(data)
            buf.write(f"{smirks} {esp_avg}\n")

            if plot:
                ax = sea.histplot(data=data, label="Espaloma")
                ax.axvline(x=float(sage), color="green", label="Sage")
                ax.axvline(x=esp_avg, color="orange", label="Espaloma Avg.")
                fig = ax.get_figure()
                pid = pmap[smirks]
                plt.legend()
                plt.title(f"{pid}: {smirks}\navg = {esp_avg}")
                fig.savefig(f"{outdir}/{i:05d}.png", dpi=300)
                plt.close()

    with open(f"{outdir}/output.dat", "w") as out:
        out.write(buf.getvalue())


if __name__ == "__main__":
    main()
