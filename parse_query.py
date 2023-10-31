# read the results produced by query.py and generate useful graphics

import os
import re
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
    ih = ff.get_parameter_handler("ImproperTorsions")

    pmap = {h.smirks: h.id for h in bh}
    pmap.update({h.smirks: h.id for h in ah})
    pmap.update({h.smirks: h.id for h in th})
    pmap.update({h.smirks: h.id for h in ih})

    buf = StringIO()
    averages = []
    with open(infile, "r") as inp:
        for i, line in enumerate(inp):
            if line.startswith("#"):
                continue

            sp = line.split()
            [smirks, count, sage, *rest] = sp
            assert int(count) == len(rest)
            data = [float(r) for r in rest]
            esp_avg = np.average(data)

            sage = float(sage)

            diff = [abs(d - sage) / d for d in data if d != 0.0]
            if len(diff) > 0:
                avg_diff = np.average(diff)
                std_diff = np.std(diff)

                averages.append(avg_diff)

            buf.write(f"{smirks} {esp_avg} {sage} {avg_diff} {std_diff}\n")

            if plot:
                ax = sea.histplot(data=data, label="Espaloma")
                ax.axvline(x=sage, color="green", label="Sage")
                ax.axvline(x=esp_avg, color="orange", label="Espaloma Avg.")
                fig = ax.get_figure()
                id_key = re.sub(r"-k[123]$", "", smirks)
                pid = pmap[id_key]
                if id_key != smirks:
                    title = f"{pid} {smirks[-2:]}"  # append k[123] to pid
                else:
                    title = pid
                plt.legend()
                plt.title(f"{title}: {smirks}\navg = {esp_avg}")
                fig.savefig(f"{outdir}/{i:05d}.png", dpi=300)
                plt.close()

    with open(f"{outdir}/output.dat", "w") as out:
        out.write(buf.getvalue())

    print(f"average MA%E: {np.average(averages) * 100: .2f}")


if __name__ == "__main__":
    main()
