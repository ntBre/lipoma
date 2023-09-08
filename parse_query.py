# read the results produced by query.py and generate useful graphics

import click
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sea


@click.command()
@click.option("--infile")
def main(infile):
    with open(infile, "r") as inp:
        for i, line in enumerate(inp):
            if line.startswith("#"):
                continue
            sp = line.split()
            [smirks, count, sage, *rest] = sp
            assert int(count) == len(rest)
            data = [float(r) for r in rest]
            ax = sea.histplot(data=data, label="Espaloma")
            esp_avg = np.average(data)
            ax.axvline(x=float(sage), color="green", label="Sage")
            ax.axvline(x=esp_avg, color="orange", label="Espaloma Avg.")
            fig = ax.get_figure()
            plt.legend()

            fig.savefig(f"output/{i:05d}.png", dpi=300)
            plt.close()


if __name__ == "__main__":
    main()
