# read the results produced by query.py and generate useful graphics

import matplotlib.pyplot as plt
import seaborn as sea
import numpy as np

with open("query.dat", "r") as inp:
    for line in inp:
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

        fig.savefig("out.png", dpi=300)
