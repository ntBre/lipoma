import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

from twod import make_records

# dict of smirks -> Records
records = make_records("msm")


def plot(record):
    mat = np.column_stack((record.eqs, record.fcs))

    kmeans = KMeans(n_clusters=2).fit(mat)
    plt.scatter(mat[:, 0], mat[:, 1], c=kmeans.labels_)
    plt.show()
    plt.close()
