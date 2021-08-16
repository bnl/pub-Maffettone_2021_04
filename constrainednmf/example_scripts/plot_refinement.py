import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from constrainednmf.companion.plotting import one_column_width



def get_data(json_path):
    df = pd.read_json(json_path)
    canonical = [
        list(df[df["mode"] == "Canonical NMF"][df["Ncomps"] == i].rwp)
        for i in range(3, 7)
    ]
    constrained = [
        list(df[df["mode"] == "Constrained NMF"][df["Ncomps"] == i].rwp)
        for i in range(3, 7)
    ]
    return constrained, canonical


def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def custom_violin(ax, data, facecolor):
    # https://matplotlib.org/stable/gallery/statistics/customized_violin.html#sphx-glr-gallery-statistics-customized-violin-py
    parts = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False, bw_method=0.5)

    for pc in parts["bodies"]:
        pc.set_facecolor(facecolor)
        pc.set_edgecolor("black")
        pc.set_alpha(1)

    quartile1, medians, quartile3 = np.array(
        [np.percentile(arr, [25, 50, 75]) for arr in data]
    ).T
    whiskers = np.array(
        [
            adjacent_values(sorted_array, q1, q3)
            for sorted_array, q1, q3 in zip(data, quartile1, quartile3)
        ]
    )
    whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]
    # whiskers_min = [np.min(arr) for arr in data]
    # whiskers_max = [np.max(arr) for arr in data]
    inds = np.arange(1, len(medians) + 1)
    ax.scatter(inds, medians, marker="o", color="white", s=20, zorder=3)
    ax.vlines(inds, quartile1, quartile3, color="k", linestyle="-", lw=3)
    ax.vlines(inds, whiskers_min, whiskers_max, color="k", linestyle="-", lw=1)
    return parts["bodies"][0]


def make_plot(*args):
    fig, ax = plt.subplots(
        1, 1, figsize=(one_column_width, one_column_width), tight_layout=True
    )
    parts = list()
    colors = ["C5", "C0"]
    for i, arg in enumerate(args):
        parts.append(custom_violin(ax, arg, colors[i]))

    ax.legend(parts, ["Canonical", "Constrained"], loc="upper left")
    return fig, ax


def plot_adjustments(ax):
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels([f"{i}" for i in range(3, 7)])
    ax.set_xlabel("Number of components")
    ax.set_ylabel("R-weighted pattern (Error)")


if __name__ == "__main__":
    path = Path(__file__).parent / ".." / "example_data" / "refinement_data.json"
    constrained, canonical = get_data(path)
    fig, ax = make_plot(canonical, constrained)
    plot_adjustments(ax)
    path = Path(__file__).parent / "example_output" / "refinement_data.pdf"
    fig.show()
    fig.savefig(path)
