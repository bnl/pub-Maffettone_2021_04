import numpy as np
import matplotlib.pyplot as plt
from constrainednmf.companion.plotting import (
    waterfall_plot,
    two_column_width,
    one_column_width,
)
from bto import get_data as get_bto_data
from molten_salt import get_data as get_salt_data


def plot_datasets():
    fig, axes = plt.subplots(
        2, 1, figsize=(one_column_width, one_column_width * 1.5), tight_layout=True
    )
    T, X = get_bto_data()
    q = np.linspace(0.011231808788013649, 24.853167100343246, 3488)
    Q = np.array([q for _ in range(X.shape[0])])
    waterfall_plot(Q, X, alt_ordinate=T, ax=axes[0], sampling=2, offset=0.3)
    axes[0].set_xlim(0, 25)

    T, X, _ = get_salt_data()
    X = (X - X.min(dim=1).values[:, None]) / (
        X.max(dim=1).values - X.min(dim=1).values
    )[:, None]
    q = np.linspace(0.5, 7.0, X.shape[1])
    Q = np.array([q for _ in range(X.shape[0])])
    waterfall_plot(Q, X, alt_ordinate=T, ax=axes[1], sampling=2, offset=0.3)
    axes[1].set_xlim(0.5, 7.0)
    # axes[0].set_xlabel("Q [$\AA^{-1}$]")
    axes[1].set_xlabel("Q [$\AA^{-1}$]")
    axes[0].set_ylabel("Temperature [K]")
    axes[1].set_ylabel("Temperature [K]")
    for text, ax in zip(["(a)", "(b)"], axes):
        ax.text(
            -0.18,
            1.05,
            text,
            horizontalalignment="right",
            verticalalignment="top",
            transform=ax.transAxes,
            bbox=dict(facecolor="white", alpha=0.5, linewidth=0),
        )
    return fig, axes


def plot_pdf(data_dir=None):
    """Checking full pdf dataset and conversion to g(r)"""

    fig, axes = plt.subplots(
        2, 1, figsize=(two_column_width / 2, two_column_width), tight_layout=True
    )

    if data_dir is None:
        data_dir = data_dir = (
            Path(__file__).parents[1] / "example_data/NaCl_CrCl3_pdf_ramp"
        )
    paths = sorted(
        list(data_dir.glob("*.gr")),
        key=lambda path: float(str(path.name).split("_")[-2][1:-1]),
    )
    big_gs = list()
    lil_gs = list()
    T = list()
    for path in paths:
        x, y = np.loadtxt(path, comments="#", unpack=True)
        mask = np.logical_and(x > 0.0, x < 30.0)
        y = y[None, mask]
        x = x[mask]
        big_g = (y - y.min()) / (y.max() - y.min())
        lil_g = y / (4 * np.pi * x) + 1  # Converting G(r) to g(r)
        lil_g = (lil_g - lil_g.min()) / (lil_g.max() - lil_g.min())
        big_gs.append(big_g)
        lil_gs.append(lil_g)
        T.append(float(str(path.name).split("_")[-2][1:-1]) + 273.0)
    lil_gs = np.concatenate(lil_gs, axis=0)
    big_gs = np.concatenate(big_gs, axis=0)

    R = np.array([x for _ in range(big_gs.shape[0])])
    waterfall_plot(R, big_gs, alt_ordinate=T, ax=axes[0], sampling=5, offset=1.0)
    waterfall_plot(R, lil_gs, alt_ordinate=T, ax=axes[1], sampling=5, offset=1.0)
    axes[0].set_title("G(r)")
    axes[1].set_title("g(r)")

    return fig, axes


if __name__ == "__main__":
    from pathlib import Path

    path = Path(__file__).parent / "example_output"
    path.mkdir(exist_ok=True)
    fig, axes = plot_datasets()
    fig.savefig(path / "experimental_datasets.pdf")
    fig.show()

    fig, axes = plot_pdf()
    fig.savefig(path / "pdf_datasets.pdf")
    fig.show()
