from constrainednmf.nmf.models import NMF
import numpy as np
import torch
from pathlib import Path
from constrainednmf.companion.plotting import decomp_plot
from constrainednmf.companion.plotting import sweep_components

torch.manual_seed(1234)
np.random.seed(1234)


def get_data(data_dir=None):
    if data_dir is None:
        data_dir = Path(__file__).parents[1] / "example_data/NaCl_CrCl3_pdf_ramp"
    paths = sorted(
        list(data_dir.glob("*.chi")),
        key=lambda path: float(str(path.name).split("_")[-2][1:-1]),
    )
    profiles = list()
    T = list()
    for path in paths:
        x, y = np.loadtxt(path, unpack=True)
        mask = np.logical_and(x > 0.5, x < 7.0)
        profiles.append(y[None, mask])
        T.append(float(str(path.name).split("_")[-2][1:-1]) + 273.0)
    X = torch.tensor(np.concatenate(profiles, axis=0), dtype=torch.float)
    T = np.array(T)
    return T, X, x[mask]


def get_pdf_data(data_dir=None):
    if data_dir is None:
        data_dir = data_dir = (
            Path(__file__).parents[1] / "example_data/NaCl_CrCl3_pdf_ramp"
        )
    paths = sorted(
        list(data_dir.glob("*.gr")),
        key=lambda path: float(str(path.name).split("_")[-2][1:-1]),
    )
    profiles = list()
    T = list()
    for path in paths:
        x, y = np.loadtxt(path, comments="#", unpack=True)
        mask = np.logical_and(x > 5.0, x < 30.0)
        y = y[None, mask]
        # x = x[mask]
        # y = y / (4 * np.pi * x) + 1  # Converting G(r) to g(r)
        y = (y - y.min()) / (y.max() - y.min())
        profiles.append(y)
        T.append(float(str(path.name).split("_")[-2][1:-1]) + 273.0)
    X = torch.tensor(np.concatenate(profiles, axis=0), dtype=torch.float)
    return T, X


def standard_nmf(X, n_components=4):
    nmf = NMF(X.shape, n_components)
    n_iter = nmf.fit(X, beta=2, tol=1e-8, max_iter=1000)
    return nmf


def plot_adjustments(axes):
    for ax in axes[1:]:
        ax.set_xlabel("Temperature [K]")
        # ax.set_ylabel("$x_\Phi$")
        ax.set_xlim(303, 963)
    for ax in axes[2:]:
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels([0, 0.25, 0.5, 0.75, 1])
    for text, ax in zip(["(a)", "(b)"], axes):
        ax.text(
            -0.1,
            1.1,
            text,
            horizontalalignment="right",
            verticalalignment="top",
            transform=ax.transAxes,
            bbox=dict(facecolor="white", alpha=0.5, linewidth=0),
        )


def make_plots(n_components=4, function="XRD"):
    from constrainednmf.nmf.utils import iterative_nmf
    from constrainednmf.nmf.metrics import Euclidean, r_factor

    if function.upper() == "XRD":
        T, X, q = get_data()
    elif function.upper() == "PDF":
        T, X = get_pdf_data()
        q = None
    figs = dict()
    axes = list()
    fig, ax = sweep_components(X, n_max=10)
    figs["elbow"] = fig
    axes.append(ax)
    nmf = standard_nmf(X, n_components)
    fig, ax = decomp_plot(nmf, T, x=q)
    plot_adjustments(ax)
    figs["conventional"] = fig
    axes.append(ax)
    print(
        f"Standard  {n_components} components MSE={Euclidean(nmf.reconstruct(nmf.H, nmf.W), X)}"
    )
    print(
        f"Standard  {n_components} components R-factor={r_factor(nmf.reconstruct(nmf.H, nmf.W), X)}"
    )
    for i, nmf in enumerate(
        iterative_nmf(
            NMF, X, n_components=n_components, beta=2, tol=1e-8, max_iter=1000
        )
    ):
        fig, ax = decomp_plot(nmf, T, x=q)
        plot_adjustments(ax)
        figs[f"iterative_{i+1}"] = fig
        axes.append(ax)
    print(
        f"AutoConstrained {n_components} components MSE={Euclidean(nmf.reconstruct(nmf.H, nmf.W), X)}"
    )
    print(
        f"AutoConstrained {n_components} components R-factor={r_factor(nmf.reconstruct(nmf.H, nmf.W), X)}"
    )

    return figs


def make_summary_plot(function="XRD"):
    import matplotlib.pyplot as plt
    from constrainednmf.nmf.utils import iterative_nmf
    from constrainednmf.companion.plotting import two_column_width

    if function.upper() == "XRD":
        T, X, q = get_data()
    elif function.upper() == "PDF":
        T, X = get_pdf_data()
        q = None
    fig, axes = plt.subplots(
        3, 2, figsize=(two_column_width, two_column_width), tight_layout=True
    )

    nmfs = []
    for n in range(3, 6):
        nmfs.append(
            iterative_nmf(NMF, X, n_components=n, beta=2, tol=1e-8, max_iter=1000)[-1]
        )
        np.savetxt(f"./example_output/molten_salts_{n}_components.txt", nmfs[-1].H.detach().numpy())
    for i, nmf in enumerate(nmfs):
        decomp_plot(nmf, T, axes=axes[i, :], x=q)
        plot_adjustments(axes[i, :])

    return fig


def make_residual(function="XRD"):
    from constrainednmf.nmf.utils import iterative_nmf
    from constrainednmf.companion.plotting import residual_plot

    if function.upper() == "XRD":
        T, X, _ = get_data()
    elif function.upper() == "PDF":
        T, X = get_pdf_data()

    nmf = iterative_nmf(NMF, X, n_components=4, beta=2, tol=1e-8, max_iter=1000)[-1]

    return residual_plot(nmf, X)


if __name__ == "__main__":
    path = Path(__file__).parent / "example_output"
    path.mkdir(exist_ok=True)
    fig = make_summary_plot()
    fig.show()
    fig.savefig(path / f"molten_salts_xrd_summary.pdf")
    for n in range(3, 7):
        figs = make_plots(n)
        for key in figs:
            figs[key].tight_layout()
            figs[key].show()
            figs[key].savefig(path / f"molten_salts_xrd_{n}_{key}.pdf")

    fig = make_summary_plot(function="pdf")
    fig.show()
    fig.savefig(path / f"molten_salts_pdf_summary.pdf")
    for n in range(3, 7):
        figs = make_plots(n, function="pdf")
        for key in figs:
            figs[key].tight_layout()
            figs[key].show()
            figs[key].savefig(path / f"molten_salts_pdf_{n}_{key}.pdf")
