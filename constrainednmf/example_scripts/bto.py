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
        data_dir = Path(__file__).parents[1] / "example_data/BaTiO3_xrd_ramp"
    paths = sorted(list(data_dir.glob("*.npy")))
    profiles = list()
    T = list()
    for path in paths:
        x = np.load(path)
        T.append(
            float(str(path).split("_")[-2][:3])
        )  # hardcoded nonsense for T from label
        profiles.append(x / x.max())
    X = torch.tensor(np.concatenate(profiles, axis=1).T, dtype=torch.float)
    T = np.array(T)
    return T, X


def standard_nmf(X):
    nmf = NMF(X.shape, 4)
    n_iter = nmf.fit(X, beta=2, tol=1e-8, max_iter=1000)
    return nmf


def plot_adjustments(axes, norm=True):
    axes[0].set_xlabel(r"2$\theta$ [deg]")
    axes[0].set_xlim(0, 25)
    for ax in axes[1:]:
        ax.axvline(185, linestyle="--", color="k")
        ax.axvline(280, linestyle="--", color="k")
        ax.axvline(400, linestyle="--", color="k")
        if norm:
            ax.set_ylim(0, 1)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_yticklabels([0, 0.25, 0.5, 0.75, 1])
        ax.set_xlabel("Temperature [K]")
        # ax.set_ylabel("$x_\Phi$")
        ax.set_xlim(150, 445)
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


def make_plots():
    from constrainednmf.nmf.utils import iterative_nmf
    from constrainednmf.nmf.metrics import Euclidean, r_factor

    T, X = get_data()
    tth = np.linspace(0.011231808788013649, 24.853167100343246, 3488)
    figs = list()
    axes = list()
    fig, ax = sweep_components(X, n_max=8)
    figs.append(fig),
    axes.append(ax)
    nmf = standard_nmf(X)
    fig, ax = decomp_plot(nmf, T, x=tth)
    plot_adjustments(ax, norm=False)
    figs.append(fig)
    axes.append(ax)
    print(f"Standard MSE={Euclidean(nmf.reconstruct(nmf.H, nmf.W), X)}")
    print(f"Standard R-factor={r_factor(nmf.reconstruct(nmf.H, nmf.W), X)}")
    for nmf in iterative_nmf(NMF, X, n_components=4, beta=2, tol=1e-8, max_iter=1000):
        fig, ax = decomp_plot(nmf, T, x=tth)
        plot_adjustments(ax)
        figs.append(fig)
        axes.append(ax)
    print(f"AutoConstrained MSE={Euclidean(nmf.reconstruct(nmf.H, nmf.W), X)}")
    print(f"AutoConstrained R-factor={r_factor(nmf.reconstruct(nmf.H, nmf.W), X)}")

    return figs


if __name__ == "__main__":
    path = Path(__file__).parent / "example_output"
    path.mkdir(exist_ok=True)
    figs = make_plots()
    for i, fig in enumerate(figs):
        fig.tight_layout()
        fig.show()
        fig.savefig(path / f"BaTiO_{i}.pdf")
