import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
import torch

one_column_width = 3.37
two_column_width = 6.69
linewidth = 0.5
fontsize = 8
mpl.rcParams.update({"font.size": fontsize, "lines.linewidth": linewidth})
mpl.style.use("tableau-colorblind10")


def subplot_label(ax, text):
    return ax.annotate(
        text,
        # units are (axes-fraction, axes-fraction)
        # # this is bottom right
        # xy=(1, 0),
        # this is the top left
        xy=(0, 1),
        xycoords="axes fraction",
        # units are absolute offset in points from xy
        xytext=(-5, 5),
        textcoords=("offset points"),
        # set the text alignment
        ha="right",
        va="bottom",
        fontweight="bold",
        fontsize="larger",
    )


def plot_dataset(x, Y):
    mpl.rcParams.update({"font.size": fontsize, "lines.linewidth": linewidth})

    fig, ax = plt.subplots(
        figsize=(one_column_width, one_column_width * 0.75), tight_layout=True
    )
    Y = np.array(Y)
    X = np.array([x for _ in range(Y.shape[0])])
    offset = np.max(Y) * 0.3
    sample = max(1, Y.shape[0] // 20)
    waterfall_plot(X, Y, ax=ax, sampling=sample, offset=offset)
    ax.set_ylabel("Dataset Index")
    return fig


def waterfall_plot(
    xs,
    ys,
    alt_ordinate=None,
    ax=None,
    sampling=1,
    offset=1.0,
    cmap="coolwarm",
    **kwargs,
):
    if ax is None:
        fig, ax = plt.subplots(
            figsize=(one_column_width, one_column_width), tight_layout=True
        )

    indicies = range(0, xs.shape[0])[::sampling]

    cmap = mpl.cm.get_cmap(cmap)
    norm = mpl.colors.Normalize(vmin=0, vmax=xs.shape[0] // sampling)

    if alt_ordinate is not None:
        idxs, labels = list(
            zip(*sorted(zip(range(ys.shape[0]), alt_ordinate), key=lambda x: x[1]))
        )
    else:
        idxs = list(range(ys.shape[0]))
        labels = list(range(ys.shape[0]))

    for plt_i, idx in enumerate(indicies):
        y = ys[idx, :]
        y = y + plt_i * offset
        x = xs[idx, :]
        ax.plot(x, y, color=cmap(norm(plt_i)))

    ax.set_ylim((0, len(indicies) * offset + 1))
    ax.set_yticks([0, (len(indicies) // 2) * offset, (len(indicies)) * offset])
    ax.set_yticklabels([labels[0], labels[len(labels) // 2], labels[-1]])


def toy_plot(model, x, Y, weights, components):
    #fontsize = 6
    mpl.rcParams.update({"font.size": fontsize, "lines.linewidth": linewidth})
    fig = plt.figure(
        tight_layout=False, figsize=(two_column_width, two_column_width / 3 * 2)
    )  # Not using tight_layout permenantly. Applying prior to legend.
    gs = GridSpec(2, 6)
    recon_axes = [fig.add_subplot(gs[0, i * 2 : (i + 1) * 2]) for i in range(3)]
    comp_ax = fig.add_subplot(gs[1, :3])
    weight_ax = fig.add_subplot(gs[1, 3:])

    with torch.no_grad():
        recon = model.reconstruct(model.H, model.W)

    ax = recon_axes[0]
    ax.plot(x, recon[0, :].data.numpy(), color="tab:red", label="Learned")
    ax.plot(x, Y[0, :], color="black", linestyle="--", label="True")
    ax = recon_axes[1]
    ax.plot(
        x, recon[recon.shape[0] // 2, :].data.numpy(), color="tab:red", label="Learned"
    )
    ax.plot(x, Y[Y.shape[0] // 2, :], color="black", linestyle="--", label="True")
    ax = recon_axes[2]
    ax.plot(x, recon[-1, :].data.numpy(), color="tab:red", label="Learned")
    ax.plot(x, Y[-1, :], color="black", linestyle="--", label="True")
    recon_axes[-1].legend(loc="upper right",
                          #fontsize="small",
                          framealpha=0.4)
    for text, ax in zip(["(a)", "(b)", "(c)"], recon_axes):
        # ax.legend(loc="center left", fontsize="small", framealpha=0.25)
        ax.text(
            0.05,
            0.98,
            text,
            horizontalalignment="center",
            verticalalignment="top",
            transform=ax.transAxes,
            # bbox=dict(facecolor="white", alpha=0.5, linewidth=0),
        )

    H = model.H.data.numpy()
    ax = comp_ax
    for i in range(H.shape[0]):
        line = ax.plot(x, H[i, :], label=f"Learned {i + 1}")
        ax.plot(
            x,
            components[i, :],
            linestyle="--",
            dashes=(5, 10),
            color=line[0].get_color(),
            label=f"True {i + 1}",
        )
    # ax.set_title("Learned Components")
    # ax.legend(loc="center left", fontsize="small", framealpha=0.25)
    ax.text(
        0.05,
        0.96,
        "(d)",
        horizontalalignment="center",
        verticalalignment="top",
        transform=ax.transAxes,
        # bbox=dict(facecolor="white", alpha=0.5, linewidth=0),
    )

    W = model.W.data.numpy()
    ax = weight_ax
    for i in range(W.shape[1]):
        line = ax.plot(W[:, i], label=f"Learned {i + 1}")
        ax.plot(
            weights.T[:, i],
            linestyle="--",
            dashes=(5, 10),
            color=line[0].get_color(),
            label=f"True {i + 1}",
        )
    # ax.set_title("Learned Weights")
    plt.tight_layout()
    ax.legend(
        bbox_to_anchor=(-0.2, -0.2),
        loc="upper center",
        ncol=W.shape[1],
        #fontsize="small",
        framealpha=0.25,
    )
    ax.text(
        0.05,
        0.96,
        "(e)",
        horizontalalignment="center",
        verticalalignment="top",
        transform=ax.transAxes,
        bbox=dict(facecolor="white", alpha=0.5, linewidth=0),
    )
    return fig


def decomp_plot(nmf, T, axes=None, x=None):
    mpl.rcParams.update({"font.size": fontsize, "lines.linewidth": linewidth})
    H = nmf.H.data.numpy()
    W = nmf.W.data.numpy()
    if axes is None:
        fig, axes = plt.subplots(2, 1, figsize=(one_column_width, 4), tight_layout=True)
    ax = axes[0]
    if x is None:
        x = np.arange(0, H.shape[1])
        xlim = (0, H.shape[1])
        xlabel = "Component feature index"
    else:
        xlim = (np.min(x), np.max(x))
        xlabel = r"Q [$\AA^{-1}$]"
    for i in range(H.shape[0]):
        ax.plot(x, H[i, :] / H[i, :].max() + i)
    # ax.set_title("Stacked Normalized Components")
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Normalized intensity")

    ax = axes[1]
    for i in range(W.shape[1]):
        ax.plot(T, W[:, i])
    ax.set_ylabel("Weights")

    # ax = axes[2]
    # for i in range(W.shape[1]):
    #     ax.plot(T, W[:, i] / W[:, i].max())
    # ax.set_title("Normalized Weights")
    return axes[0].figure, axes


def residual_plot(nmf, X, idx=None, ax=None):
    """Plot X, reconstruction, and residual of an index"""
    mpl.rcParams.update({"font.size": fontsize, "lines.linewidth": linewidth})

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(one_column_width, one_column_width), tight_layout=True
        )
    if idx is None:
        idx = 0

    x_pred = nmf.reconstruct(nmf.H, nmf.W)[idx, :].detach().numpy()
    x_true = X[idx, :].detach().numpy()
    residual = x_true - x_pred

    ax.plot(x_true + residual.max(), "k-")
    ax.plot(x_pred + residual.max(), "b--")
    ax.plot(residual, "r")
    return fig


def sweep_components(X, n_max=None, n_min=2):
    """
    Sweeps over all values of n_components and returns a plot of losses vs n_components
    Parameters
    ----------
    X : Tensor
    n_max : int
        Max n in search, default X.shape[0]
    n_min : int
        Min n in search, default 2

    Returns
    -------
    fig
    """
    from constrainednmf.nmf.models import NMF
    mpl.rcParams.update({"font.size": fontsize, "lines.linewidth": linewidth})

    if n_max is None:
        n_max = X.shape[0]

    losses = list()
    kl_losses = list()
    for n_components in range(n_min, n_max + 1):
        nmf = NMF(X.shape, n_components)
        nmf.fit(X, beta=2, tol=1e-8, max_iter=500)
        losses.append(nmf.loss(X, beta=2))
        kl_losses.append(nmf.loss(X, beta=1))

    fig = plt.figure(
        figsize=(one_column_width, one_column_width / 2), tight_layout=True
    )  # create a figure object
    ax = fig.add_subplot(1, 1, 1)
    axes = [ax, ax.twinx()]
    x = list(range(n_min, n_max + 1))
    axes[0].plot(x, losses, color="C0", label="MSE Loss")
    axes[0].set_ylabel("MSE Loss", color="C0")
    axes[1].plot(x, kl_losses, color="C1", label="KL Loss")
    axes[1].set_ylabel("KL Loss", color="C1")
    ax.set_xlabel("Number of components")
    return fig, axes
