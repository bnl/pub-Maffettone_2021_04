import torch


def normalize(x: torch.Tensor, axis=0) -> torch.Tensor:
    return x / x.sum(axis, keepdim=True)


def iterative_nmf(
    NMFClass, X, n_components, *, beta=2, tol=1e-8, max_iter=1000, **kwargs
):
    """
    Utility for performing NMF on a stream of data along a common state variable
    (Temperature or composition), that coincides with the data ordering.

    Parameters
    ----------
    NMFClass : class
        Child class of NMFBase
    X : Tensor
        Data to perform NMF on
    n_components : int
        Number of components for NMF
    beta : int
        Beta for determining loss function
    tol : float
        Optimization tolerance
    max_iter : int
        Maximum optimization iterations
    kwargs : dict
        Passed to intialization of NMF

    Returns
    -------
    nmfs : list of NMF instances

    """
    nmfs = list()
    initial_components = [torch.rand(1, X.shape[-1]) for _ in range(n_components)]
    fix_components = [False for _ in range(n_components)]

    # Start on bounding the outer components
    initial_components[0] = X[0, :].reshape(1, -1)
    fix_components[0] = True
    initial_components[-1] = X[-1, :].reshape(1, -1)
    fix_components[-1] = True

    nmf = NMFClass(
        X.shape,
        n_components,
        initial_components=initial_components,
        fix_components=fix_components,
        **kwargs
    )
    nmf.fit(X, beta=beta, tol=tol, max_iter=max_iter)
    nmfs.append(nmf)

    if len(nmf.W.shape) == 3:
        convolutional = True
    else:
        convolutional = False
    visited = {0, n_components - 1}
    for _ in range(n_components - 2):

        # Find next most prominent weight
        if convolutional:
            indices = (
                nmf.W.sum(axis=-1).max(axis=0).values.argsort(descending=True).numpy()
            )
        else:
            indices = nmf.W.max(axis=0).values.argsort(descending=True).numpy()
        for i in indices:
            if i in visited:
                continue
            else:
                visited.add(i)
                weight_idx = i
                break

        # Find most important component to that weight
        if convolutional:
            pattern_idx = int(nmf.W.sum(axis=-1).argmax(axis=0)[weight_idx])
        else:
            pattern_idx = int(nmf.W.argmax(axis=0)[weight_idx])

        # Lock and run
        initial_components[weight_idx] = X[pattern_idx, :].reshape(1, -1)
        fix_components[weight_idx] = True
        nmf = NMFClass(
            X.shape,
            n_components,
            initial_components=initial_components,
            fix_components=fix_components,
            **kwargs
        )
        nmf.fit(X, beta=beta, tol=tol, max_iter=max_iter)
        nmfs.append(nmf)

    return nmfs
