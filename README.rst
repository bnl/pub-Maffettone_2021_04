=======================================
pub-Maffettone_2021_04: Constrained NMF
=======================================

Python replication of `this paper <https://arxiv.org/abs/2104.00864>`_.
This work uses PyTorch for constrained non-negative matrix factorization.


Abstract
========
Non-negative Matrix Factorization (NMF) methods offer an appealing unsupervised learning method for real-time analysis of streaming spectral data in time-sensitive data collection, such as in situ characterization of materials. However, canonical NMF methods are optimized to reconstruct a full dataset as closely as possible, with no underlying requirement that the reconstruction produces components or weights representative of the true physical processes. In this work, we demonstrate how constraining NMF weights or components, provided as known or assumed priors, can provide significant improvement in revealing true underlying phenomena. We present a PyTorch based method for efficiently applying constrained NMF and demonstrate this on several synthetic examples. When applied to streaming experimentally measured spectral data, an expert researcher-in-the-loop can provide and dynamically adjust the constraints. This set of interactive priors to the NMF model can, for example, contain known or identified independent components, as well as functional expectations about the mixing of components. We demonstrate this application on measured X-ray diffraction and pair distribution function data from in situ beamline experiments. Details of the method are described, and general guidance provided to employ constrained NMF in extraction of critical information and insights during in situ and high-throughput experiments.


Getting Started
===============

Installation guide
******************


Install from github::

    $ python3 -m venv nmf_env
    $ source nmf_env/bin/activate
    $ git clone https://github.com/bnl/pub-Maffettone_2021_04
    $ cd pub-Maffettone_2021_04
    $ python -m pip install --upgrade pip wheel
    $ python -m pip install .


Recreating the published results
********************************
All figures presented in the paper and the supplementary information can be recreated by running the files in
`example_scripts` using `python -m`. For example ::

    $ python -m constrainednmf/example_scripts/simulation_constraining_weights.py

Will yield a simulated dataset of mixing Gaussians and the results from canonical NMF as well as constrained NMF where the
weights have been constrained.
