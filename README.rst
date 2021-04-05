=======================================
pub-Maffettone_2021_04: Constrained NMF
=======================================

.. image:: https://img.shields.io/travis/bnl/pub-maffettone_2021_04.svg
        :target: https://travis-ci.org/bnl/pub-maffettone_2021_04

.. image:: https://img.shields.io/pypi/v/pub-maffettone_2021_04.svg
        :target: https://pypi.python.org/pypi/pub-maffettone_2021_04


Python replication of `this paper: <https://arxiv.org/abs/2104.00864>`_.

* Free software: 3-clause BSD license

Overview
========
Non-negative Matrix Factorization (NMF) methods offer an appealing unsupervised learning method for real-time analysis of streaming spectral data in time-sensitive data collection, such as in situ characterization of materials. However, canonical NMF methods are optimized to reconstruct a full dataset as closely as possible, with no underlying requirement that the reconstruction produces components or weights representative of the true physical processes. In this work, we demonstrate how constraining NMF weights or components, provided as known or assumed priors, can provide significant improvement in revealing true underlying phenomena. We present a PyTorch based method for efficiently applying constrained NMF and demonstrate this on several synthetic examples. When applied to streaming experimentally measured spectral data, an expert researcher-in-the-loop can provide and dynamically adjust the constraints. This set of interactive priors to the NMF model can, for example, contain known or identified independent components, as well as functional expectations about the mixing of components. We demonstrate this application on measured X-ray diffraction and pair distribution function data from in situ beamline experiments. Details of the method are described, and general guidance provided to employ constrained NMF in extraction of critical information and insights during in situ and high-throughput experiments.
