<p align="center">
  <img src="logo/lfkit_logo.svg" width="150">
</p>

# LFKit

[![CI](https://img.shields.io/github/actions/workflow/status/cosmology-kit/lfkit/ci.yml?branch=main&label=CI&color=28A8C8&style=flat-square)](https://github.com/cosmology-kit/lfkit/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/github/actions/workflow/status/cosmology-kit/lfkit/docs.yml?branch=main&label=docs&color=28A8C8&style=flat-square)](https://github.com/cosmology-kit/lfkit/actions/workflows/docs.yml)
[![License](https://img.shields.io/github/license/cosmology-kit/lfkit?color=28A8C8&style=flat-square)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/py-lfkit?color=28A8C8&style=flat-square)](https://pypi.org/project/py-lfkit/)

**LFKit** is a modular Python toolkit for luminosity function modeling, photometric corrections,
and related utilities for cosmological analyses.

It is designed to provide reusable building blocks for working with luminosity functions, 
magnitude conversions, k-corrections, e-corrections, and photometric response handling in a
clear and extensible way.

## Features

LFKit currently includes tools for:

- luminosity function modeling
- Schechter and evolving luminosity function utilities
- magnitude and luminosity conversions
- k-corrections and e-corrections
- photometric response and filter-related helpers
- interpolation, validation, and general utility functions

The package is intended to stay modular, so components can be used independently or combined
into larger analysis pipelines.


## Documentation

[View the documentation](https://cosmology-kit.github.io/lfkit/main/)

## Installation

Install LFKit from PyPI:

```bash
python -m pip install py-lfkit
````

Clone the repository and install it locally:

```bash
git clone https://github.com/cosmology-kit/lfkit.git
cd lfkit
python -m pip install -e .
```


# Citation

If you use **LFKit** in your research, please cite it.

```bibtex
@software{sarcevic2026lfkit,
  title   = {LFKit: Luminosity functions and photometric corrections toolkit},
  author  = {Šarčević, Nikolina},
  year    = {2026},
  version = {0.1.4},
  url     = {https://github.com/cosmology-kit/lfkit}
}
```


## License
MIT License © 2025 Niko Šarčević, Matthijs van der Wild, and contributors.