.. |lfkitlogo| image:: /_static/logos/lfkit_logo.png
   :alt: LFKit logo black
   :width: 50px


|lfkitlogo| Installation
========================

LFKit can be installed from PyPI. The PyPI distribution name is ``py-lfkit``::

   pip3 install py-lfkit

LFKit can additionally be installed from source.
To create an editable installation, clone the repository and run::

   git clone https://github.com/cosmology-kit/lfkit.git
   cd lfkit
   pip install -e .

Alternatively, install directly from GitHub with ``pip``::

   pip install git+https://github.com/cosmology-kit/lfkit.git


Development installation
------------------------

For development work, clone the repository and install LFKit in editable mode
with the development dependencies::

   git clone https://github.com/cosmology-kit/lfkit.git
   cd lfkit
   pip install -e ".[dev]"

This allows local changes to the source code to be picked up without
reinstalling the package.


Optional dependencies
---------------------

Some LFKit examples use optional third-party packages for cosmology calculations,
photometric corrections, or plotting. Install these separately when needed by a
specific example page.

For example, if an example uses CCL, install CCL following the instructions for
your system and Python environment before running that example.
