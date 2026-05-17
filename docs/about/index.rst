.. |lfkitlogo| image:: /_static/logos/lfkit_logo.png
   :alt: LFKit logo black
   :width: 50px

|lfkitlogo| About LFKit
=======================

LFKit is a Python toolkit for working with galaxy luminosity functions
and the photometric quantities required to construct them from
observational data.

The package provides lightweight building blocks for converting observed
galaxy photometry into rest-frame quantities and for modelling galaxy
luminosity functions used in astrophysical and cosmological analyses.


Core components
^^^^^^^^^^^^^^^

.. grid:: 2
   :gutter: 3

   .. grid-item-card::
      :link: corrections_overview
      :link-type: doc
      :shadow: md

      **Photometric Corrections**
      ^^^

      Tools for evaluating photometric **k-corrections** and
      **luminosity evolution (e) corrections**, and for working with
      filter response curves and spectral energy distributions (SEDs).

   .. grid-item-card::
      :link: photometry_overview
      :link-type: doc
      :shadow: md

      **Photometric Tools**
      ^^^

      Utilities for constructing and analyzing galaxy luminosity
      functions, including conversions between observed magnitudes
      and rest-frame absolute magnitudes.


.. toctree::
   :maxdepth: 1
   :hidden:

   corrections_overview
   photometry_overview