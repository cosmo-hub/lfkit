.. |lfkitlogo| image:: /_static/logos/lfkit_logo-icon.png
   :alt: LFKit logo
   :width: 60px

|lfkitlogo| LFKit
=================

**LFKit** is a Python toolkit for modelling galaxy luminosity functions,
photometric corrections, magnitude conversions, and magnitude-limited catalog
selection.

It provides a modular interface for turning luminosity function models into
number densities, completeness fractions, LF-weighted redshift trends, and
observable or missing catalog populations. The same machinery can be used in
photometric redshift modelling, intrinsic alignment modelling, cluster science,
GW-cosmology catalog completeness, survey forecasting, or any analysis that
connects galaxy luminosities, magnitudes, redshift evolution, and observed
catalog limits.

Getting started
---------------

Start with the theory overview for the main conventions and definitions, or use
the examples section for executable workflows with plots.

.. grid:: 2
   :gutter: 2

   .. grid-item-card::
      :link: about/index
      :link-type: doc
      :shadow: md

      **Theory and overview**
      ^^^
      Core concepts, conventions, and package scope, including luminosity
      functions, photometry, corrections, and catalog selection.

   .. grid-item-card::
      :link: examples/index
      :link-type: doc
      :shadow: md

      **Examples**
      ^^^
      Runnable examples for luminosity function models, magnitude conversions,
      integrals, corrections, redshift trends, and catalog completeness.

Documentation
-------------

.. toctree::
   :maxdepth: 1

   installation
   about/index
   examples/index
   api/index
   citation
   contributing
   license
