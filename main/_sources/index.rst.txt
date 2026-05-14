.. |lfkitlogo| image:: /_static/logos/lfkit_logo-icon.png
   :alt: LFKit logo
   :width: 60px

|lfkitlogo| LFKit
=================

**LFKit** is a toolkit for modelling galaxy luminosity functions,
photometric corrections, and magnitude-limited catalog completeness.

It provides a clean interface for building theoretical luminosity functions,
including redshift-dependent parameter models, and for connecting apparent
magnitude limits to observable and missing galaxy number densities.

LFKit is designed to be science-use-case agnostic: the same luminosity function
machinery can be used in photometric-redshift modelling, intrinsic-alignment
modelling, cluster science, GW-cosmology catalog completeness, or any other
analysis that needs luminosity function-based number densities.

Getting started
---------------

Use the examples section for runnable workflows with plots, or the API
reference for detailed documentation of the public classes and functions.

.. grid:: 3
   :gutter: 2

   .. grid-item-card::
      :link: about/index
      :link-type: doc
      :shadow: md

      **About**
      ^^^
      Overview of the package, scope, and design choices.

   .. grid-item-card::
      :link: examples/index
      :link-type: doc
      :shadow: md

      **Examples**
      ^^^
      Runnable examples for luminosity functions, corrections, and catalog
      completeness.

   .. grid-item-card::
      :link: api/index
      :link-type: doc
      :shadow: md

      **API reference**
      ^^^
      Public classes, functions, and modules.

.. toctree::
   :maxdepth: 2
   :caption: Documentation
   :hidden:

   about/index
   examples/index
   api/index