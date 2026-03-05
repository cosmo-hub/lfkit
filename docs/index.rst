.. raw:: html

   <h1 class="lfkit-title">
     <span class="lfkit-blue">LFKit</span>
     <span class="lfkit-red">Documentation</span>
   </h1>

LFKit is a Python library for **luminosity function (LF) modeling** and related
photometric utilities.

Rather than computing luminosity functions directly from catalogs, LFKit
provides **general LF model families, survey presets, and helper utilities**
that make it easier to fit luminosity function parameters within your own
analysis pipeline (for example using **GAMA-style survey setups**).

The package also includes a lightweight module for **k- and e-corrections**,
designed to integrate easily with inference frameworks.

Overview
--------

+-------------------------------+--------------------------------------------+
| **LFKit provides**            | **LFKit does not**                         |
+===============================+============================================+
| Flexible luminosity function  | Measure luminosity functions from raw      |
| model implementations         | galaxy catalogs                            |
+-------------------------------+--------------------------------------------+
| Standard LF parameterizations | Perform full survey simulations            |
+-------------------------------+--------------------------------------------+
| Survey presets for common     | Replace photometry pipelines               |
| observational setups          |                                            |
+-------------------------------+--------------------------------------------+
| Utilities for fitting LF      | Perform inference automatically            |
| parameters                    | (you supply your own optimizer/MCMC)       |
+-------------------------------+--------------------------------------------+
| Simple k- and e-correction    | Provide full photometric calibration       |
| utilities                     | frameworks                                 |
+-------------------------------+--------------------------------------------+

Documentation
-------------

.. grid:: 3
   :gutter: 2

   .. grid-item-card::
      :link: about/index
      :link-type: doc
      :shadow: md

      **About LFKit**
      ^^^
      Project scope, design goals, and background.

   .. grid-item-card::
      :link: examples/index
      :link-type: doc
      :shadow: md

      **Examples**
      ^^^
      Example workflows using LF models and presets.

   .. grid-item-card::
      :link: api/modules
      :link-type: doc
      :shadow: md

      **API Reference**
      ^^^
      Full reference for modules, classes, and functions.

.. toctree::
   :maxdepth: 2
   :hidden:

   about/index
   examples/index
   api/modules