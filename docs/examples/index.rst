.. |lfkitlogo| image:: /_static/logos/lfkit_logo-icon.png
   :alt: LFKit logo
   :width: 50px

|lfkitlogo| Examples
====================

Working, executable examples for using LFKit.

.. grid:: 2
   :gutter: 2

   .. grid-item-card::
      :link: kcorrect_examples
      :link-type: doc
      :shadow: md

      **kcorrect examples**
      ^^^
      Build :math:`k(z)` with the kcorrect backend from a single rest-frame
      color anchor, compare anchors, and explore output-band choices.

      +++
      *Backends:* kcorrect

   .. grid-item-card::
      :link: poggianti_examples
      :link-type: doc
      :shadow: md

      **Poggianti (1997) examples**
      ^^^
      Evaluate :math:`k(z)`, :math:`e(z)`, and :math:`k(z)-e(z)` from the
      Poggianti tabulations; compare galaxy types and bands.

      +++
      *Backends:* Poggianti (1997)

   .. grid-item-card::
      :link: luminosity_function_examples
      :link-type: doc
      :shadow: md

      **Luminosity-function examples**
      ^^^
      Build standard, evolving, and double Schechter luminosity functions,
      evaluate :math:`\phi(M, z)`, and compare model behaviour.

      +++
      *API:* LuminosityFunction

   .. grid-item-card::
      :link: conditional_luminosity_function_examples
      :link-type: doc
      :shadow: md

      **Conditional luminosity-function examples**
      ^^^
      Build conditional luminosity functions, including redshift-dependent
      Schechter models and central/satellite components.

      +++
      *API:* LuminosityFunction

   .. grid-item-card::
      :link: catalog_completeness_examples
      :link-type: doc
      :shadow: md

      **Catalog-completeness examples**
      ^^^
      Compute observed and missing number densities for a magnitude-limited
      catalog and visualize completeness as a function of redshift.

      +++
      *API:* LuminosityFunction

.. toctree::
   :maxdepth: 1
   :hidden:

   kcorrect_examples
   poggianti_examples
   luminosity_function_examples
   conditional_luminosity_function_examples
   catalog_completeness_examples
