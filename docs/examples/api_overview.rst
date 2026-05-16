.. |lfkitlogo| image:: /_static/logos/lfkit_logo-icon.png
   :alt: LFKit logo
   :width: 50px

|lfkitlogo| API overview
========================

This page gives a high-level overview of LFKit's public API, how the main
example pages fit together, and which page to use for each type of calculation.

LFKit is organized around a small number of user-facing entry points. Most
workflows start by creating a luminosity function model and then using grouped
methods attached to that model for integrals, completeness calculations,
redshift-density trends, and magnitude or luminosity conversions.


Start here
----------

For ordinary luminosity functions, start with :class:`lfkit.LuminosityFunction`:

.. code-block:: python

   from lfkit import LuminosityFunction

   lf = LuminosityFunction.schechter(
       phi_star=1.0e-3,
       m_star=-20.5,
       alpha=-1.1,
   )

   phi = lf.phi(-20.0)

The object ``lf`` represents the chosen luminosity function model. Once the
model is created, the same object can be used for direct evaluation, magnitude
integrals, survey completeness calculations, redshift-density calculations, and
diagnostic conversions.

Photometric corrections are exposed separately through :class:`lfkit.Corrections`:

.. code-block:: python

   from lfkit import Corrections

Correction functions can be passed into calculations that need k-corrections,
evolution corrections, or other magnitude corrections.


Main API areas
--------------

The main public API areas are:

* luminosity function models,
* conditional luminosity function models,
* luminosity function integrals,
* magnitude-limited completeness,
* LF-weighted redshift-density calculations,
* magnitude and luminosity conversions,
* photometric corrections,
* model discovery utilities.


Which tool do I need?
---------------------

.. list-table::
   :header-rows: 1
   :widths: 28 42 30

   * - Task
     - Use
     - Example page
   * - Evaluate or compare luminosity function models
     - :class:`lfkit.LuminosityFunction`
     - :doc:`luminosity function models <luminosity_function_models>`
   * - Integrate a luminosity function over absolute magnitude
     - ``lf.integrals``
     - :doc:`magnitude_integrals`
   * - Convert between apparent magnitude, absolute magnitude, and luminosity distance
     - ``lf.magnitudes``
     - :doc:`magnitudes_and_luminosities`
   * - Work with luminosity ratios and Schechter luminosity variables
     - ``lf.luminosities``
     - :doc:`magnitudes_and_luminosities`
   * - Estimate observed, missing, and out-of-catalog fractions
     - ``lf.completeness``
     - :doc:`catalog completeness <catalog_completeness>`
   * - Build magnitude-limited or volume-weighted redshift trends
     - ``lf.redshift_density``
     - :doc:`redshift_density`
   * - Work with conditional luminosity functions
     - ``ConditionalLuminosityFunction``
     - :doc:`conditional luminosity function <conditional_luminosity_function>`
   * - Use k-corrections from the kcorrect backend
     - :class:`lfkit.Corrections`
     - :doc:`kcorrect_examples`
   * - Use Poggianti correction tables
     - :class:`lfkit.Corrections`
     - :doc:`poggianti_examples`
   * - Inspect available model names
     - ``available_models()`` methods
     - :doc:`model_registry`


Basic workflow
--------------

A typical LFKit workflow has three steps:

1. choose a luminosity function model,
2. evaluate it directly or pass it to one of the grouped calculation namespaces,
3. add survey limits, correction functions, or cosmology-dependent distances
   when the calculation needs them.

For example:

.. code-block:: python

   from lfkit import LuminosityFunction

   lf = LuminosityFunction.schechter(
       phi_star=1.0e-3,
       m_star=-20.5,
       alpha=-1.1,
   )

   phi = lf.phi(-20.0)

   number_density = lf.integrals.number_density(
       redshift=0.5,
       m_bright=-24.0,
       m_faint=-16.0,
   )

   catalog_fraction = lf.completeness.catalog_fraction(
       cosmo,
       redshift=0.5,
       m_lim=24.0,
       m_bright=-24.0,
       m_faint=-16.0,
       h=0.7,
   )

The first line evaluates the luminosity function at a single absolute
magnitude. The integral example computes the number density over a finite
absolute magnitude range. The completeness example adds an apparent magnitude
limit and asks what fraction of the intrinsic luminosity function is visible in
a survey.


Luminosity function models
--------------------------

Use :class:`lfkit.LuminosityFunction` for ordinary luminosity function models.

For a standard Schechter model:

.. code-block:: python

   lf = LuminosityFunction.schechter(
       phi_star=1.0e-3,
       m_star=-20.5,
       alpha=-1.1,
   )

   phi = lf.phi(absolute_mag)

For an evolving model, pass redshift when evaluating the model:

.. code-block:: python

   phi = lf.phi(absolute_mag, redshift)

The luminosity function examples page shows how to compare models, evaluate
evolving parameters, and visualize luminosity function behavior:

.. code-block:: rst

   :doc:`luminosity function models <luminosity_function_models>`


Grouped methods
---------------

A luminosity function object exposes grouped methods for common calculations:

.. list-table::
   :header-rows: 1
   :widths: 25 55 20

   * - Namespace
     - Purpose
     - Example
   * - ``lf.integrals``
     - Integrals over absolute magnitude.
     - ``lf.integrals.number_density(...)``
   * - ``lf.completeness``
     - Magnitude-limited catalog fractions and missing fractions.
     - ``lf.completeness.catalog_fraction(...)``
   * - ``lf.redshift_density``
     - LF-weighted redshift trends and magnitude-limited number densities.
     - ``lf.redshift_density.weighted(...)``
   * - ``lf.magnitudes``
     - Apparent/absolute magnitude and luminosity-distance conversions.
     - ``lf.magnitudes.absolute_from_luminosity_distance(...)``
   * - ``lf.luminosities``
     - Luminosity ratios and related helpers.
     - ``lf.luminosities.ratio_from_magnitudes(...)``

This grouping is meant to make notebook workflows easier to read. Start from
the model object, then choose the namespace that matches the calculation.


Magnitude integrals
-------------------

Use ``lf.integrals`` when you want to integrate a luminosity function over an
absolute magnitude range.

For example:

.. code-block:: python

   number_density = lf.integrals.number_density(
       redshift=0.5,
       m_bright=-24.0,
       m_faint=-16.0,
       n_m=800,
   )

Other useful quantities include luminosity density, mean luminosity, and
selection-weighted number density.

See the dedicated page for complete examples:

.. code-block:: rst

   :doc:`magnitude_integrals`


Completeness calculations
-------------------------

Use ``lf.completeness`` when a survey apparent magnitude limit is part of the
calculation.

For example:

.. code-block:: python

   catalog_fraction = lf.completeness.catalog_fraction(
       cosmo,
       redshift=0.5,
       m_lim=24.0,
       m_bright=-24.0,
       m_faint=-16.0,
       h=0.7,
   )

This answers a common survey question: given an apparent magnitude limit, what
fraction of the intrinsic luminosity function is visible at a given redshift?

The same namespace can also be used to compute observed number densities,
missing number densities, out-of-catalog fractions, and the absolute magnitude
limit implied by an apparent magnitude cut.

See the dedicated page for complete examples:

.. code-block:: rst

   :doc:`catalog completeness <catalog_completeness>`


Redshift-density calculations
-----------------------------

Use ``lf.redshift_density`` when you want to build an LF-weighted redshift trend.

For example:

.. code-block:: python

   weighted_density = lf.redshift_density.weighted(
       redshift,
       m_lim=24.0,
       m_bright=-24.0,
       luminosity_distance_mpc_fn=luminosity_distance_mpc,
       volume_weight_fn=volume_weight,
       n_m=800,
   )

This is useful when LFKit is used as part of a survey, tomography, or forecasting
workflow.

See the dedicated page for complete examples:

.. code-block:: rst

   :doc:`redshift_density`


Magnitude and luminosity conversions
------------------------------------

Use ``lf.magnitudes`` for apparent magnitude, absolute magnitude, and
luminosity-distance conversions.

For example:

.. code-block:: python

   absolute_mag = lf.magnitudes.absolute_from_luminosity_distance(
       apparent_mag,
       luminosity_distance_mpc,
   )

Use ``lf.luminosities`` for luminosity ratios and related luminosity variables:

.. code-block:: python

   luminosity_ratio = lf.luminosities.ratio_from_magnitudes(
       absolute_mag,
       m_star,
   )

These helpers are especially useful for diagnostics, selection functions, and
plots that compare magnitude and luminosity conventions.

See the dedicated page for complete examples:

.. code-block:: rst

   :doc:`magnitudes_and_luminosities`


Conditional luminosity functions
--------------------------------

Use ``ConditionalLuminosityFunction`` when the luminosity distribution is
conditioned on another variable, such as halo mass.

A typical workflow looks like:

.. code-block:: python

   from lfkit import ConditionalLuminosityFunction

   clf = ConditionalLuminosityFunction.schechter(
       phi_star=1.0,
       l_star=1.0e10,
       alpha=-1.1,
   )

   phi = clf.phi(luminosity, halo_mass)

Conditional luminosity function examples are kept on a separate page because
they answer a different modeling question from ordinary luminosity functions.

See the dedicated page for complete examples:

.. code-block:: rst

   :doc:`conditional luminosity function <conditional_luminosity_function>`


Photometric corrections
-----------------------

Use :class:`lfkit.Corrections` when you want to construct or evaluate
photometric corrections.

Correction callables can be passed into calculations that need k-corrections or
evolution corrections. For example:

.. code-block:: python

   number_density = lf.redshift_density.integrated_number_density(
       redshift,
       m_lim=24.0,
       m_bright=-24.0,
       luminosity_distance_mpc_fn=luminosity_distance_mpc,
       k_correction_fn=k_correction,
       e_correction_fn=e_correction,
       n_m=800,
   )

The magnitude convention used by LFKit is:

.. math::

   M = m - \mu - K + E,

or equivalently,

.. math::

   m = M + \mu + K - E.

See the correction-specific pages for complete examples:

.. code-block:: rst

   :doc:`kcorrect_examples`
   :doc:`poggianti_examples`


Model discovery
---------------

LFKit provides methods for inspecting available model names. These are useful in
notebooks, examples, and validation scripts.

For ordinary luminosity functions:

.. code-block:: python

   from lfkit import LuminosityFunction

   LuminosityFunction.available_models()
   LuminosityFunction.available_from_m_models()
   LuminosityFunction.available_parameter_models()

For conditional luminosity functions:

.. code-block:: python

   from lfkit import ConditionalLuminosityFunction

   ConditionalLuminosityFunction.available_models()

See the dedicated page for complete examples:

.. code-block:: rst

   :doc:`model_registry`


Example pages
-------------

The examples are split by topic so that each page stays focused.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Page
     - Contents
   * - :doc:`luminosity function models <luminosity_function_models>`
     - Basic luminosity function models, model comparison, evolving parameters, and luminosity function surfaces.
   * - :doc:`conditional luminosity function <conditional_luminosity_function>`
     - Conditional luminosity function models and halo-mass-dependent luminosity distributions.
   * - :doc:`magnitude_integrals`
     - Number density, luminosity density, mean luminosity, and selection-weighted integrals.
   * - :doc:`magnitudes_and_luminosities`
     - Magnitude conversions, luminosity-distance helpers, and luminosity-ratio helpers.
   * - :doc:`catalog completeness <catalog_completeness>`
     - Observed and missing number densities, catalog fractions, and out-of-catalog fractions.
   * - :doc:`redshift_density`
     - Magnitude-limited and volume-weighted LF redshift trends.
   * - :doc:`kcorrect_examples`
     - Examples using the kcorrect backend.
   * - :doc:`poggianti_examples`
     - Examples using Poggianti correction tables.
   * - :doc:`model_registry`
     - Registered model names and model-discovery helpers.


Next steps
----------

If you are new to LFKit, start with the luminosity function examples page and
then move to the calculation that matches your use case:

* use :doc:`magnitude_integrals` for intrinsic LF integrals,
* use :doc:`catalog completeness <catalog_completeness>` for survey magnitude limits,
* use :doc:`redshift_density` for LF-weighted redshift trends,
* use :doc:`magnitudes_and_luminosities` for conversions and diagnostics,
* use :doc:`kcorrect_examples` or :doc:`poggianti_examples` when corrections are
  needed.
