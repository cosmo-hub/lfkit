.. |lfkitlogo| image:: /_static/logos/lfkit_logo-icon.png
   :alt: LFKit logo
   :width: 50px

|lfkitlogo| API overview
========================

This page gives a high-level overview of how LFKit's public API is organized.

LFKit is built around a small number of user-facing entry points. The goal is
that most users should not need to import low-level functions from
``lfkit.photometry`` directly. Instead, they can start from the public API
objects and use grouped namespaces for related calculations.

The main API areas are:

* luminosity function models,
* conditional luminosity function models,
* luminosity function integrals,
* magnitude-limited completeness,
* LF-weighted redshift-density calculations,
* magnitude and luminosity conversions,
* photometric corrections.

The detailed examples are split across separate pages. This page is only a map
of the public API and the intended workflow.


Main entry points
-----------------

The most important public objects are imported from :mod:`lfkit`:

.. code-block:: python

   from lfkit import LuminosityFunction
   from lfkit import Corrections

If conditional luminosity functions are exposed through a separate public class,
the intended import should be:

.. code-block:: python

   from lfkit import ConditionalLuminosityFunction

The public API is organized so that users first create a model object and then
call grouped methods from that object.

For example, a standard luminosity function workflow starts with:

.. code-block:: python

   from lfkit import LuminosityFunction

   lf = LuminosityFunction.schechter(
       phi_star=1.0e-3,
       m_star=-20.5,
       alpha=-1.1,
   )

   phi = lf.phi(-20.0)

The object ``lf`` stores the chosen luminosity function model and exposes
namespaces for common calculations.


Luminosity-function API
-----------------------

The :class:`lfkit.LuminosityFunction` object is the main interface for ordinary
luminosity function models.

It provides constructors for supported LF parameterizations, for example:

.. code-block:: python

   lf = LuminosityFunction.schechter(
       phi_star=1.0e-3,
       m_star=-20.5,
       alpha=-1.1,
   )

   lf = LuminosityFunction.double_schechter(
       phi_star=1.0e-3,
       m_star=-20.5,
       alpha=-1.1,
       beta=-1.5,
       m_transition=-19.5,
   )

   lf = LuminosityFunction.evolving_schechter(
       phi_model="linear_p",
       phi_kwargs={"phi_0_star": 1.0e-3, "p": 0.7},
       m_star_model="linear_q",
       m_star_kwargs={"m_0_star": -20.5, "q": 0.8, "z_ref": 0.1},
       alpha_model="constant",
       alpha_kwargs={"alpha": -1.1},
   )

Once created, the object can evaluate the luminosity function:

.. code-block:: python

   phi = lf.phi(absolute_mag)

For evolving models, pass redshift as well:

.. code-block:: python

   phi = lf.phi(absolute_mag, redshift)

The same object can also expose the redshift-dependent parameters:

.. code-block:: python

   phi_star, m_star, alpha = lf.parameters(redshift)


Grouped namespaces
------------------

A :class:`lfkit.LuminosityFunction` object groups related functionality into
small namespaces.

The main namespaces are:

.. list-table::
   :header-rows: 1
   :widths: 25 55 20

   * - Namespace
     - Purpose
     - Example
   * - ``lf.integrals``
     - Magnitude integrals of the bound luminosity function.
     - ``lf.integrals.number_density(...)``
   * - ``lf.redshift_density``
     - Magnitude-limited and volume-weighted redshift-density calculations.
     - ``lf.redshift_density.weighted(...)``
   * - ``lf.completeness``
     - Magnitude-limited catalog completeness and missing fractions.
     - ``lf.completeness.catalog_fraction(...)``
   * - ``lf.magnitudes``
     - Apparent/absolute magnitude and distance-modulus helpers.
     - ``lf.magnitudes.absolute_from_luminosity_distance(...)``
   * - ``lf.luminosities``
     - Luminosity-ratio and Schechter helper functions.
     - ``lf.luminosities.ratio_from_magnitudes(...)``

This keeps the public API readable. Users can discover functionality from the
model object without needing to remember which low-level module contains each
calculation.


Magnitude integrals
-------------------

The ``integrals`` namespace evaluates integrals over absolute magnitude for the
luminosity function stored in ``lf``.

Typical methods include number density, luminosity density, mean luminosity,
and selection-weighted number density.

.. code-block:: python

   number_density = lf.integrals.number_density(
       redshift,
       m_bright=-24.0,
       m_faint=-16.0,
       n_m=800,
   )

   luminosity_density = lf.integrals.luminosity_density(
       redshift,
       m_bright=-24.0,
       m_faint=-16.0,
       n_m=800,
   )

   mean_luminosity = lf.integrals.mean_luminosity(
       redshift,
       m_bright=-24.0,
       m_faint=-16.0,
       n_m=800,
   )

Selection weights can be supplied through a callable:

.. code-block:: python

   def selection_fn(absolute_mag, z):
       limiting_mag = -18.5 - 1.2 * z
       width = 0.35
       return 1.0 / (1.0 + np.exp((absolute_mag - limiting_mag) / width))

   selected_density = lf.integrals.selection_weighted_number_density(
       redshift,
       selection_fn=selection_fn,
       m_bright=-24.0,
       m_faint=-14.0,
       n_m=800,
   )

The luminosity function callable is inserted internally by the API, so users
only provide the redshift grid, magnitude bounds, and optional selection
function.


Completeness calculations
-------------------------

The ``completeness`` namespace handles magnitude-limited catalog calculations.

These methods are useful when a survey has an apparent-magnitude limit
``m_lim`` and the user wants to know which part of the intrinsic luminosity
function is visible at each redshift.

Typical quantities are:

.. code-block:: python

   observed = lf.completeness.observed_number_density(
       cosmo,
       redshift,
       m_lim=24.0,
       m_bright=-24.0,
       m_faint=-16.0,
       n_m=800,
       h=0.7,
   )

   missing = lf.completeness.missing_number_density(
       cosmo,
       redshift,
       m_lim=24.0,
       m_bright=-24.0,
       m_faint=-16.0,
       n_m=800,
       h=0.7,
   )

   catalog_fraction = lf.completeness.catalog_fraction(
       cosmo,
       redshift,
       m_lim=24.0,
       m_bright=-24.0,
       m_faint=-16.0,
       n_m=800,
       h=0.7,
   )

   out_of_catalog_fraction = lf.completeness.out_of_catalog_fraction(
       cosmo,
       redshift,
       m_lim=24.0,
       m_bright=-24.0,
       m_faint=-16.0,
       n_m=800,
       h=0.7,
   )

The same namespace also exposes the absolute-magnitude limit implied by an
apparent-magnitude cut:

.. code-block:: python

   m_limit = lf.completeness.absolute_magnitude_limit(
       cosmo,
       redshift,
       m_lim=24.0,
       h=0.7,
   )

This is often the first diagnostic to check before interpreting completeness
fractions.


Redshift-density calculations
-----------------------------

The ``redshift_density`` namespace is for building LF-weighted redshift trends.

These methods are useful when LFKit is used as an ingredient in survey
forecasting or tomography construction.

A magnitude-limited number density can be computed with:

.. code-block:: python

   number_density = lf.redshift_density.integrated_number_density(
       redshift,
       m_lim=24.0,
       m_bright=-24.0,
       luminosity_distance_mpc_fn=luminosity_distance_mpc,
       n_m=800,
   )

A volume-weighted redshift trend can be computed with:

.. code-block:: python

   weighted_density = lf.redshift_density.weighted(
       redshift,
       m_lim=24.0,
       m_bright=-24.0,
       luminosity_distance_mpc_fn=luminosity_distance_mpc,
       volume_weight_fn=volume_weight,
       n_m=800,
   )

Here ``luminosity_distance_mpc_fn`` and ``volume_weight_fn`` are callables
supplied by the user or by another cosmology package.

This design keeps LFKit independent of one specific cosmology backend for these
generic redshift-density utilities.


Magnitude and luminosity helpers
--------------------------------

The ``magnitudes`` namespace provides public helpers for converting between
apparent magnitude, absolute magnitude, and luminosity distance.

For example:

.. code-block:: python

   absolute_mag = lf.magnitudes.absolute_from_luminosity_distance(
       apparent_mag,
       luminosity_distance_mpc,
   )

   apparent_mag = lf.magnitudes.apparent_from_luminosity_distance(
       absolute_mag,
       luminosity_distance_mpc,
   )

The ``luminosities`` namespace provides luminosity-ratio helpers:

.. code-block:: python

   luminosity_ratio = lf.luminosities.ratio_from_magnitudes(
       absolute_mag,
       m_star,
   )

These helpers are useful for diagnostics, selection functions, and examples
where the user wants to inspect the magnitude-luminosity mapping directly.


Conditional luminosity function API
-----------------------------------

Conditional luminosity functions describe luminosity distributions conditioned
on another variable, usually halo mass.

They should be kept conceptually separate from ordinary luminosity functions.
A conditional luminosity function object should be responsible for CLF models,
while :class:`lfkit.LuminosityFunction` remains responsible for ordinary LF
models.

A typical CLF workflow should look like:

.. code-block:: python

   clf = ConditionalLuminosityFunction.schechter(
       phi_star=1.0,
       l_star=1.0e10,
       alpha=-1.1,
   )

   phi = clf.phi(luminosity, halo_mass)

or, for magnitude-based CLF models:

.. code-block:: python

   phi = clf.phi(absolute_mag, halo_mass)

The detailed CLF examples should live on a separate page. This overview page
only records the architectural boundary:

* ordinary LF models belong to ``LuminosityFunction``,
* conditional LF models belong to ``ConditionalLuminosityFunction``,
* shared numerical helpers should stay in lower-level utility modules,
* the public API should avoid duplicating the low-level model code.


Photometric corrections
-----------------------

Photometric corrections are exposed separately through :class:`lfkit.Corrections`.

This keeps corrections independent from the luminosity function model itself.
Users can construct or evaluate corrections and then pass them into magnitude,
completeness, or redshift-density calculations when needed.

For example, correction callables can be passed into LF calculations that need
k-corrections or evolution corrections:

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

The sign convention used by the magnitude helpers is:

.. math::

   M = m - \mu - K + E,

and equivalently,

.. math::

   m = M + \mu + K - E.

This means that corrections can be supplied without hard-coding one correction
backend into the luminosity function API.


Available models
----------------

The API can report which models are registered.

For luminosity function models:

.. code-block:: python

   from lfkit import LuminosityFunction

   LuminosityFunction.available_models()
   LuminosityFunction.available_from_m_models()
   LuminosityFunction.available_parameter_models()

For conditional luminosity function models, the matching API should be:

.. code-block:: python

   from lfkit import ConditionalLuminosityFunction

   ConditionalLuminosityFunction.available_models()

These discovery methods are useful in examples, notebooks, and validation
scripts.


Recommended example-page split
------------------------------

The detailed examples should stay split by topic rather than collected into one
large page.

A useful organization is:

.. list-table::
   :header-rows: 1
   :widths: 28 52

   * - Page
     - Contents
   * - ``api_overview``
     - High-level organization of the public API.
   * - ``luminosity_function_examples``
     - Basic LF models, model comparison, evolving parameters, and LF surfaces.
   * - ``conditional_luminosity_function_examples``
     - CLF models and halo-mass-dependent luminosity distributions.
   * - ``magnitude_integrals``
     - Number density, luminosity density, mean luminosity, and selection-weighted integrals.
   * - ``magnitudes_and_luminosities``
     - Magnitude conversions, luminosity-distance helpers, and luminosity-ratio helpers.
   * - ``catalog_completeness_examples``
     - Observed/missing number densities and catalog fractions.
   * - ``redshift_density``
     - Magnitude-limited and volume-weighted LF redshift trends.
   * - ``kcorrect_examples``
     - Examples using the kcorrect backend.
   * - ``poggianti_examples``
     - Examples using Poggianti correction tables.
   * - ``model_registry``
     - Registered models and how to inspect them.

This keeps each page small enough to read and maintain.


Which API should I use?
-----------------------

Use :class:`lfkit.LuminosityFunction` when you want to evaluate or integrate an
ordinary luminosity function:

.. code-block:: python

   lf = LuminosityFunction.schechter(...)
   phi = lf.phi(...)
   number_density = lf.integrals.number_density(...)

Use the ``completeness`` namespace when a survey apparent-magnitude limit is
part of the calculation:

.. code-block:: python

   fraction = lf.completeness.catalog_fraction(...)

Use the ``redshift_density`` namespace when constructing an LF-weighted
redshift trend:

.. code-block:: python

   nz = lf.redshift_density.weighted(...)

Use the ``magnitudes`` and ``luminosities`` namespaces for conversions and
diagnostics:

.. code-block:: python

   m_abs = lf.magnitudes.absolute_from_luminosity_distance(...)
   l_ratio = lf.luminosities.ratio_from_magnitudes(...)

Use :class:`lfkit.Corrections` when constructing or evaluating photometric
corrections.

Use ``ConditionalLuminosityFunction`` when the model is conditional on halo mass
or another external variable.


Design principle
----------------

The public API should be thin and user-facing.

Low-level modules should contain the numerical implementation. Public API
classes should organize those functions into discoverable workflows without
duplicating the underlying model code.

In practice, this means:

* model constructors store the selected model and parameters,
* bound namespaces inject the stored LF callable automatically,
* correction callables are passed explicitly where needed,
* low-level functions remain available for specialist use,
* examples should use the public API wherever possible.

This keeps the docs readable while preserving the flexibility of the underlying
photometry modules.
