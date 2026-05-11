.. |lfkitlogo| image:: /_static/logos/lfkit_logo.png
   :alt: LFKit logo black
   :width: 50px

|lfkitlogo| Luminosity Functions
================================

LFKit provides a public luminosity-function interface through
:class:`~lfkit.api.lumfunc.LuminosityFunction`.

For most users, the recommended import is:

.. code-block:: python

   from lfkit import LuminosityFunction

The ``LuminosityFunction`` object defines and evaluates luminosity-function
models in rest-frame absolute-magnitude space. It can also evaluate luminosity
functions from apparent magnitudes, convert between apparent and absolute
magnitudes, and compute number-density quantities for magnitude-limited catalog
selections.

File reading is intentionally not handled by this API. Catalog-derived
luminosity-function parameters, magnitude limits, or correction models should
be loaded elsewhere and passed in as scalars, arrays, or correction objects.


Magnitude-space luminosity functions
------------------------------------

LFKit luminosity functions are evaluated in rest-frame absolute magnitude
:math:`M`.

The standard magnitude-space Schechter luminosity function is

.. math::

   \phi(M) =
   0.4 \ln(10) \, \phi_\star \,
   x^{\alpha + 1} \exp(-x),

where

.. math::

   x = 10^{-0.4(M - M_\star)}.

Here:

- :math:`\phi_\star` is the normalization,
- :math:`M_\star` is the characteristic magnitude,
- :math:`\alpha` is the faint-end slope.

By convention, more negative magnitudes are brighter.


Standard Schechter model
------------------------

Use :meth:`~lfkit.api.lumfunc.LuminosityFunction.schechter` to construct a
Schechter luminosity function with fixed parameters.

.. code-block:: python

   from lfkit import LuminosityFunction

   lf = LuminosityFunction.schechter(
       phi_star=1.0e-3,
       m_star=-20.5,
       alpha=-1.1,
   )

   phi = lf.phi(absolute_mag, z)

The returned ``phi`` values are number densities per magnitude evaluated at the
input absolute magnitudes.


Evolving Schechter model
------------------------

Use :meth:`~lfkit.api.lumfunc.LuminosityFunction.evolving_schechter` to build a
Schechter luminosity function with redshift-dependent parameters.

The evolving model evaluates

.. math::

   \phi_\star(z), \quad M_\star(z), \quad \alpha(z),

and then evaluates the Schechter function at each redshift.

.. code-block:: python

   from lfkit import LuminosityFunction

   lf = LuminosityFunction.evolving_schechter(
       phi_model="linear_p",
       phi_kwargs={"phi_0_star": 1.0e-3, "p": 1.0},
       m_star_model="linear_q",
       m_star_kwargs={"m_0_star": -20.5, "q": 1.2, "z_ref": 0.1},
       alpha_model="constant",
       alpha_kwargs={"alpha": -1.1},
   )

   phi = lf.phi(absolute_mag, z)

You can also evaluate the evolving parameters directly:

.. code-block:: python

   phi_star, m_star, alpha = lf.parameters(z)


Double Schechter model
----------------------

Use :meth:`~lfkit.api.lumfunc.LuminosityFunction.double_schechter` to build a
double-power-law Schechter-style luminosity function.

.. code-block:: python

   from lfkit import LuminosityFunction

   lf = LuminosityFunction.double_schechter(
       phi_star=1.0e-3,
       m_star=-20.5,
       alpha=-1.0,
       beta=-1.5,
       m_transition=-18.0,
   )

   phi = lf.phi(absolute_mag, z)

This model is useful when an additional slope or transition is needed beyond
the standard Schechter form.


Built-in parameter evolution models
-----------------------------------

Redshift-dependent luminosity-function parameters are handled by parameter
evolution models.

The built-in options are:

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Parameter
     - Model name
     - Form
   * - ``phi_star``
     - ``constant``
     - :math:`\phi_\star(z) = \phi_\star`
   * - ``phi_star``
     - ``linear_p``
     - :math:`\phi_\star(z) = \phi_{0,\star} 10^{0.4 p z}`
   * - ``M_star``
     - ``constant``
     - :math:`M_\star(z) = M_\star`
   * - ``M_star``
     - ``linear_q``
     - :math:`M_\star(z) = M_{0,\star} - q(z - z_{\rm ref})`
   * - ``alpha``
     - ``constant``
     - :math:`\alpha(z) = \alpha`
   * - ``alpha``
     - ``linear``
     - :math:`\alpha(z) = \alpha_0 + \alpha_1(z - z_{\rm ref})`

To inspect the available models:

.. code-block:: python

   models = LuminosityFunction.available_parameter_models()

Custom parameter-evolution models can be registered through:

.. code-block:: python

   LuminosityFunction.register_phi_star_model(name, model)
   LuminosityFunction.register_m_star_model(name, model)
   LuminosityFunction.register_alpha_model(name, model)

Each registered model should accept redshift values as its first argument and
return NumPy-compatible parameter values.


Evaluating from apparent magnitudes
-----------------------------------

The luminosity-function object can evaluate from apparent magnitudes.

In this case, LFKit first converts apparent magnitude :math:`m` into absolute
magnitude :math:`M` using

.. math::

   M = m - \mu(z) - K(z) + E(z),

and then evaluates :math:`\phi(M, z)`.

.. code-block:: python

   from lfkit import Corrections, LuminosityFunction

   corr = Corrections.poggianti(
       band="r",
       gal_type="E",
   )

   lf = LuminosityFunction.schechter(
       phi_star=1.0e-3,
       m_star=-20.5,
       alpha=-1.1,
   )

   phi = lf.phi_from_m(
       cosmo,
       z,
       apparent_mag,
       corrections=corr,
   )

The ``corrections`` argument is optional. If omitted, LFKit evaluates the
conversion without k- or e-corrections.


Magnitude conversion helpers
----------------------------

The same object can convert between apparent and absolute magnitudes using the
LFKit magnitude convention.

.. code-block:: python

   absolute_mag = lf.absolute_magnitude(
       cosmo,
       z,
       apparent_mag,
       corrections=corr,
   )

   apparent_mag = lf.apparent_magnitude(
       cosmo,
       z,
       absolute_mag,
       corrections=corr,
   )

The absolute-magnitude limit corresponding to an apparent-magnitude catalog cut
can be computed with:

.. code-block:: python

   m_limit = lf.absolute_magnitude_limit(
       cosmo,
       z,
       m_lim=24.5,
       corrections=corr,
   )


Integrated number density
-------------------------

Use :meth:`~lfkit.api.lumfunc.LuminosityFunction.integrated_number_density` to
integrate the luminosity function over an absolute-magnitude range.

.. code-block:: python

   n_total = lf.integrated_number_density(
       z,
       m_bright=-24.0,
       m_faint=-14.0,
   )

This computes the number density inside the requested absolute-magnitude
interval.


Magnitude-limited catalog completeness
--------------------------------------

A luminosity function can be split into observed and missing components for a
magnitude-limited catalog.

The core idea is:

1. convert the apparent catalog limit :math:`m_{\rm lim}` into an
   absolute-magnitude limit :math:`M_{\rm lim}(z)`,
2. integrate the luminosity function over a finite absolute-magnitude range,
3. split the population into observed and missing pieces.

The limiting absolute magnitude is

.. math::

   M_{\rm lim}(z) = m_{\rm lim} - \mu(z) - K(z) + E(z).


Observed number density
^^^^^^^^^^^^^^^^^^^^^^^

The observed, or in-catalog, number density is

.. math::

   n_{\rm obs}(z) =
   \int_{M_{\rm bright}}^{\min[M_{\rm lim}(z), M_{\rm faint}]}
   \phi(M, z) \, dM.

Use:

.. code-block:: python

   n_obs = lf.observed_number_density(
       cosmo,
       z,
       m_lim=24.5,
       m_bright=-24.0,
       m_faint=-14.0,
       corrections=corr,
   )


Missing number density
^^^^^^^^^^^^^^^^^^^^^^

The missing, or out-of-catalog, number density is

.. math::

   n_{\rm miss}(z) =
   \int_{\max[M_{\rm lim}(z), M_{\rm bright}]}^{M_{\rm faint}}
   \phi(M, z) \, dM.

Use:

.. code-block:: python

   n_miss = lf.missing_number_density(
       cosmo,
       z,
       m_lim=24.5,
       m_bright=-24.0,
       m_faint=-14.0,
       corrections=corr,
   )


Completeness fractions
^^^^^^^^^^^^^^^^^^^^^^

The catalog completeness fraction is

.. math::

   f_{\rm obs}(z) =
   \frac{n_{\rm obs}(z)}
        {n_{\rm obs}(z) + n_{\rm miss}(z)}.

The out-of-catalog fraction is

.. math::

   f_{\rm miss}(z) = 1 - f_{\rm obs}(z).

Use:

.. code-block:: python

   f_obs = lf.catalog_completeness(
       cosmo,
       z,
       m_lim=24.5,
       m_bright=-24.0,
       m_faint=-14.0,
       corrections=corr,
   )

   f_miss = lf.out_of_catalog_fraction(
       cosmo,
       z,
       m_lim=24.5,
       m_bright=-24.0,
       m_faint=-14.0,
       corrections=corr,
   )


Avoiding double-counted evolution
---------------------------------

There are two different places where luminosity evolution can enter an
analysis:

1. the apparent-to-absolute magnitude conversion through :math:`E(z)`,
2. the luminosity-function model through redshift evolution of :math:`M_\star(z)`.

For this reason, be careful when using an evolving Schechter model with
non-zero ``linear_q`` evolution together with an explicit evolution correction.

This is not always wrong, but the two definitions should be intentionally
separated.


Typical workflow
----------------

A typical luminosity-function workflow is:

1. define a cosmology,
2. define a luminosity-function model,
3. evaluate :math:`\phi(M, z)` directly or :math:`\phi(m, z)` from apparent
   magnitudes,
4. compute integrated, observed, or missing number densities if using a
   magnitude-limited catalog.

For example:

.. code-block:: python

   from lfkit import Corrections, LuminosityFunction

   corr = Corrections.poggianti(
       band="r",
       gal_type="E",
   )

   lf = LuminosityFunction.evolving_schechter(
       phi_model="linear_p",
       phi_kwargs={"phi_0_star": 1.0e-3, "p": 1.0},
       m_star_model="linear_q",
       m_star_kwargs={"m_0_star": -20.5, "q": 1.2, "z_ref": 0.1},
       alpha_model="constant",
       alpha_kwargs={"alpha": -1.1},
   )

   phi = lf.phi_from_m(
       cosmo,
       z,
       apparent_mag,
       corrections=corr,
   )

   n_obs = lf.observed_number_density(
       cosmo,
       z,
       m_lim=24.5,
       m_bright=-24.0,
       m_faint=-14.0,
       corrections=corr,
   )


Lower-level functions
---------------------

The high-level API is recommended for most scripts, examples, notebooks, and
downstream package interfaces.

The lower-level functions in ``lfkit.photometry`` are useful when:

- testing individual mathematical pieces,
- adding new luminosity-function models,
- adding new parameter-evolution models,
- debugging the magnitude convention,
- building new public API objects,
- integrating LFKit into specialized workflows.


What LFKit does not do here
---------------------------

LFKit does not read survey catalogs, apply angular masks, or model survey-area
incompleteness in this layer.

For magnitude-limited catalog applications, LFKit models the part of the
selection function that comes from the apparent-magnitude limit. Other effects,
such as unobserved sky area, angular masks, spectroscopic targeting, blending,
or color cuts, should be handled by the calling analysis code.
