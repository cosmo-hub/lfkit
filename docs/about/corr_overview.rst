.. |lfkitlogo| image:: /_static/logos/lfkit_logo.png
   :alt: LFKit logo black
   :width: 50px

|lfkitlogo| Photometric Corrections
===================================


Luminosity function measurements, survey forecasts, and cosmological inference
require comparing galaxy magnitudes across redshift in a consistent rest-frame
definition. Observed magnitudes are not directly comparable between redshifts
due to two effects:

1. **Bandpass shifting (k-correction)**
   A fixed observed filter samples different rest-frame wavelengths as redshift changes.

2. **Luminosity evolution (e-correction)**
   The intrinsic luminosity of galaxies evolves with cosmic time.

For a given redshift :math:`z`, photometric band, and galaxy SED (or an SED
representation), one often needs:

- :math:`k(z)` in that band,
- :math:`e(z)` describing luminosity evolution (optional),
- and their combination.

``lfkit`` provides a unified interface for constructing and evaluating these
corrections through a single small object: :class:`~lfkit.api.corrections.Corrections`.


Definitions and sign convention
-------------------------------

k-correction
^^^^^^^^^^^^

The **k-correction** :math:`k(z)` accounts for bandpass shifting.
Observing a rest-frame SED :math:`f_\lambda(\lambda)` at redshift :math:`z`
through a filter response :math:`R(\lambda)` probes

.. math::

   f_\lambda\!\left(\frac{\lambda}{1+z}\right).

Because galaxy SEDs are not flat, the mapping between observed-frame and
rest-frame flux depends on:

- redshift,
- the photometric response curve,
- the galaxy SED (or its template representation).

The k-correction is therefore a redshifting / bandpass effect driven by SED
shape and filter throughput.

e-correction
^^^^^^^^^^^^

The **e-correction** :math:`e(z)` models intrinsic luminosity evolution due to
stellar population aging, star formation history, assembly, and metallicity
evolution. It quantifies how much brighter or fainter a galaxy of a given type
would be at redshift :math:`z` compared to the present day (or a pivot redshift
:math:`z_{\rm piv}`).

Unlike the k-correction, this is a physical evolution model and may depend on:

- redshift,
- galaxy type,
- cosmology (via lookback time :math:`t_{\rm lb}(z)`).

Combined magnitude relation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Throughout ``lfkit`` we adopt the Poggianti-style convention

.. math::

   M = m - DM(z) - k(z) + e(z),

where

- :math:`m` is the observed apparent magnitude,
- :math:`DM(z)` is the distance modulus,
- :math:`M` is the rest-frame absolute magnitude.

If galaxies were brighter in the past, then :math:`e(z)` is typically
**negative** at :math:`z > z_{\rm piv}` for pivoted evolution models.


Color anchors and flux normalization
------------------------------------

The kcorrect backend in ``lfkit`` constructs an SED mixture from a **single
rest-frame two-band color constraint**

.. math::

   (m_a - m_b) = \mathrm{color\_value}.

Since magnitudes are logarithmic fluxes, this is equivalent to the flux-ratio
constraint

.. math::

   \frac{f_a}{f_b} = 10^{-0.4(m_a - m_b)}.

This ratio constrains the **SED shape** between the two bands. However, a single
color does not determine an absolute flux scale: multiplying the entire SED by a
constant leaves the ratio unchanged.

For numerical stability, the coefficient fit uses an internal flux
normalization in one band (an “anchor” choice) to construct a concrete
photometry vector for the solver.

Importantly, for a **single color anchor** this normalization cancels out and
does not affect the resulting :math:`k(z)` curve (up to numerical precision).
In practice:

- the **color** determines the SED shape (template mixture),
- the internal normalization fixes only an arbitrary amplitude for fitting,
- the resulting :math:`k(z)` depends only on the implied SED shape.


Response names in kcorrect
--------------------------

kcorrect internally uses *response curves* (filter throughput files)
identified by **response names** (file stems). Examples include
``sdss_r0``, ``sdss_g0``, ``bessell_V``, etc.

In the :class:`lfkit.Corrections` API, these response names must be used
directly when constructing kcorrect-based corrections.

For example:

.. code-block:: python

   corr = Corrections.kcorrect(
       response_out="sdss_r0",
       color=("sdss_g0", "sdss_r0"),
       color_value=0.8,
   )

Here:

- ``response_out`` is the response curve for which :math:`K(z)` is evaluated.
- ``color`` defines the two-band rest-frame color constraint used to
  determine the SED mixture.

LFKit includes helper utilities for mapping survey-style
``(filterset, band)`` pairs to kcorrect response names, but the
:class:`lfkit.Corrections` interface itself expects **response names**
rather than survey/band combinations.


The :class:`~lfkit.api.corrections.Corrections` object
------------------------------------------------------

The central object is :class:`~lfkit.api.corrections.Corrections`.

A ``Corrections`` instance wraps two callables:

- ``k_func(z) -> k(z)`` in magnitudes
- ``e_func(z) -> e(z)`` in magnitudes (or ``None`` for no evolution)

and exposes a stable evaluation interface:

- :meth:`~lfkit.api.corrections.Corrections.k`
- :meth:`~lfkit.api.corrections.Corrections.e`
- :meth:`~lfkit.api.corrections.Corrections.ke`

All methods accept scalar or array-like redshifts and return ``numpy.ndarray``
of dtype ``float``.


Construction paths provided by LFKit
------------------------------------

Poggianti (1997) tabulations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:meth:`~lfkit.api.corrections.Corrections.poggianti`

Build k- and e-corrections from the Poggianti (1997) literature tables:

- loads tabulated :math:`k(z)` and :math:`e(z)` for a chosen ``band`` and
  ``gal_type``,
- builds smooth interpolators,
- supports turning evolution off via ``e_model="none"``.

This path is fast, reproducible, and commonly used in forecasting.

kcorrect (one color anchor)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

:meth:`~lfkit.api.corrections.Corrections.kcorrect`

Compute k-corrections using the ``kcorrect`` backend by fitting a template
mixture from a single rest-frame two-band color constraint:

- fit non-negative template coefficients consistent with the requested color,
- evaluate :math:`k(z)` on a redshift grid for a chosen output response,
- optionally enforce :math:`k(0)=0` via ``anchor_z0=True``,
- wrap the result in an interpolator for fast evaluation.

The kcorrect backend provides :math:`k(z)` only (bandpass shifting). Physical
luminosity evolution :math:`e(z)` must be provided by an evolution backend
(e.g. Poggianti) if needed.


Metadata and provenance
-----------------------

Each constructor populates ``Corrections.meta`` with diagnostic information
such as:

- ``k_backend`` / ``e_backend``,
- band / response information,
- color constraint parameters (for kcorrect),
- interpolation settings,
- anchoring status,
- finite redshift validity range.

This preserves scientific traceability without changing the evaluation API.