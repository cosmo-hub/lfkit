.. |lfkitlogo| image:: /_static/logos/lfkit_logo.png
   :alt: LFKit logo black
   :width: 50px

|lfkitlogo| Photometry Overview
===============================

A luminosity function describes the abundance of galaxies as a function of
intrinsic brightness. It is a number-density distribution: it tells us how many
galaxies exist per unit volume and per luminosity or magnitude interval. If it
is normalized by the total number density, it can also be interpreted as a
probability density over luminosity or magnitude.

In luminosity units, the luminosity function is often written as
:math:`\Phi(L, z)`, where :math:`L` is luminosity and :math:`z` is redshift.
The quantity

.. math::

   \Phi(L, z)\,\mathrm{d}L

is the comoving number density of galaxies with luminosities between
:math:`L` and :math:`L+\mathrm{d}L` at redshift :math:`z`.

Equivalently, in absolute magnitude space, the luminosity function is written as
:math:`\Phi(M, z)`, where :math:`M` is absolute magnitude. In this case,

.. math::

   \Phi(M, z)\,\mathrm{d}M

is the comoving number density of galaxies with absolute magnitudes between
:math:`M` and :math:`M+\mathrm{d}M`.

Luminosity and absolute magnitude are intrinsic quantities. Surveys instead
measure fluxes, which are usually reported as apparent magnitudes :math:`m`.
Therefore, using a luminosity function with observed data usually requires
converting between apparent magnitude and absolute magnitude using a luminosity
distance, and sometimes additional photometric corrections.

For worked examples of the LFKit public API, see the dedicated example pages:

- :doc:`../examples/luminosity_function_models`
- :doc:`../examples/magnitudes_and_luminosities`
- :doc:`../examples/magnitude_integrals`
- :doc:`../examples/redshift_density`
- :doc:`../examples/catalog_completeness`


Luminosity and magnitude
------------------------

Luminosity :math:`L` is the total energy emitted by a galaxy per unit time.
Flux :math:`F` is the energy received by an observer per unit area and per unit
time. The two are related by the luminosity distance :math:`d_L`:

.. math::

   F = \frac{L}{4\pi d_L^2}.

This is why luminosity is not directly observed. It is inferred from the
measured flux once a distance has been specified.

Astronomy often uses magnitudes instead of luminosities. Apparent magnitude
:math:`m` describes how bright an object appears to the observer. Absolute
magnitude :math:`M` describes the intrinsic brightness of the object, defined as
the apparent magnitude it would have at a standard reference distance.

By convention, more negative magnitudes are brighter. A galaxy with
:math:`M=-22` is brighter than a galaxy with :math:`M=-18`.

Luminosity and absolute magnitude are related logarithmically. Relative to a
reference luminosity :math:`L_0`, the absolute magnitude can be written as

.. math::

   M = M_0 - 2.5 \log_{10}\left(\frac{L}{L_0}\right),

or equivalently,

.. math::

   \frac{L}{L_0}
   =
   10^{-0.4(M - M_0)}.

Here :math:`M_0` is the magnitude corresponding to the reference luminosity
:math:`L_0`. This relation is why brighter objects have smaller, more negative
magnitudes.

The conversion between apparent and absolute magnitude can be written as

.. math::

   M = m - \mu(z) - K(z) + E(z),

where:

- :math:`M` is absolute magnitude,
- :math:`m` is apparent magnitude,
- :math:`\mu(z)` is the distance modulus,
- :math:`K(z)` is the k-correction,
- :math:`E(z)` is the evolution correction,
- :math:`z` is redshift.

The distance modulus encodes the effect of distance. The k-correction accounts
for observing a redshifted galaxy spectrum through a fixed bandpass. The
evolution correction accounts for intrinsic luminosity evolution of the galaxy
population, depending on the convention adopted in the analysis.


Magnitude-space luminosity functions
------------------------------------

LFKit works primarily in rest-frame absolute magnitude space. This is a natural
choice for galaxy luminosity functions because absolute magnitude is an
intrinsic brightness variable.

A magnitude-space luminosity function :math:`\Phi(M, z)` gives the number
density of galaxies per unit magnitude. If :math:`\Phi(M, z)` has units of
:math:`{\rm Mpc}^{-3}\,{\rm mag}^{-1}`, then integrating it over a finite
absolute magnitude interval gives a number density in
:math:`{\rm Mpc}^{-3}`:

.. math::

   n(z) =
   \int_{M_{\rm bright}}^{M_{\rm faint}}
   \Phi(M, z)\,\mathrm{d}M.

Here:

- :math:`n(z)` is the integrated number density at redshift :math:`z`,
- :math:`M_{\rm bright}` is the bright absolute magnitude limit,
- :math:`M_{\rm faint}` is the faint absolute magnitude limit,
- :math:`\Phi(M, z)` is the luminosity function per unit magnitude.

Because brighter galaxies have more negative magnitudes,
:math:`M_{\rm bright}` is usually more negative than :math:`M_{\rm faint}`.


The Schechter luminosity function
---------------------------------

A common model for galaxy luminosity functions is the Schechter function. In
luminosity space, it is usually written as

.. math::

   \Phi(L)\,\mathrm{d}L =
   \phi_\star
   \left(\frac{L}{L_\star}\right)^\alpha
   \exp\left(-\frac{L}{L_\star}\right)
   \frac{\mathrm{d}L}{L_\star}.

Here:

- :math:`L` is galaxy luminosity,
- :math:`L_\star` is the characteristic luminosity,
- :math:`\phi_\star` is the normalization,
- :math:`\alpha` is the faint-end slope.

The Schechter form combines two behaviours. At low luminosities, the model is
approximately a power law controlled by :math:`\alpha`. At high luminosities,
the exponential term suppresses the abundance of very bright galaxies.


Schechter function in magnitude space
-------------------------------------

The magnitude-space form follows from the luminosity-space form by using the
luminosity ratio

.. math::

   \frac{L}{L_\star}
   =
   10^{-0.4(M - M_\star)}.

The change of variables from luminosity to magnitude also introduces the factor
:math:`0.4\ln(10)`.

In absolute magnitude space, the Schechter luminosity function can be written as

.. math::

   \Phi(M) =
   0.4 \ln(10) \, \phi_\star \,
   x^{\alpha + 1} \exp(-x),

with

.. math::

   x = 10^{-0.4(M - M_\star)}.

Here:

- :math:`M` is absolute magnitude,
- :math:`M_\star` is the characteristic absolute magnitude,
- :math:`\phi_\star` is the normalization,
- :math:`\alpha` is the faint-end slope,
- :math:`x` is the luminosity ratio :math:`L/L_\star` written in magnitude form.

The parameter :math:`M_\star` marks the transition between the power-law part of
the luminosity function and the exponential bright-end cutoff. The parameter
:math:`\alpha` controls how rapidly the abundance rises toward fainter
magnitudes. More negative values of :math:`\alpha` produce a steeper faint end.

The normalization :math:`\phi_\star` sets the overall abundance scale. If
:math:`\phi_\star` is supplied in :math:`{\rm Mpc}^{-3}`, then
:math:`\Phi(M)` is usually interpreted as a number density per magnitude,
:math:`{\rm Mpc}^{-3}\,{\rm mag}^{-1}`.


Redshift evolution
------------------

Galaxy populations evolve with redshift, so luminosity function parameters are
often allowed to depend on :math:`z`. A redshift-dependent Schechter model can
be written schematically as

.. math::

   \Phi(M, z) =
   \Phi\left(M \mid \phi_\star(z), M_\star(z), \alpha(z)\right).

Here:

- :math:`\phi_\star(z)` describes evolution in the overall normalization,
- :math:`M_\star(z)` describes evolution in the characteristic magnitude,
- :math:`\alpha(z)` describes evolution in the faint-end slope.

Changing :math:`\phi_\star(z)` changes the total abundance scale. Changing
:math:`M_\star(z)` shifts the characteristic magnitude where the luminosity
function turns over. Changing :math:`\alpha(z)` mainly changes the relative
abundance of faint galaxies.

Different analyses use different parameterizations for this evolution. For
example, one may use a constant parameter, a linear trend with redshift, or a
survey-specific empirical model. The important point is that the luminosity
function model and the photometric evolution correction should be defined
consistently.


Apparent magnitude limits
-------------------------

Observed catalogs are often selected by an apparent magnitude limit
:math:`m_{\rm lim}`. A luminosity function, however, is usually evaluated in
absolute magnitude space. The corresponding absolute magnitude limit is

.. math::

   M_{\rm lim}(z)
   =
   m_{\rm lim} - \mu(z) - K(z) + E(z).

Here:

- :math:`M_{\rm lim}(z)` is the redshift-dependent absolute magnitude limit,
- :math:`m_{\rm lim}` is the apparent magnitude limit of the catalog,
- :math:`\mu(z)` is the distance modulus,
- :math:`K(z)` is the k-correction,
- :math:`E(z)` is the evolution correction.

The dependence on :math:`z` is important. The same apparent magnitude limit
corresponds to different intrinsic luminosities at different redshifts. At
higher redshift, a fixed apparent magnitude cut usually selects only brighter
galaxies.

This is the basic reason magnitude-limited samples become increasingly
incomplete for faint galaxies at larger distances.


Number-density integrals
------------------------

Integrating a luminosity function over magnitude gives the number density of
galaxies inside a chosen magnitude range:

.. math::

   n(z) =
   \int_{M_{\rm bright}}^{M_{\rm faint}}
   \Phi(M, z)\,\mathrm{d}M.

This quantity is useful when the luminosity function is used to predict the
abundance of a galaxy sample. Changing the integration limits changes the
population being counted. A brighter cut selects only luminous galaxies, while a
fainter cut includes more of the faint galaxy population.

For a magnitude-limited catalog, the observed number density can be written as

.. math::

   n_{\rm obs}(z) =
   \int_{M_{\rm bright}}^{\min[M_{\rm lim}(z), M_{\rm faint}]}
   \Phi(M, z)\,\mathrm{d}M.

The missing, or out-of-catalog, number density can be written as

.. math::

   n_{\rm miss}(z) =
   \int_{\max[M_{\rm lim}(z), M_{\rm bright}]}^{M_{\rm faint}}
   \Phi(M, z)\,\mathrm{d}M.

Here:

- :math:`n_{\rm obs}(z)` is the number density above the catalog selection,
- :math:`n_{\rm miss}(z)` is the number density below the catalog selection,
- :math:`M_{\rm lim}(z)` is the absolute magnitude limit implied by the apparent
  magnitude cut.

These definitions split the same reference luminosity function into the part
that is observable and the part that is missed by the magnitude limit.


Completeness fractions
----------------------

The catalog completeness fraction is the fraction of the reference population
that is retained by the magnitude limit:

.. math::

   f_{\rm obs}(z) =
   \frac{n_{\rm obs}(z)}
        {n_{\rm obs}(z) + n_{\rm miss}(z)}.

The missing fraction is

.. math::

   f_{\rm miss}(z) = 1 - f_{\rm obs}(z).

Here:

- :math:`f_{\rm obs}(z)` is the observed or in-catalog fraction,
- :math:`f_{\rm miss}(z)` is the missing or out-of-catalog fraction.

These fractions only describe the selection caused by the apparent magnitude
limit. Other survey effects, such as masks, blending, targeting, color cuts, or
spectroscopic failures, are separate selection effects and should be modeled
elsewhere.


LF-weighted redshift trends
---------------------------

A luminosity function can also be used to build a redshift-dependent selection
trend. For a magnitude-limited sample, one common ingredient is the
magnitude-integrated luminosity function as a function of redshift:

.. math::

   S(z) =
   \int_{M_{\rm bright}}^{M_{\rm lim}(z)}
   \Phi(M, z)\,\mathrm{d}M.

Here:

- :math:`S(z)` is the luminosity function selection factor,
- :math:`M_{\rm bright}` is the bright integration limit,
- :math:`M_{\rm lim}(z)` is the redshift-dependent faint limit implied by the
  apparent magnitude cut.

This selection factor describes how much of the luminosity function is retained
at each redshift. It is not by itself a full survey redshift distribution. To
build a redshift distribution, it is often combined with a volume factor or
another redshift-dependent weight:

.. math::

   n(z) \propto S(z)\,W(z),

where :math:`W(z)` is a chosen redshift or volume weight.

The exact form of :math:`W(z)` depends on the analysis. For example, it may
represent the comoving volume element, an input parent population, or another
survey-specific weighting function.


Luminosity evolution and double counting
----------------------------------------

Luminosity evolution can enter an analysis in more than one place. It may appear
in the photometric conversion through an evolution correction :math:`E(z)`, or
it may appear directly in the luminosity function through a redshift-dependent
parameter such as :math:`M_\star(z)`.

These two choices are not automatically equivalent. Using both at the same time
can be correct if the conventions are defined carefully, but it can also double
count evolution if both terms describe the same physical effect.

A useful rule is to keep the roles separate:

- :math:`E(z)` belongs to the apparent-to-absolute magnitude conversion,
- :math:`M_\star(z)`, :math:`\phi_\star(z)`, and :math:`\alpha(z)` belong to the
  luminosity function model.

The analysis should define which part of the evolution is handled by the
photometric correction and which part is handled by the luminosity function
parameterization.


What LFKit models
-----------------

LFKit focuses on the luminosity function side of these calculations. In this
layer, the relevant ingredients are intrinsic luminosity or magnitude,
redshift-dependent luminosity function parameters, apparent-to-absolute
magnitude conversions, and number-density integrals.

LFKit does not model every survey selection effect. Angular masks, survey area,
blending, targeting, spectroscopic success rates, and other catalog-level
effects should be handled by the calling analysis code.

The theory described here is implemented in the public LFKit interface and shown
with executable examples in the example pages.