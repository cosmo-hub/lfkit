.. |lfkitlogo| image:: /_static/logos/lfkit_logo-icon.png
   :alt: LFKit logo
   :width: 50px

|lfkitlogo| Luminosity function models
======================================

This section introduces the luminosity function models exposed by
:class:`lfkit.LuminosityFunction`. These models describe the abundance of
galaxies as a function of magnitude, usually written as :math:`\Phi(M)`.

The examples focus on constructing, evaluating, visualizing, and comparing
luminosity function models. Magnitude integrals, completeness calculations,
apparent magnitude limits, redshift-density weighting, and conditional
luminosity functions are covered on separate pages.

The API is centered on :class:`lfkit.LuminosityFunction`. A luminosity function
object stores the chosen model and evaluates it through
:meth:`lfkit.LuminosityFunction.phi`.

The number-density units follow the normalization supplied to the luminosity
function. For example, if ``phi_star`` is supplied in
:math:`{\rm Mpc}^{-3}\,{\rm mag}^{-1}`, then :math:`\Phi(M)` has units of
:math:`{\rm Mpc}^{-3}\,{\rm mag}^{-1}`.

.. toctree::
   :maxdepth: 1

   model_registry
   schechter_models
   gaussian_models
   power_law_models
   composite_models