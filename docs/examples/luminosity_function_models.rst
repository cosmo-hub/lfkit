.. |lfkitlogo| image:: /_static/logos/lfkit_logo-icon.png
   :alt: LFKit logo
   :width: 50px

|lfkitlogo| Luminosity function models
======================================

This page introduces the luminosity function models exposed by
:class:`lfkit.LuminosityFunction`.

The examples focus on constructing, evaluating, visualizing, and comparing
luminosity function models. Magnitude integrals, completeness calculations,
apparent-magnitude limits, redshift-density weighting, and conditional
luminosity functions are covered on separate pages.

The API is centered on :class:`lfkit.LuminosityFunction`. A luminosity function
object stores the chosen model and evaluates it through
:meth:`lfkit.LuminosityFunction.phi`.

The number-density units follow the normalization supplied to the luminosity
function. For example, if ``phi_star`` is supplied in
:math:`{\rm Mpc}^{-3}\,{\rm mag}^{-1}`, then :math:`\Phi(M)` has units of
:math:`{\rm Mpc}^{-3}\,{\rm mag}^{-1}`.


Schechter-family models
-----------------------

The Schechter family is the main luminosity function model family currently
exposed by LFKit. It includes the standard Schechter model, double-Schechter
variants, and redshift-evolving Schechter models.

These models are useful for describing galaxy luminosity functions with a
power-law faint end and an exponential bright-end cutoff. The examples below
show how to construct, evaluate, compare, and inspect Schechter-family models.


Standard Schechter luminosity function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A Schechter luminosity function can be created with
:meth:`lfkit.LuminosityFunction.schechter`. The returned object evaluates
:math:`\Phi(M)` through :meth:`lfkit.LuminosityFunction.phi`.

.. plot::
   :include-source: True
   :width: 520

   import numpy as np
   import matplotlib.pyplot as plt
   import cmasher as cmr

   from lfkit import LuminosityFunction

   LABEL_SIZE = 15
   TICK_SIZE = 13
   TITLE_SIZE = 17

   lf = LuminosityFunction.schechter(
       phi_star=1.0e-3,
       m_star=-20.5,
       alpha=-1.1,
   )

   absolute_mag = np.linspace(-24.0, -14.0, 500)
   phi = lf.phi(absolute_mag)

   fig, ax = plt.subplots(figsize=(7.0, 5.0))
   ax.plot(
       absolute_mag,
       phi,
       lw=3,
       color=cmr.take_cmap_colors("cmr.guppy", 1, cmap_range=(0.72, 0.9))[0],
   )

   ax.set_yscale("log")
   ax.invert_xaxis()
   ax.set_xlabel("Absolute magnitude $M$", fontsize=LABEL_SIZE)
   ax.set_ylabel(
       r"$\Phi(M)$ [$\mathrm{Mpc}^{-3}\,\mathrm{mag}^{-1}$]",
       fontsize=LABEL_SIZE,
   )
   ax.set_title("Schechter luminosity function", fontsize=TITLE_SIZE)
   ax.tick_params(axis="both", labelsize=TICK_SIZE)
   plt.tight_layout()


Standard Schechter luminosity function with apparent-magnitude axis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Schechter luminosity function is evaluated in absolute magnitude. A
secondary x-axis can show the corresponding apparent magnitude at a fixed
luminosity distance using the LFKit magnitude converters.

This keeps the model-native absolute-magnitude axis while also showing where
the same magnitude range would appear observationally.

.. plot::
   :include-source: True
   :width: 560

   import numpy as np
   import matplotlib.pyplot as plt
   import cmasher as cmr

   from lfkit import LuminosityFunction

   LABEL_SIZE = 15
   TICK_SIZE = 13
   TITLE_SIZE = 17

   lf = LuminosityFunction.schechter(
       phi_star=1.0e-3,
       m_star=-20.5,
       alpha=-1.1,
   )

   luminosity_distance_mpc = 3500.0

   def absolute_to_apparent(absolute_mag):
       return lf.magnitudes.apparent_from_luminosity_distance(
           absolute_mag,
           luminosity_distance_mpc,
       )

   def apparent_to_absolute(apparent_mag):
       return lf.magnitudes.absolute_from_luminosity_distance(
           apparent_mag,
           luminosity_distance_mpc,
       )

   absolute_mag = np.linspace(-24.0, -14.0, 500)
   phi = lf.phi(absolute_mag)

   fig, ax = plt.subplots(figsize=(7.2, 5.0))
   ax.plot(
       absolute_mag,
       phi,
       lw=3,
       color=cmr.take_cmap_colors("cmr.guppy", 1, cmap_range=(0.72, 0.9))[0],
   )

   secax = ax.secondary_xaxis(
       "top",
       functions=(absolute_to_apparent, apparent_to_absolute),
   )

   ax.set_yscale("log")
   ax.invert_xaxis()
   ax.set_xlabel("Absolute magnitude $M$", fontsize=LABEL_SIZE)
   ax.set_ylabel(
       r"$\Phi(M)$ [$\mathrm{Mpc}^{-3}\,\mathrm{mag}^{-1}$]",
       fontsize=LABEL_SIZE,
   )
   ax.set_title(
       "Schechter luminosity function with apparent-magnitude axis",
       fontsize=TITLE_SIZE,
   )
   ax.tick_params(axis="both", labelsize=TICK_SIZE)

   secax.set_xlabel("Apparent magnitude $m$", fontsize=LABEL_SIZE)
   secax.tick_params(axis="x", labelsize=TICK_SIZE)

   plt.tight_layout()



Comparing Schechter slopes
~~~~~~~~~~~~~~~~~~~~~~~~~~

Changing :math:`\alpha` modifies the faint-end behaviour of the luminosity
function.

This comparison shows how the faint-end slope changes the abundance of faint
galaxies while keeping the other Schechter parameters fixed. More negative
values of :math:`\alpha` produce a steeper rise toward faint magnitudes.

This is useful because the faint-end slope often controls how strongly low
luminosity galaxies contribute to integrated quantities, such as number density
or luminosity density. Even if the bright end is almost unchanged, the total
abundance can change noticeably when the faint end is modified.

.. plot::
   :include-source: True
   :width: 520

   import numpy as np
   import matplotlib.pyplot as plt
   import cmasher as cmr

   from lfkit import LuminosityFunction

   LABEL_SIZE = 15
   TICK_SIZE = 13
   TITLE_SIZE = 17
   LEGEND_SIZE = 15

   colors = cmr.take_cmap_colors("cmr.guppy", 3, cmap_range=(0.03, 0.26))

   absolute_mag = np.linspace(-24.0, -14.0, 500)
   alphas = [-0.8, -1.1, -1.4]

   fig, ax = plt.subplots(figsize=(7.0, 5.0))

   for alpha, color in zip(alphas, colors):
       lf = LuminosityFunction.schechter(
           phi_star=1.0e-3,
           m_star=-20.5,
           alpha=alpha,
       )
       ax.plot(
           absolute_mag,
           lf.phi(absolute_mag),
           lw=3,
           color=color,
           label=rf"$\alpha={alpha}$",
       )

   ax.set_yscale("log")
   ax.invert_xaxis()
   ax.set_xlabel("Absolute magnitude $M$", fontsize=LABEL_SIZE)
   ax.set_ylabel(
       r"$\Phi(M)$ [$\mathrm{Mpc}^{-3}\,\mathrm{mag}^{-1}$]",
       fontsize=LABEL_SIZE,
   )
   ax.set_title("Effect of the faint-end slope", fontsize=TITLE_SIZE)
   ax.tick_params(axis="both", labelsize=TICK_SIZE)
   ax.legend(frameon=True, fontsize=LEGEND_SIZE, loc="best")
   plt.tight_layout()


Double Schechter luminosity function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The API also exposes a double-Schechter constructor. This is useful for models
that need extra flexibility at the faint end while retaining a Schechter-like
bright-end cutoff.

.. plot::
   :include-source: True
   :width: 520

   import numpy as np
   import matplotlib.pyplot as plt
   import cmasher as cmr

   from lfkit import LuminosityFunction

   LABEL_SIZE = 15
   TICK_SIZE = 13
   TITLE_SIZE = 17
   LEGEND_SIZE = 15

   single = LuminosityFunction.schechter(
       phi_star=1.0e-3,
       m_star=-20.5,
       alpha=-1.1,
   )

   double = LuminosityFunction.double_schechter(
       phi_star=1.0e-3,
       m_star=-20.5,
       alpha=-1.1,
       beta=-1.5,
       m_transition=-19.5,
   )

   absolute_mag = np.linspace(-24.0, -14.0, 500)
   colors = cmr.take_cmap_colors("cmr.guppy", 2, cmap_range=(0.15, 0.85))

   fig, ax = plt.subplots(figsize=(7.0, 5.0))
   ax.plot(
       absolute_mag,
       single.phi(absolute_mag),
       lw=3,
       color=colors[0],
       label="Schechter",
   )
   ax.plot(
       absolute_mag,
       double.phi(absolute_mag),
       lw=3,
       color=colors[1],
       label="Double Schechter",
   )

   ax.set_yscale("log")
   ax.invert_xaxis()
   ax.set_xlabel("Absolute magnitude $M$", fontsize=LABEL_SIZE)
   ax.set_ylabel(
       r"$\Phi(M)$ [$\mathrm{Mpc}^{-3}\,\mathrm{mag}^{-1}$]",
       fontsize=LABEL_SIZE,
   )
   ax.set_title("Schechter and double-Schechter models", fontsize=TITLE_SIZE)
   ax.tick_params(axis="both", labelsize=TICK_SIZE)
   ax.legend(frameon=True, fontsize=LEGEND_SIZE, loc="best")
   plt.tight_layout()


Evolving Schechter luminosity function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An evolving Schechter luminosity function lets the Schechter parameters depend
on redshift through LFKit's registered parameter models. This is useful when the
same LF object should evaluate :math:`\Phi(M, z)` at many redshifts.

.. plot::
   :include-source: True
   :width: 520

   import numpy as np
   import matplotlib.pyplot as plt
   import cmasher as cmr

   from lfkit import LuminosityFunction

   LABEL_SIZE = 15
   TICK_SIZE = 13
   TITLE_SIZE = 17
   LEGEND_SIZE = 15

   lf = LuminosityFunction.evolving_schechter(
       phi_model="linear_p",
       phi_kwargs={"phi_0_star": 1.0e-3, "p": 0.7},
       m_star_model="linear_q",
       m_star_kwargs={"m_0_star": -20.5, "q": 0.8, "z_ref": 0.1},
       alpha_model="constant",
       alpha_kwargs={"alpha": -1.1},
   )

   absolute_mag = np.linspace(-24.0, -14.0, 500)
   redshifts = [0.1, 0.6, 1.1]
   colors = cmr.take_cmap_colors("cmr.guppy", 3, cmap_range=(0.03, 0.26))

   fig, ax = plt.subplots(figsize=(7.0, 5.0))

   for z_value, color in zip(redshifts, colors):
       phi = lf.phi(absolute_mag, z_value)
       ax.plot(
           absolute_mag,
           phi,
           lw=3,
           color=color,
           label=rf"$z={z_value}$",
       )

   ax.set_yscale("log")
   ax.invert_xaxis()
   ax.set_xlabel("Absolute magnitude $M$", fontsize=LABEL_SIZE)
   ax.set_ylabel(
       r"$\Phi(M, z)$ [$\mathrm{Mpc}^{-3}\,\mathrm{mag}^{-1}$]",
       fontsize=LABEL_SIZE,
   )
   ax.set_title("Evolving Schechter luminosity function", fontsize=TITLE_SIZE)
   ax.tick_params(axis="both", labelsize=TICK_SIZE)
   ax.legend(frameon=True, fontsize=LEGEND_SIZE, loc="best")
   plt.tight_layout()


Evolving Schechter luminosity function with apparent-magnitude axis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The evolving Schechter model is evaluated as :math:`\Phi(M, z)`. A secondary
x-axis can show the apparent magnitude corresponding to the absolute-magnitude
range at a chosen reference luminosity distance.

Here, the curves are evaluated at several redshifts, while the upper apparent
magnitude axis is defined for the reference redshift :math:`z=0.6`. This keeps
the bottom axis model-native and avoids mixing several different
distance-redshift mappings into one top axis.

.. plot::
   :include-source: True
   :width: 560

   import numpy as np
   import matplotlib.pyplot as plt
   import cmasher as cmr

   from lfkit import LuminosityFunction

   LABEL_SIZE = 15
   TICK_SIZE = 13
   TITLE_SIZE = 17
   LEGEND_SIZE = 15

   lf = LuminosityFunction.evolving_schechter(
       phi_model="linear_p",
       phi_kwargs={"phi_0_star": 1.0e-3, "p": 0.7},
       m_star_model="linear_q",
       m_star_kwargs={"m_0_star": -20.5, "q": 0.8, "z_ref": 0.1},
       alpha_model="constant",
       alpha_kwargs={"alpha": -1.1},
   )

   absolute_mag = np.linspace(-24.0, -14.0, 500)
   redshifts = [0.1, 0.6, 1.1]

   reference_redshift = 0.6
   luminosity_distance_mpc = {
       0.1: 460.0,
       0.6: 3500.0,
       1.1: 7600.0,
   }
   reference_luminosity_distance_mpc = luminosity_distance_mpc[reference_redshift]

   def absolute_to_apparent(absolute_mag):
       return lf.magnitudes.apparent_from_luminosity_distance(
           absolute_mag,
           reference_luminosity_distance_mpc,
       )

   def apparent_to_absolute(apparent_mag):
       return lf.magnitudes.absolute_from_luminosity_distance(
           apparent_mag,
           reference_luminosity_distance_mpc,
       )

   colors = cmr.take_cmap_colors("cmr.guppy", 3, cmap_range=(0.03, 0.26))

   fig, ax = plt.subplots(figsize=(7.2, 5.0))

   for z_value, color in zip(redshifts, colors):
       phi = lf.phi(absolute_mag, z_value)
       ax.plot(
           absolute_mag,
           phi,
           lw=3,
           color=color,
           label=rf"$z={z_value}$",
       )

   secax = ax.secondary_xaxis(
       "top",
       functions=(absolute_to_apparent, apparent_to_absolute),
   )

   ax.set_yscale("log")
   ax.invert_xaxis()
   ax.set_xlabel("Absolute magnitude $M$", fontsize=LABEL_SIZE)
   ax.set_ylabel(
       r"$\Phi(M, z)$ [$\mathrm{Mpc}^{-3}\,\mathrm{mag}^{-1}$]",
       fontsize=LABEL_SIZE,
   )
   ax.set_title(
       "Evolving Schechter luminosity function with apparent-magnitude axis",
       fontsize=TITLE_SIZE,
   )
   ax.tick_params(axis="both", labelsize=TICK_SIZE)
   ax.legend(frameon=True, fontsize=LEGEND_SIZE, loc="best")

   secax.set_xlabel(
       rf"Apparent magnitude $m$ at $z={reference_redshift}$",
       fontsize=LABEL_SIZE,
   )
   secax.tick_params(axis="x", labelsize=TICK_SIZE)

   plt.tight_layout()


Inspecting evolving parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For evolving models, :meth:`lfkit.LuminosityFunction.parameters` evaluates the
registered parameter models at the requested redshift. This is useful for
checking the physical behaviour before using the LF in number-density or
selection calculations.

.. plot::
   :include-source: True
   :width: 520

   import numpy as np
   import matplotlib.pyplot as plt
   import cmasher as cmr

   from lfkit import LuminosityFunction

   LABEL_SIZE = 15
   TICK_SIZE = 13
   TITLE_SIZE = 17
   LEGEND_SIZE = 15

   lf = LuminosityFunction.evolving_schechter(
       phi_model="linear_p",
       phi_kwargs={"phi_0_star": 1.0e-3, "p": 0.7},
       m_star_model="linear_q",
       m_star_kwargs={"m_0_star": -20.5, "q": 0.8, "z_ref": 0.1},
       alpha_model="constant",
       alpha_kwargs={"alpha": -1.1},
   )

   redshift = np.linspace(0.0, 1.5, 200)
   phi_star, m_star, alpha = lf.parameters(redshift)
   colors = cmr.take_cmap_colors("cmr.guppy", 3, cmap_range=(0.1, 0.9))

   fig, ax = plt.subplots(figsize=(7.0, 5.0))
   ax.plot(
       redshift,
       phi_star / 1.0e-3,
       lw=3,
       color=colors[0],
       label=r"$\phi_*/10^{-3}$",
   )
   ax.plot(
       redshift,
       m_star,
       lw=3,
       color=colors[1],
       label=r"$M_*$",
   )
   ax.plot(
       redshift,
       alpha,
       lw=3,
       color=colors[2],
       label=r"$\alpha$",
   )

   ax.set_xlabel("Redshift $z$", fontsize=LABEL_SIZE)
   ax.set_ylabel("Parameter value", fontsize=LABEL_SIZE)
   ax.set_title("Evolving Schechter parameters", fontsize=TITLE_SIZE)
   ax.tick_params(axis="both", labelsize=TICK_SIZE)
   ax.legend(frameon=True, fontsize=LEGEND_SIZE, loc="best")
   plt.tight_layout()


Evolving Schechter surface
~~~~~~~~~~~~~~~~~~~~~~~~~~

The same evolving model can be shown over the full magnitude-redshift plane.
The filled colour scale shows :math:`\log_{10}\Phi(M, z)`, while contours mark
constant abundance levels.

.. plot::
   :include-source: True
   :width: 560

   import numpy as np
   import matplotlib.pyplot as plt

   from lfkit import LuminosityFunction

   LABEL_SIZE = 15
   TICK_SIZE = 13
   TITLE_SIZE = 17

   lf = LuminosityFunction.evolving_schechter(
       phi_model="linear_p",
       phi_kwargs={"phi_0_star": 1.0e-3, "p": 0.7},
       m_star_model="linear_q",
       m_star_kwargs={"m_0_star": -20.5, "q": 0.8, "z_ref": 0.1},
       alpha_model="constant",
       alpha_kwargs={"alpha": -1.1},
   )

   absolute_mag = np.linspace(-24.0, -16.0, 220)
   redshift = np.linspace(0.0, 1.5, 180)
   mag_grid, z_grid = np.meshgrid(absolute_mag, redshift)

   phi = lf.phi(mag_grid, z_grid)
   log_phi = np.log10(phi)

   fig, ax = plt.subplots(figsize=(7.2, 5.0))
   mesh = ax.pcolormesh(
       absolute_mag,
       redshift,
       log_phi,
       shading="auto",
       cmap="cmr.guppy",
   )

   contour_levels = [-5.0, -4.0, -3.0, -2.0]
   contours = ax.contour(
       absolute_mag,
       redshift,
       log_phi,
       levels=contour_levels,
       colors="white",
       linewidths=1.2,
   )
   ax.clabel(contours, inline=True, fontsize=TICK_SIZE, fmt=r"$10^{%.0f}$")

   ax.invert_xaxis()
   ax.set_xlabel("Absolute magnitude $M$", fontsize=LABEL_SIZE)
   ax.set_ylabel("Redshift $z$", fontsize=LABEL_SIZE)
   ax.set_title("Evolving Schechter LF surface", fontsize=TITLE_SIZE)
   ax.tick_params(axis="both", labelsize=TICK_SIZE)

   cbar = fig.colorbar(mesh, ax=ax)
   cbar.set_label(
       r"$\log_{10}\Phi(M, z)$ "
       r"[$\log_{10}(\mathrm{Mpc}^{-3}\,\mathrm{mag}^{-1})$]",
       fontsize=LABEL_SIZE,
   )
   cbar.ax.tick_params(labelsize=TICK_SIZE)

   plt.tight_layout()


Other luminosity function parametrizations
------------------------------------------

Additional luminosity function parametrizations can be added here as they are
implemented in the public API.

Examples may include Saunders or modified-Schechter models, double-power-law
forms, lognormal-inspired parametrizations, or other survey-specific luminosity
function models. This section is intentionally kept as a placeholder so the
page can grow beyond the Schechter family without mixing all models under one
flat heading structure.


Available models
----------------

The API can report the registered luminosity function models and parameter
models. This is useful for examples, validation, and interactive exploration.

.. code-block:: python

   from lfkit import LuminosityFunction

   LuminosityFunction.available_models()
   LuminosityFunction.available_from_m_models()
   LuminosityFunction.available_parameter_models()
