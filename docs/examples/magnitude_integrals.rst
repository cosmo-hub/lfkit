.. |lfkitlogo| image:: /_static/logos/lfkit_logo-icon.png
   :alt: LFKit logo
   :width: 50px

|lfkitlogo| Luminosity-function magnitude integrals
===================================================

This page shows how to integrate a bound
:class:`lfkit.LuminosityFunction` over absolute magnitude.

The examples use the ``lf.integrals`` namespace. These methods insert the
luminosity function callable internally, so users only provide the redshift
values, magnitude limits, and any optional weighting functions.

Magnitude integrals are useful when a luminosity function is used to predict
number densities, luminosity-weighted summaries, or selected fractions over a
finite magnitude range.

The number-density units follow the normalization supplied to the luminosity
function. For example, if ``phi_star`` is supplied in
:math:`{\rm Mpc}^{-3}\,{\rm mag}^{-1}`, then magnitude-integrated number
densities have units of :math:`{\rm Mpc}^{-3}`.


Integrated number density
-------------------------

The integrated number density is the luminosity function integrated over a
finite absolute-magnitude range.

This example compares a bright sample to a broader sample that also includes
fainter galaxies. The broader magnitude range gives a larger number density
because more galaxies are included in the integral.

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

   redshift = np.linspace(0.05, 1.5, 180)

   n_bright = lf.integrals.number_density(
       redshift,
       m_bright=-24.0,
       m_faint=-20.0,
       n_m=800,
   )

   n_total = lf.integrals.number_density(
       redshift,
       m_bright=-24.0,
       m_faint=-16.0,
       n_m=800,
   )

   colors = cmr.take_cmap_colors("cmr.guppy", 2, cmap_range=(0.2, 0.9))

   fig, ax = plt.subplots(figsize=(7.0, 5.0))
   ax.plot(
       redshift,
       n_bright,
       lw=3,
       color=colors[0],
       label=r"$-24 \leq M \leq -20$",
   )
   ax.plot(
       redshift,
       n_total,
       lw=3,
       color=colors[1],
       label=r"$-24 \leq M \leq -16$",
   )

   ax.set_yscale("log")
   ax.set_xlabel("Redshift $z$", fontsize=LABEL_SIZE)
   ax.set_ylabel(r"Number density [$\mathrm{Mpc}^{-3}$]", fontsize=LABEL_SIZE)
   ax.set_title("Integrated LF number density", fontsize=TITLE_SIZE)
   ax.tick_params(axis="both", labelsize=TICK_SIZE)
   ax.legend(frameon=True, fontsize=LEGEND_SIZE, loc="best")
   plt.tight_layout()


Cumulative number density
-------------------------

The number density can also be viewed as a cumulative function of the faint
absolute-magnitude limit.

This diagnostic is useful for checking how much faint galaxies contribute to
the total abundance. As the faint limit moves to less negative magnitudes, more
of the luminosity function is included.

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

   magnitude_limits = np.linspace(-23.0, -15.0, 120)

   number_density = lf.integrals.number_density(
       0.0,
       m_bright=-25.0,
       m_faint=magnitude_limits,
       n_m=800,
   )

   fig, ax = plt.subplots(figsize=(7.0, 5.0))
   ax.plot(
       magnitude_limits,
       number_density,
       lw=3,
       color=cmr.take_cmap_colors("cmr.guppy", 1, cmap_range=(0.72, 0.9))[0],
   )

   ax.set_yscale("log")
   ax.invert_xaxis()
   ax.set_xlabel(
       r"Faint absolute-magnitude limit $M_{\rm faint}$",
       fontsize=LABEL_SIZE,
   )
   ax.set_ylabel(
       r"$n(M < M_{\rm faint})$ [$\mathrm{Mpc}^{-3}$]",
       fontsize=LABEL_SIZE,
   )
   ax.set_title("Cumulative LF number density", fontsize=TITLE_SIZE)
   ax.tick_params(axis="both", labelsize=TICK_SIZE)
   plt.tight_layout()


Luminosity density and mean luminosity
--------------------------------------

The same namespace can compute luminosity-weighted summaries such as luminosity
density and mean luminosity over a selected magnitude range.

The curves below are normalized by their first redshift value to emphasize the
relative redshift trend rather than the absolute normalization.

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

   redshift = np.linspace(0.05, 1.5, 180)

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

   colors = cmr.take_cmap_colors("cmr.guppy", 2, cmap_range=(0.2, 0.9))

   fig, ax = plt.subplots(figsize=(7.0, 5.0))
   ax.plot(
       redshift,
       luminosity_density / luminosity_density[0],
       lw=3,
       color=colors[0],
       label="Luminosity density",
   )
   ax.plot(
       redshift,
       mean_luminosity / mean_luminosity[0],
       lw=3,
       color=colors[1],
       label="Mean luminosity",
   )

   ax.set_xlabel("Redshift $z$", fontsize=LABEL_SIZE)
   ax.set_ylabel("Value relative to first redshift bin", fontsize=LABEL_SIZE)
   ax.set_title("Luminosity-weighted LF summaries", fontsize=TITLE_SIZE)
   ax.tick_params(axis="both", labelsize=TICK_SIZE)
   ax.legend(frameon=True, fontsize=LEGEND_SIZE, loc="best")
   plt.tight_layout()


Selection-weighted number density
---------------------------------

Selection weights can be applied directly through the integrals API. This is
useful when a sample is not selected by a hard magnitude cut alone.

This example uses a smooth selection function in absolute magnitude. The
selected number density is lower than the total number density because the
selection downweights part of the magnitude range.

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

   redshift = np.linspace(0.05, 1.5, 180)

   def soft_selection(absolute_mag, z):
       limiting_mag = -18.5 - 1.2 * z
       width = 0.35
       return 1.0 / (1.0 + np.exp((absolute_mag - limiting_mag) / width))

   n_total = lf.integrals.number_density(
       redshift,
       m_bright=-24.0,
       m_faint=-14.0,
       n_m=800,
   )

   n_selected = lf.integrals.selection_weighted_number_density(
       redshift,
       selection_fn=soft_selection,
       m_bright=-24.0,
       m_faint=-14.0,
       n_m=800,
   )

   colors = cmr.take_cmap_colors("cmr.guppy", 2, cmap_range=(0.2, 0.9))

   fig, ax = plt.subplots(figsize=(7.0, 5.0))
   ax.plot(redshift, n_total, lw=3, color=colors[0], label="Total")
   ax.plot(
       redshift,
       n_selected,
       lw=3,
       color=colors[1],
       label="Selection weighted",
   )

   ax.set_yscale("log")
   ax.set_xlabel("Redshift $z$", fontsize=LABEL_SIZE)
   ax.set_ylabel(r"Number density [$\mathrm{Mpc}^{-3}$]", fontsize=LABEL_SIZE)
   ax.set_title("Selection-weighted LF number density", fontsize=TITLE_SIZE)
   ax.tick_params(axis="both", labelsize=TICK_SIZE)
   ax.legend(frameon=True, fontsize=LEGEND_SIZE, loc="best")
   plt.tight_layout()


Selection fraction
------------------

The selected fraction is the ratio between the selection-weighted number density
and the total number density over the same reference magnitude range.

This diagnostic is useful for checking how strongly a soft selection function
changes the effective sample abundance as a function of redshift.

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

   lf = LuminosityFunction.evolving_schechter(
       phi_model="linear_p",
       phi_kwargs={"phi_0_star": 1.0e-3, "p": 0.7},
       m_star_model="linear_q",
       m_star_kwargs={"m_0_star": -20.5, "q": 0.8, "z_ref": 0.1},
       alpha_model="constant",
       alpha_kwargs={"alpha": -1.1},
   )

   redshift = np.linspace(0.05, 1.5, 180)

   def soft_selection(absolute_mag, z):
       limiting_mag = -18.5 - 1.2 * z
       width = 0.35
       return 1.0 / (1.0 + np.exp((absolute_mag - limiting_mag) / width))

   n_total = lf.integrals.number_density(
       redshift,
       m_bright=-24.0,
       m_faint=-14.0,
       n_m=800,
   )

   n_selected = lf.integrals.selection_weighted_number_density(
       redshift,
       selection_fn=soft_selection,
       m_bright=-24.0,
       m_faint=-14.0,
       n_m=800,
   )

   selected_fraction = n_selected / n_total

   fig, ax = plt.subplots(figsize=(7.0, 5.0))
   ax.plot(
       redshift,
       selected_fraction,
       lw=3,
       color=cmr.take_cmap_colors("cmr.guppy", 1, cmap_range=(0.72, 0.9))[0],
   )

   ax.set_ylim(-0.05, 1.05)
   ax.set_xlabel("Redshift $z$", fontsize=LABEL_SIZE)
   ax.set_ylabel("Selected fraction", fontsize=LABEL_SIZE)
   ax.set_title("Fraction retained by the soft selection", fontsize=TITLE_SIZE)
   ax.tick_params(axis="both", labelsize=TICK_SIZE)
   plt.tight_layout()
