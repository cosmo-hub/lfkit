.. |lfkitlogo| image:: /_static/logos/lfkit_logo-icon.png
   :alt: LFKit logo
   :width: 50px

|lfkitlogo| Luminosity function examples
========================================

This page provides executable examples showing how to use
:class:`lfkit.LuminosityFunction` to construct, evaluate, and compare
luminosity function models.

Luminosity functions describe how common galaxies are as a function of their
absolute magnitude. In these examples, brighter galaxies appear on the left
because astronomical magnitudes decrease for brighter objects.

The luminosity function normalization sets the units. For example, if
:math:`\phi_\star` is supplied in comoving :math:`{\rm Mpc}^{-3}\,{\rm mag}^{-1}`,
then :math:`\phi(M, z)` has units of
:math:`{\rm Mpc}^{-3}\,{\rm mag}^{-1}`, and magnitude-integrated number
densities have units of :math:`{\rm Mpc}^{-3}`.

The examples that connect apparent and absolute magnitude use
``corrections=None``. This means that no :math:`K`-correction or evolution
correction is applied. Users can pass their own correction model through the
``corrections`` argument, for example a :class:`lfkit.Corrections` object or any
compatible correction callable used by the LFKit magnitude-conversion methods.

All examples below are executable via ``.. plot::``.


Standard Schechter luminosity function
--------------------------------------

A standard Schechter luminosity function has fixed parameters
:math:`\phi_\star`, :math:`M_\star`, and :math:`\alpha`.

This plot shows the basic shape of a Schechter luminosity function as a
function of absolute magnitude. The function decreases rapidly at the bright end
and rises toward the faint end, reflecting the usual picture that very luminous
galaxies are rare while faint galaxies are more common.

The y-axis is shown on a logarithmic scale because luminosity functions often
span several orders of magnitude. This makes both the bright-end cutoff and the
faint-end behaviour visible in the same figure.

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

   colors_blue = cmr.take_cmap_colors("cmr.guppy", 3, cmap_range=(0.72, 0.96))

   absolute_mag = np.linspace(-24.0, -14.0, 500)

   lf = LuminosityFunction.schechter(
       phi_star=1.0e-3,
       m_star=-20.5,
       alpha=-1.1,
   )

   phi = lf.phi(absolute_mag)

   fig, ax = plt.subplots(figsize=(7.0, 5.0))
   ax.plot(absolute_mag, phi, lw=3, color=colors_blue[1])
   ax.set_yscale("log")
   ax.invert_xaxis()
   ax.set_xlabel("Absolute magnitude $M$", fontsize=LABEL_SIZE)
   ax.set_ylabel(r"$\phi(M)$ [$\mathrm{Mpc}^{-3}\,\mathrm{mag}^{-1}$]", fontsize=LABEL_SIZE)
   ax.set_title("Standard Schechter luminosity function", fontsize=TITLE_SIZE)
   ax.tick_params(axis="both", labelsize=TICK_SIZE)
   plt.tight_layout()


Comparing Schechter slopes
--------------------------

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

   colors_red = cmr.take_cmap_colors("cmr.guppy", 3, cmap_range=(0.03, 0.26))

   absolute_mag = np.linspace(-24.0, -14.0, 500)
   alphas = [-0.8, -1.1, -1.4]

   fig, ax = plt.subplots(figsize=(7.0, 5.0))

   for alpha, color in zip(alphas, colors_red):
       lf = LuminosityFunction.schechter(
           phi_star=1.0e-3,
           m_star=-20.5,
           alpha=alpha,
       )
       phi = lf.phi(absolute_mag)
       ax.plot(absolute_mag, phi, lw=3, color=color, label=rf"$\alpha={alpha}$")

   ax.set_yscale("log")
   ax.invert_xaxis()
   ax.set_xlabel("Absolute magnitude $M$", fontsize=LABEL_SIZE)
   ax.set_ylabel(r"$\phi(M)$ [$\mathrm{Mpc}^{-3}\,\mathrm{mag}^{-1}$]", fontsize=LABEL_SIZE)
   ax.set_title("Effect of the faint-end slope", fontsize=TITLE_SIZE)
   ax.tick_params(axis="both", labelsize=TICK_SIZE)
   ax.legend(frameon=True, fontsize=LEGEND_SIZE, loc="best")
   plt.tight_layout()


Evolving Schechter luminosity function
--------------------------------------

An evolving Schechter model allows the luminosity function parameters to vary
with redshift.

This plot compares the luminosity function at several redshifts. Instead of
using one fixed curve for all epochs, the evolving model allows the amplitude
and characteristic magnitude to change with redshift.

This kind of plot is useful for visualizing galaxy evolution in a compact way.
Changes in normalization alter the overall abundance, while shifts in
:math:`M_\star` move the turnover of the luminosity function toward brighter or
fainter magnitudes.

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

   colors_blue = cmr.take_cmap_colors("cmr.guppy", 3, cmap_range=(0.72, 0.96))

   absolute_mag = np.linspace(-24.0, -14.0, 500)

   lf = LuminosityFunction.evolving_schechter(
       phi_model="linear_p",
       phi_kwargs={"phi_0_star": 1.0e-3, "p": 0.8},
       m_star_model="linear_q",
       m_star_kwargs={"m_0_star": -20.5, "q": 1.0, "z_ref": 0.1},
       alpha_model="constant",
       alpha_kwargs={"alpha": -1.1},
   )

   redshifts = [0.1, 0.5, 1.0]

   fig, ax = plt.subplots(figsize=(7.0, 5.0))

   for z_value, color in zip(redshifts, colors_blue):
       z = np.full_like(absolute_mag, z_value)
       phi = lf.phi(absolute_mag, z)
       ax.plot(absolute_mag, phi, lw=3, color=color, label=rf"$z={z_value}$")

   ax.set_yscale("log")
   ax.invert_xaxis()
   ax.set_xlabel("Absolute magnitude $M$", fontsize=LABEL_SIZE)
   ax.set_ylabel(r"$\phi(M, z)$ [$\mathrm{Mpc}^{-3}\,\mathrm{mag}^{-1}$]", fontsize=LABEL_SIZE)
   ax.set_title("Evolving Schechter luminosity function", fontsize=TITLE_SIZE)
   ax.tick_params(axis="both", labelsize=TICK_SIZE)
   ax.legend(frameon=True, fontsize=LEGEND_SIZE, loc="best")
   plt.tight_layout()


Inspecting evolving parameters
------------------------------

The evolving luminosity function parameters can also be evaluated directly.

This plot shows the redshift evolution of the parameters that define the
luminosity function. Instead of plotting :math:`\phi(M, z)` itself, it shows how
:math:`\phi_\star`, :math:`M_\star`, and :math:`\alpha` change with redshift.

This is useful when checking whether an evolving model behaves as expected
before using it in a larger calculation. For example, it can help verify that
the normalization evolves smoothly, that :math:`M_\star` shifts in the intended
direction, and that the faint-end slope remains in a sensible range.

.. plot::
   :include-source: True
   :width: 620

   import numpy as np
   import matplotlib.pyplot as plt
   import cmasher as cmr

   from lfkit import LuminosityFunction

   LABEL_SIZE = 15
   TICK_SIZE = 13
   TITLE_SIZE = 17
   LEGEND_SIZE = 15

   colors_blue = cmr.take_cmap_colors("cmr.guppy", 3, cmap_range=(0.72, 0.96))
   colors_red = cmr.take_cmap_colors("cmr.guppy", 3, cmap_range=(0.03, 0.26))

   z = np.linspace(0.0, 1.5, 300)

   lf = LuminosityFunction.evolving_schechter(
       phi_model="linear_p",
       phi_kwargs={"phi_0_star": 1.0e-3, "p": 0.8},
       m_star_model="linear_q",
       m_star_kwargs={"m_0_star": -20.5, "q": 1.0, "z_ref": 0.1},
       alpha_model="linear",
       alpha_kwargs={"alpha_0": -1.1, "alpha_1": -0.1, "z_ref": 0.1},
   )

   phi_star, m_star, alpha = lf.parameters(z)

   fig, axes = plt.subplots(
       3,
       1,
       figsize=(7.0, 7.2),
       sharex=True,
       constrained_layout=True,
   )

   axes[0].plot(z, phi_star, lw=3, color=colors_blue[1])
   axes[0].set_ylabel(r"$\phi_\star$", fontsize=LABEL_SIZE)
   axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
   axes[0].set_title("Evolving LF parameters", fontsize=TITLE_SIZE)

   axes[1].plot(z, m_star, lw=3, color=colors_red[1])
   axes[1].set_ylabel(r"$M_\star$", fontsize=LABEL_SIZE)

   axes[2].plot(z, alpha, lw=3, color=colors_blue[2])
   axes[2].set_xlabel("Redshift $z$", fontsize=LABEL_SIZE)
   axes[2].set_ylabel(r"$\alpha$", fontsize=LABEL_SIZE)

   for ax in axes:
       ax.tick_params(axis="both", labelsize=TICK_SIZE)

   plt.tight_layout()


Double Schechter luminosity function
------------------------------------

A double Schechter-style model can be used when the luminosity function needs
additional structure beyond the standard Schechter form.

This plot compares a standard Schechter model with a double-Schechter-style
model. The double model adds extra flexibility around the faint end, where a
single power-law slope may not describe the full galaxy population well.

This type of comparison is useful when testing whether a simple one-component
model is sufficient. If the two curves differ strongly at faint magnitudes,
integrated quantities that depend on faint galaxies may also differ.

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

   colors_blue = cmr.take_cmap_colors("cmr.guppy", 3, cmap_range=(0.72, 0.96))
   colors_red = cmr.take_cmap_colors("cmr.guppy", 3, cmap_range=(0.03, 0.26))

   absolute_mag = np.linspace(-24.0, -14.0, 500)

   standard = LuminosityFunction.schechter(
       phi_star=1.0e-3,
       m_star=-20.5,
       alpha=-1.1,
   )

   double = LuminosityFunction.double_schechter(
       phi_star=1.0e-3,
       m_star=-20.5,
       alpha=-1.0,
       beta=-1.5,
       m_transition=-18.0,
   )

   fig, ax = plt.subplots(figsize=(7.0, 5.0))

   ax.plot(
       absolute_mag,
       standard.phi(absolute_mag),
       lw=3,
       color=colors_blue[1],
       label="Standard Schechter",
   )
   ax.plot(
       absolute_mag,
       double.phi(absolute_mag),
       lw=3,
       color=colors_red[1],
       label="Double Schechter",
   )

   ax.set_yscale("log")
   ax.invert_xaxis()
   ax.set_xlabel("Absolute magnitude $M$", fontsize=LABEL_SIZE)
   ax.set_ylabel(r"$\phi(M)$ [$\mathrm{Mpc}^{-3}\,\mathrm{mag}^{-1}$]", fontsize=LABEL_SIZE)
   ax.set_title("Standard and double Schechter models", fontsize=TITLE_SIZE)
   ax.tick_params(axis="both", labelsize=TICK_SIZE)
   ax.legend(frameon=True, fontsize=LEGEND_SIZE, loc="best")
   plt.tight_layout()


Integrated number density
-------------------------

A luminosity function can be integrated over magnitude to estimate the number
density of galaxies brighter than a chosen absolute-magnitude limit.

This plot shows how the cumulative number density changes as progressively
fainter galaxies are included. At very bright limits, only rare luminous
galaxies contribute. As the magnitude limit becomes fainter, more galaxies are
included and the integrated number density increases.

This is one of the most common ways a luminosity function enters survey
calculations. Instead of using the value of :math:`\phi(M)` at one magnitude,
the model is integrated over the part of the galaxy population that the sample
selects.

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

   colors_blue = cmr.take_cmap_colors("cmr.guppy", 3, cmap_range=(0.72, 0.96))

   lf = LuminosityFunction.schechter(
       phi_star=1.0e-3,
       m_star=-20.5,
       alpha=-1.1,
   )

   magnitude_limits = np.linspace(-23.0, -15.0, 120)

   number_density = lf.integrated_number_density(
       z=0.0,
       m_bright=-25.0,
       m_faint=magnitude_limits,
       n_m=800,
   )

   fig, ax = plt.subplots(figsize=(7.0, 5.0))
   ax.plot(magnitude_limits, number_density, lw=3, color=colors_blue[1])
   ax.set_yscale("log")
   ax.invert_xaxis()
   ax.set_xlabel(r"Faint absolute-magnitude limit $M_{\rm lim}$", fontsize=LABEL_SIZE)
   ax.set_ylabel(r"$n(M < M_{\rm lim})$ [$\mathrm{Mpc}^{-3}$]", fontsize=LABEL_SIZE)
   ax.set_title("Integrated number density", fontsize=TITLE_SIZE)
   ax.tick_params(axis="both", labelsize=TICK_SIZE)
   plt.tight_layout()


Magnitude-redshift luminosity function surface
----------------------------------------------

For an evolving luminosity function, the abundance depends on both absolute
magnitude and redshift.

This plot shows the luminosity function amplitude across the
magnitude-redshift plane. The horizontal direction shows which galaxies are
bright or faint, while the vertical direction shows how the model changes with
redshift.

The filled colour scale shows :math:`\log_{10}\phi(M, z)`. The white contours
mark constant :math:`\log_{10}\phi(M, z)` levels at -5, -4, -3, and -2. These
contours make it easier to see where equal-abundance regions sit in the
magnitude-redshift plane.

This view is helpful for checking the full two-dimensional behaviour of an
evolving model. It makes it easier to see whether the bright end, faint end, and
redshift evolution combine smoothly across the range of interest.

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

   absolute_mag = np.linspace(-24.0, -18.0, 220)
   redshift = np.linspace(0.0, 1.5, 180)

   mag_grid, z_grid = np.meshgrid(absolute_mag, redshift)

   lf = LuminosityFunction.evolving_schechter(
       phi_model="linear_p",
       phi_kwargs={"phi_0_star": 1.0e-3, "p": 0.8},
       m_star_model="linear_q",
       m_star_kwargs={"m_0_star": -20.5, "q": 1.0, "z_ref": 0.1},
       alpha_model="constant",
       alpha_kwargs={"alpha": -1.1},
   )

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
   ax.set_title("Evolving luminosity function", fontsize=TITLE_SIZE)
   ax.tick_params(axis="both", labelsize=TICK_SIZE)

   cbar = fig.colorbar(mesh, ax=ax)
   cbar.set_label(
       r"$\log_{10}\phi(M, z)$ [$\log_{10}(\mathrm{Mpc}^{-3}\,\mathrm{mag}^{-1})$]",
       fontsize=LABEL_SIZE,
   )
   cbar.ax.tick_params(labelsize=TICK_SIZE)

   plt.tight_layout()


Magnitude-limited LF-weighted redshift trend
--------------------------------------------

A survey magnitude limit selects different parts of the luminosity function at
different redshifts.

This example shows LF-weighted redshift trends for several fixed
apparent-magnitude limits. The absolute-magnitude limit is computed from the
cosmology-dependent distance modulus. At lower redshift, a flux-limited sample
can include relatively faint galaxies. At higher redshift, the same
apparent-magnitude limit corresponds to a brighter absolute-magnitude cut, so
only intrinsically brighter galaxies remain in the selected sample.

The result is not intended to be a full survey :math:`n(z)`, because it does not
include the survey volume element. Instead, it shows the luminosity function
selection factor that later enters magnitude-limited redshift-distribution
calculations.

.. plot::
   :include-source: True
   :width: 620

   import numpy as np
   import matplotlib.pyplot as plt
   import cmasher as cmr
   import pyccl as ccl

   from lfkit import LuminosityFunction

   LABEL_SIZE = 15
   TICK_SIZE = 13
   TITLE_SIZE = 17
   LEGEND_SIZE = 15

   colors_blue = cmr.take_cmap_colors("cmr.guppy", 3, cmap_range=(0.72, 0.96))

   cosmo = ccl.Cosmology(
       Omega_c=0.25,
       Omega_b=0.05,
       h=0.7,
       sigma8=0.8,
       n_s=0.96,
       transfer_function="bbks",
       matter_power_spectrum="linear",
   )

   lf = LuminosityFunction.evolving_schechter(
       phi_model="linear_p",
       phi_kwargs={"phi_0_star": 1.0e-3, "p": 0.6},
       m_star_model="linear_q",
       m_star_kwargs={"m_0_star": -20.5, "q": 0.8, "z_ref": 0.1},
       alpha_model="constant",
       alpha_kwargs={"alpha": -1.1},
   )

   redshift = np.linspace(0.05, 1.5, 160)
   apparent_limits = [23.5, 24.5, 25.5]

   fig, ax = plt.subplots(figsize=(7.0, 5.0))

   for m_lim, color in zip(apparent_limits, colors_blue):
       m_limit = lf.absolute_magnitude_limit(
           cosmo,
           redshift,
           m_lim=m_lim,
           corrections=None,
       )

       lf_selection = lf.integrated_number_density(
           z=redshift,
           m_bright=-25.0,
           m_faint=m_limit,
           n_m=700,
       )
       lf_selection /= np.trapezoid(lf_selection, redshift)

       ax.plot(
           redshift,
           lf_selection,
           lw=3,
           color=color,
           label=rf"$m_{{\rm lim}}={m_lim}$",
       )

   ax.set_xlabel("Redshift $z$", fontsize=LABEL_SIZE)
   ax.set_ylabel("LF selection", fontsize=LABEL_SIZE)
   ax.set_title("Magnitude-limited LF selection", fontsize=TITLE_SIZE)
   ax.tick_params(axis="both", labelsize=TICK_SIZE)
   ax.legend(frameon=True, fontsize=LEGEND_SIZE, loc="best")
   plt.tight_layout()


Cosmology dependence of the absolute-magnitude limit
----------------------------------------------------

The apparent-to-absolute magnitude conversion depends on cosmology through the
luminosity distance and distance modulus.

This plot compares the absolute-magnitude limit implied by the same apparent
magnitude cut in several cosmologies. The luminosity function is not used in
this plot; the comparison isolates the selection boundary itself.

Users can replace the entries in the ``cosmologies`` dictionary with any
:class:`pyccl.Cosmology` objects relevant to their own analysis.

.. plot::
   :include-source: True
   :width: 620

   import numpy as np
   import matplotlib.pyplot as plt
   import cmasher as cmr
   import pyccl as ccl

   from lfkit import LuminosityFunction

   LABEL_SIZE = 15
   TICK_SIZE = 13
   TITLE_SIZE = 17
   LEGEND_SIZE = 15

   colors_red = cmr.take_cmap_colors("cmr.guppy", 3, cmap_range=(0.03, 0.26))

   cosmologies = {
       r"$\Omega_{\rm m}=0.25,\ h=0.70$": ccl.Cosmology(
           Omega_c=0.20,
           Omega_b=0.05,
           h=0.70,
           sigma8=0.8,
           n_s=0.96,
           transfer_function="bbks",
           matter_power_spectrum="linear",
       ),
       r"$\Omega_{\rm m}=0.30,\ h=0.70$": ccl.Cosmology(
           Omega_c=0.25,
           Omega_b=0.05,
           h=0.70,
           sigma8=0.8,
           n_s=0.96,
           transfer_function="bbks",
           matter_power_spectrum="linear",
       ),
       r"$\Omega_{\rm m}=0.35,\ h=0.70$": ccl.Cosmology(
           Omega_c=0.30,
           Omega_b=0.05,
           h=0.70,
           sigma8=0.8,
           n_s=0.96,
           transfer_function="bbks",
           matter_power_spectrum="linear",
       ),
   }

   z = np.linspace(0.05, 1.5, 250)

   lf = LuminosityFunction.schechter(
       phi_star=1.0e-3,
       m_star=-20.5,
       alpha=-1.1,
   )

   m_limits = {}

   for label, cosmo in cosmologies.items():
       m_limits[label] = lf.absolute_magnitude_limit(
           cosmo,
           z,
           m_lim=24.5,
           corrections=None,
       )

   reference_label = r"$\Omega_{\rm m}=0.30,\ h=0.70$"
   reference = m_limits[reference_label]

   fig, (ax_top, ax_bottom) = plt.subplots(
       2,
       1,
       figsize=(7.0, 6.2),
       sharex=True,
       gridspec_kw={"height_ratios": [3, 1]},
       constrained_layout=True,
   )

   for (label, m_limit), color in zip(m_limits.items(), colors_red):
       ax_top.plot(z, m_limit, lw=3, color=color, label=label)
       ax_bottom.plot(z, m_limit - reference, lw=2.5, color=color)

   ax_top.invert_yaxis()
   ax_top.set_ylabel(r"$M_{\rm lim}(z)$", fontsize=LABEL_SIZE)
   ax_top.set_title(r"Cosmology dependence of $M_{\rm lim}(z)$", fontsize=TITLE_SIZE)
   ax_top.legend(frameon=True, fontsize=LEGEND_SIZE, loc="best")

   ax_bottom.axhline(0.0, lw=1.0, color="0.3")
   ax_bottom.set_xlabel("Redshift $z$", fontsize=LABEL_SIZE)
   ax_bottom.set_ylabel(r"$\Delta M_{\rm lim}$", fontsize=LABEL_SIZE)

   for ax in (ax_top, ax_bottom):
       ax.tick_params(axis="both", labelsize=TICK_SIZE)

   plt.tight_layout()


Cosmology dependence of LF selection
------------------------------------

The LF-weighted selection factor also depends on cosmology because the same
apparent-magnitude cut maps to a different absolute-magnitude limit.

This example keeps the luminosity function and apparent-magnitude limit fixed,
then changes only the cosmology. The curves are normalized to unit integral over
redshift, so the comparison emphasizes changes in shape rather than absolute
normalization.

The lower panel shows the residual relative to the reference cosmology,
:math:`\Omega_{\rm m}=0.30,\ h=0.70`.

This is still not a full survey :math:`n(z)`, because it does not include the
cosmological volume element. It is the luminosity function selection factor
alone.

.. plot::
   :include-source: True
   :width: 620

   import numpy as np
   import matplotlib.pyplot as plt
   import cmasher as cmr
   import pyccl as ccl

   from lfkit import LuminosityFunction

   LABEL_SIZE = 15
   TICK_SIZE = 13
   TITLE_SIZE = 17
   LEGEND_SIZE = 15

   colors_red = cmr.take_cmap_colors("cmr.guppy", 3, cmap_range=(0.03, 0.26))

   cosmologies = {
       r"$\Omega_{\rm m}=0.25,\ h=0.70$": ccl.Cosmology(
           Omega_c=0.20,
           Omega_b=0.05,
           h=0.70,
           sigma8=0.8,
           n_s=0.96,
           transfer_function="bbks",
           matter_power_spectrum="linear",
       ),
       r"$\Omega_{\rm m}=0.30,\ h=0.70$": ccl.Cosmology(
           Omega_c=0.25,
           Omega_b=0.05,
           h=0.70,
           sigma8=0.8,
           n_s=0.96,
           transfer_function="bbks",
           matter_power_spectrum="linear",
       ),
       r"$\Omega_{\rm m}=0.35,\ h=0.70$": ccl.Cosmology(
           Omega_c=0.30,
           Omega_b=0.05,
           h=0.70,
           sigma8=0.8,
           n_s=0.96,
           transfer_function="bbks",
           matter_power_spectrum="linear",
       ),
   }

   redshift = np.linspace(0.05, 1.5, 160)

   lf = LuminosityFunction.evolving_schechter(
       phi_model="linear_p",
       phi_kwargs={"phi_0_star": 1.0e-3, "p": 0.6},
       m_star_model="linear_q",
       m_star_kwargs={"m_0_star": -20.5, "q": 0.8, "z_ref": 0.1},
       alpha_model="constant",
       alpha_kwargs={"alpha": -1.1},
   )

   selections = {}

   for label, cosmo in cosmologies.items():
       m_limit = lf.absolute_magnitude_limit(
           cosmo,
           redshift,
           m_lim=24.5,
           corrections=None,
       )

       lf_selection = lf.integrated_number_density(
           z=redshift,
           m_bright=-25.0,
           m_faint=m_limit,
           n_m=700,
       )
       lf_selection /= np.trapezoid(lf_selection, redshift)

       selections[label] = lf_selection

   reference_label = r"$\Omega_{\rm m}=0.30,\ h=0.70$"
   reference = selections[reference_label]

   fig, (ax_top, ax_bottom) = plt.subplots(
       2,
       1,
       figsize=(7.0, 6.2),
       sharex=True,
       gridspec_kw={"height_ratios": [3, 1]},
       constrained_layout=True,
   )

   for (label, lf_selection), color in zip(selections.items(), colors_red):
       ax_top.plot(redshift, lf_selection, lw=3, color=color, label=label)
       ax_bottom.plot(redshift, lf_selection - reference, lw=2.5, color=color)

   ax_top.set_ylabel("LF selection", fontsize=LABEL_SIZE)
   ax_top.set_title(r"Cosmology dependence of LF selection", fontsize=TITLE_SIZE)
   ax_top.legend(frameon=True, fontsize=LEGEND_SIZE, loc="best")

   ax_bottom.axhline(0.0, lw=1.0, color="0.3")
   ax_bottom.set_xlabel("Redshift $z$", fontsize=LABEL_SIZE)
   ax_bottom.set_ylabel(r"$\Delta$ selection", fontsize=LABEL_SIZE)

   for ax in (ax_top, ax_bottom):
       ax.tick_params(axis="both", labelsize=TICK_SIZE)

   plt.tight_layout()


Cosmology and volume-weighted LF redshift trend
-----------------------------------------------

A full LF-based redshift trend for a magnitude-limited sample should include
both the luminosity function selection and the cosmological volume element.

This example multiplies the magnitude-integrated luminosity function by a
simple comoving-volume weight per steradian, :math:`\chi^2(z) / H(z)`, up to an
overall constant. The result is closer to the ingredient used in LF-dependent
:math:`n(z)` construction.

The curves are normalized to unit integral over redshift, so the comparison
shows how cosmology changes the shape of the redshift trend. The lower panel
shows the residual relative to the reference cosmology,
:math:`\Omega_{\rm m}=0.30,\ h=0.70`.

The absolute normalization depends on survey area, LF normalization, and the
exact volume convention used by the calling analysis.

.. plot::
   :include-source: True
   :width: 620

   import numpy as np
   import matplotlib.pyplot as plt
   import cmasher as cmr
   import pyccl as ccl

   from lfkit import LuminosityFunction

   LABEL_SIZE = 15
   TICK_SIZE = 13
   TITLE_SIZE = 17
   LEGEND_SIZE = 15

   colors_red = cmr.take_cmap_colors("cmr.guppy", 3, cmap_range=(0.03, 0.26))

   cosmologies = {
       r"$\Omega_{\rm m}=0.25,\ h=0.70$": ccl.Cosmology(
           Omega_c=0.20,
           Omega_b=0.05,
           h=0.70,
           sigma8=0.8,
           n_s=0.96,
           transfer_function="bbks",
           matter_power_spectrum="linear",
       ),
       r"$\Omega_{\rm m}=0.30,\ h=0.70$": ccl.Cosmology(
           Omega_c=0.25,
           Omega_b=0.05,
           h=0.70,
           sigma8=0.8,
           n_s=0.96,
           transfer_function="bbks",
           matter_power_spectrum="linear",
       ),
       r"$\Omega_{\rm m}=0.35,\ h=0.70$": ccl.Cosmology(
           Omega_c=0.30,
           Omega_b=0.05,
           h=0.70,
           sigma8=0.8,
           n_s=0.96,
           transfer_function="bbks",
           matter_power_spectrum="linear",
       ),
   }

   redshift = np.linspace(0.05, 1.5, 160)

   lf = LuminosityFunction.evolving_schechter(
       phi_model="linear_p",
       phi_kwargs={"phi_0_star": 1.0e-3, "p": 0.6},
       m_star_model="linear_q",
       m_star_kwargs={"m_0_star": -20.5, "q": 0.8, "z_ref": 0.1},
       alpha_model="constant",
       alpha_kwargs={"alpha": -1.1},
   )

   trends = {}

   for label, cosmo in cosmologies.items():
       scale_factor = 1.0 / (1.0 + redshift)

       chi = ccl.comoving_radial_distance(cosmo, scale_factor)
       h_over_h0 = ccl.h_over_h0(cosmo, scale_factor)

       volume_weight = chi**2 / h_over_h0

       m_limit = lf.absolute_magnitude_limit(
           cosmo,
           redshift,
           m_lim=24.5,
           corrections=None,
       )

       lf_selection = lf.integrated_number_density(
           z=redshift,
           m_bright=-25.0,
           m_faint=m_limit,
           n_m=700,
       )

       weighted_trend = volume_weight * lf_selection
       weighted_trend /= np.trapezoid(weighted_trend, redshift)

       trends[label] = weighted_trend

   reference_label = r"$\Omega_{\rm m}=0.30,\ h=0.70$"
   reference = trends[reference_label]

   fig, (ax_top, ax_bottom) = plt.subplots(
       2,
       1,
       figsize=(7.0, 6.2),
       sharex=True,
       gridspec_kw={"height_ratios": [3, 1]},
       constrained_layout=True,
   )

   for (label, weighted_trend), color in zip(trends.items(), colors_red):
       ax_top.plot(redshift, weighted_trend, lw=3, color=color, label=label)
       ax_bottom.plot(redshift, weighted_trend - reference, lw=2.5, color=color)

   ax_top.set_ylabel("Volume-weighted LF trend", fontsize=LABEL_SIZE)
   ax_top.set_title(r"Cosmology dependence with volume weighting", fontsize=TITLE_SIZE)
   ax_top.legend(frameon=True, fontsize=LEGEND_SIZE, loc="best")

   ax_bottom.axhline(0.0, lw=1.0, color="0.3")
   ax_bottom.set_xlabel("Redshift $z$", fontsize=LABEL_SIZE)
   ax_bottom.set_ylabel(r"$\Delta$ trend", fontsize=LABEL_SIZE)

   for ax in (ax_top, ax_bottom):
       ax.tick_params(axis="both", labelsize=TICK_SIZE)

   plt.tight_layout()
