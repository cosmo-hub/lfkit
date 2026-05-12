.. |lfkitlogo| image:: /_static/logos/lfkit_logo-icon.png
   :alt: LFKit logo
   :width: 50px

|lfkitlogo| Conditional luminosity-function examples
====================================================

This page shows how to use :class:`lfkit.LuminosityFunction` to construct and
evaluate conditional luminosity functions.

A conditional luminosity function has the form :math:`\Phi(M \mid x)`, where
:math:`M` is absolute magnitude and :math:`x` is an external conditioning
variable. In these examples, the public :class:`lfkit.LuminosityFunction`
interface uses redshift as the conditioning variable through ``phi(M, z)``.

Conditional luminosity functions are useful when the luminosity function
parameters depend on another quantity, such as redshift, halo mass, galaxy
type, environment, richness, or stellar mass. The examples below focus on
redshift-dependent behaviour because it fits naturally into the main LFKit API.

The examples include conditional Schechter models and central/satellite
components. The central component is represented by a narrow lognormal term,
while the satellite component is represented by a modified Schechter-like term.

All examples below are executable via ``.. plot::``.

The number-density units follow the normalization of the luminosity function.
For example, if the amplitudes are supplied in comoving
:math:`{\rm Mpc}^{-3}\,{\rm mag}^{-1}`, then :math:`\Phi(M \mid z)` has units of
:math:`{\rm Mpc}^{-3}\,{\rm mag}^{-1}`.


Conditional Schechter luminosity function
-----------------------------------------

A conditional Schechter luminosity function allows one or more Schechter
parameters to depend on the conditioning variable.

This example makes the normalization and characteristic magnitude depend on
redshift. The faint-end slope is kept fixed. The result is still a Schechter
luminosity function at each redshift, but the curve changes as the conditioning
variable changes.

This is useful when a single fixed luminosity function is too restrictive but a
full tabulated model is not needed.

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
   redshifts = [0.1, 0.6, 1.1]

   lf = LuminosityFunction.conditional_schechter(
       phi_star=lambda z: 1.0e-3 * (1.0 + z) ** 0.8,
       m_star=lambda z: -20.5 - 0.7 * (z - 0.1),
       alpha=-1.1,
   )

   fig, ax = plt.subplots(figsize=(7.0, 5.0))

   for z_value, color in zip(redshifts, colors_blue):
       z = np.full_like(absolute_mag, z_value)
       phi = lf.phi(absolute_mag, z)

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
   ax.set_ylabel(r"$\Phi(M \mid z)$ [$\mathrm{Mpc}^{-3}\,\mathrm{mag}^{-1}$]", fontsize=LABEL_SIZE)
   ax.set_title("Conditional Schechter luminosity function", fontsize=TITLE_SIZE)
   ax.tick_params(axis="both", labelsize=TICK_SIZE)
   ax.legend(frameon=True, fontsize=LEGEND_SIZE, loc="best")
   plt.tight_layout()


Conditional Schechter surface
-----------------------------

The same conditional Schechter model can be visualized across the full
magnitude-redshift plane.

The filled colour scale shows :math:`\log_{10}\Phi(M \mid z)`. The white
contours mark constant :math:`\log_{10}\Phi(M \mid z)` levels at -5, -4, -3,
and -2. These contours make it easier to see where equal-abundance regions sit
as both magnitude and redshift change.

This plot is useful for checking that the conditional model varies smoothly
across the range where it will be used.

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

   absolute_mag = np.linspace(-24.0, -16.0, 220)
   redshift = np.linspace(0.0, 1.5, 180)

   mag_grid, z_grid = np.meshgrid(absolute_mag, redshift)

   lf = LuminosityFunction.conditional_schechter(
       phi_star=lambda z: 1.0e-3 * (1.0 + z) ** 0.8,
       m_star=lambda z: -20.5 - 0.7 * (z - 0.1),
       alpha=-1.1,
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
   ax.set_title("Conditional Schechter LF surface", fontsize=TITLE_SIZE)
   ax.tick_params(axis="both", labelsize=TICK_SIZE)

   cbar = fig.colorbar(mesh, ax=ax)
   cbar.set_label(
       r"$\log_{10}\Phi(M \mid z)$ [$\log_{10}(\mathrm{Mpc}^{-3}\,\mathrm{mag}^{-1})$]",
       fontsize=LABEL_SIZE,
   )
   cbar.ax.tick_params(labelsize=TICK_SIZE)

   plt.tight_layout()


Conditional evolving Schechter model
------------------------------------

A conditional evolving Schechter model uses LFKit's registered parameter models
to define how :math:`\phi_\star`, :math:`M_\star`, and :math:`\alpha` depend on
the conditioning variable.

This example is similar to the standard evolving Schechter interface, but it is
included here because it can be used in the same conditional-LF workflow. The
conditioning variable is passed through ``phi(M, z)``.

This is useful when the desired parameter evolution already matches one of
LFKit's registered parameter models.

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
   redshifts = [0.1, 0.6, 1.1]

   lf = LuminosityFunction.conditional_evolving_schechter(
       phi_model="linear_p",
       phi_kwargs={"phi_0_star": 1.0e-3, "p": 0.7},
       m_star_model="linear_q",
       m_star_kwargs={"m_0_star": -20.5, "q": 0.8, "z_ref": 0.1},
       alpha_model="constant",
       alpha_kwargs={"alpha": -1.1},
   )

   fig, ax = plt.subplots(figsize=(7.0, 5.0))

   for z_value, color in zip(redshifts, colors_red):
       z = np.full_like(absolute_mag, z_value)
       phi = lf.phi(absolute_mag, z)

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
   ax.set_ylabel(r"$\Phi(M \mid z)$ [$\mathrm{Mpc}^{-3}\,\mathrm{mag}^{-1}$]", fontsize=LABEL_SIZE)
   ax.set_title("Conditional evolving Schechter model", fontsize=TITLE_SIZE)
   ax.tick_params(axis="both", labelsize=TICK_SIZE)
   ax.legend(frameon=True, fontsize=LEGEND_SIZE, loc="best")
   plt.tight_layout()


Central lognormal conditional luminosity function
-------------------------------------------------

A central-galaxy conditional luminosity function can be represented by a narrow
lognormal component in luminosity, written here in magnitude space.

This example shows a central component whose mean absolute magnitude becomes
brighter with redshift. The scatter is kept fixed. The peak therefore shifts in
absolute magnitude while retaining a similar width.

This type of component is useful in central/satellite decompositions where the
central galaxy population is concentrated around a characteristic luminosity at
fixed condition.

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

   absolute_mag = np.linspace(-24.0, -16.0, 500)
   redshifts = [0.1, 0.6, 1.1]

   lf = LuminosityFunction.central_lognormal_conditional(
       mean_absolute_mag=lambda z: -20.8 - 0.6 * (z - 0.1),
       sigma_log_luminosity=0.18,
       amplitude=lambda z: 8.0e-4 * (1.0 + z) ** 0.4,
   )

   fig, ax = plt.subplots(figsize=(7.0, 5.0))

   for z_value, color in zip(redshifts, colors_blue):
       z = np.full_like(absolute_mag, z_value)
       phi = lf.phi(absolute_mag, z)

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
   ax.set_ylabel(r"$\Phi_{\rm cen}(M \mid z)$ [$\mathrm{Mpc}^{-3}\,\mathrm{mag}^{-1}$]", fontsize=LABEL_SIZE)
   ax.set_title("Central lognormal conditional LF", fontsize=TITLE_SIZE)
   ax.tick_params(axis="both", labelsize=TICK_SIZE)
   ax.legend(frameon=True, fontsize=LEGEND_SIZE, loc="best")
   plt.tight_layout()


Satellite modified-Schechter conditional luminosity function
------------------------------------------------------------

A satellite conditional luminosity function can be represented by a modified
Schechter-like component.

This example uses a satellite model with an exponential cutoff
:math:`\exp[-(L/L_\star)^2]`. Compared with the central lognormal component,
the satellite component is broader and contributes more strongly across a wider
range of faint magnitudes.

This is useful for modeling satellite populations separately from central
galaxies.

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
   redshifts = [0.1, 0.6, 1.1]

   lf = LuminosityFunction.satellite_modified_schechter_conditional(
       phi_star=lambda z: 1.2e-3 * (1.0 + z) ** 0.5,
       m_star=lambda z: -19.9 - 0.5 * (z - 0.1),
       alpha=lambda z: -1.05 - 0.10 * z,
   )

   fig, ax = plt.subplots(figsize=(7.0, 5.0))

   for z_value, color in zip(redshifts, colors_red):
       z = np.full_like(absolute_mag, z_value)
       phi = lf.phi(absolute_mag, z)

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
   ax.set_ylabel(r"$\Phi_{\rm sat}(M \mid z)$ [$\mathrm{Mpc}^{-3}\,\mathrm{mag}^{-1}$]", fontsize=LABEL_SIZE)
   ax.set_title("Satellite modified-Schechter conditional LF", fontsize=TITLE_SIZE)
   ax.tick_params(axis="both", labelsize=TICK_SIZE)
   ax.legend(frameon=True, fontsize=LEGEND_SIZE, loc="best")
   plt.tight_layout()


Central and satellite components
--------------------------------

The central and satellite conditional luminosity functions can be combined into
a single central-plus-satellite model.

This plot separates the central component, the satellite component, and their
sum at a fixed redshift. The central component is narrow and localized around
the mean central magnitude. The satellite component is broader and dominates
over a wider faint-magnitude range.

This decomposition is useful when checking which part of the galaxy population
drives the total luminosity function.

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
   c_mid = 0.5 * (np.array(colors_blue[1]) + np.array(colors_red[1]))

   absolute_mag = np.linspace(-24.0, -14.0, 500)
   z_value = 0.6
   z = np.full_like(absolute_mag, z_value)

   central = LuminosityFunction.central_lognormal_conditional(
       mean_absolute_mag=lambda z: -20.8 - 0.6 * (z - 0.1),
       sigma_log_luminosity=0.18,
       amplitude=lambda z: 8.0e-4 * (1.0 + z) ** 0.4,
   )

   satellite = LuminosityFunction.satellite_modified_schechter_conditional(
       phi_star=lambda z: 1.2e-3 * (1.0 + z) ** 0.5,
       m_star=lambda z: -19.9 - 0.5 * (z - 0.1),
       alpha=lambda z: -1.05 - 0.10 * z,
   )

   total = LuminosityFunction.central_satellite_conditional(
       central_mean_absolute_mag=lambda z: -20.8 - 0.6 * (z - 0.1),
       central_sigma_log_luminosity=0.18,
       central_amplitude=lambda z: 8.0e-4 * (1.0 + z) ** 0.4,
       satellite_phi_star=lambda z: 1.2e-3 * (1.0 + z) ** 0.5,
       satellite_alpha=lambda z: -1.05 - 0.10 * z,
       satellite_m_star=lambda z: -19.9 - 0.5 * (z - 0.1),
   )

   central_phi = central.phi(absolute_mag, z)
   satellite_phi = satellite.phi(absolute_mag, z)
   total_phi = total.phi(absolute_mag, z)

   fig, ax = plt.subplots(figsize=(7.0, 5.0))

   ax.plot(
       absolute_mag,
       central_phi,
       lw=3,
       color=colors_blue[1],
       label="Central",
   )
   ax.plot(
       absolute_mag,
       satellite_phi,
       lw=3,
       color=colors_red[1],
       label="Satellite",
   )
   ax.plot(
       absolute_mag,
       total_phi,
       lw=3,
       color=c_mid,
       label="Central + satellite",
   )

   ax.set_yscale("log")
   ax.invert_xaxis()
   ax.set_xlabel("Absolute magnitude $M$", fontsize=LABEL_SIZE)
   ax.set_ylabel(r"$\Phi(M \mid z)$ [$\mathrm{Mpc}^{-3}\,\mathrm{mag}^{-1}$]", fontsize=LABEL_SIZE)
   ax.set_title(r"Central and satellite components at $z=0.6$", fontsize=TITLE_SIZE)
   ax.tick_params(axis="both", labelsize=TICK_SIZE)
   ax.legend(frameon=True, fontsize=LEGEND_SIZE, loc="best")
   plt.tight_layout()


Central-plus-satellite evolution
--------------------------------

The central-plus-satellite model can also be evaluated at several redshifts.

This plot shows how the total conditional luminosity function changes when the
central and satellite parameters vary with redshift. The model combines a
narrow central component with a broader satellite component at each redshift.

This is useful when the total luminosity function is needed, but the model
still keeps a physically interpretable central/satellite split.

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
   redshifts = [0.1, 0.6, 1.1]

   lf = LuminosityFunction.central_satellite_conditional(
       central_mean_absolute_mag=lambda z: -20.8 - 0.6 * (z - 0.1),
       central_sigma_log_luminosity=0.18,
       central_amplitude=lambda z: 8.0e-4 * (1.0 + z) ** 0.4,
       satellite_phi_star=lambda z: 1.2e-3 * (1.0 + z) ** 0.5,
       satellite_alpha=lambda z: -1.05 - 0.10 * z,
       satellite_m_star=lambda z: -19.9 - 0.5 * (z - 0.1),
   )

   fig, ax = plt.subplots(figsize=(7.0, 5.0))

   for z_value, color in zip(redshifts, colors_blue):
       z = np.full_like(absolute_mag, z_value)
       phi = lf.phi(absolute_mag, z)

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
   ax.set_ylabel(r"$\Phi_{\rm cen+sat}(M \mid z)$ [$\mathrm{Mpc}^{-3}\,\mathrm{mag}^{-1}$]", fontsize=LABEL_SIZE)
   ax.set_title("Central-plus-satellite conditional LF", fontsize=TITLE_SIZE)
   ax.tick_params(axis="both", labelsize=TICK_SIZE)
   ax.legend(frameon=True, fontsize=LEGEND_SIZE, loc="best")
   plt.tight_layout()


Integrated conditional number density
-------------------------------------

A conditional luminosity function can be integrated over absolute magnitude at
each value of the conditioning variable.

This example integrates the central component, satellite component, and total
central-plus-satellite luminosity function over a fixed absolute-magnitude
range. The result shows how the selected number density changes with redshift.

This type of calculation is useful when the conditional luminosity function is
used as an ingredient in redshift-distribution or abundance calculations.

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
   c_mid = 0.5 * (np.array(colors_blue[1]) + np.array(colors_red[1]))

   redshift = np.linspace(0.05, 1.5, 180)

   central = LuminosityFunction.central_lognormal_conditional(
       mean_absolute_mag=lambda z: -20.8 - 0.6 * (z - 0.1),
       sigma_log_luminosity=0.18,
       amplitude=lambda z: 8.0e-4 * (1.0 + z) ** 0.4,
   )

   satellite = LuminosityFunction.satellite_modified_schechter_conditional(
       phi_star=lambda z: 1.2e-3 * (1.0 + z) ** 0.5,
       m_star=lambda z: -19.9 - 0.5 * (z - 0.1),
       alpha=lambda z: -1.05 - 0.10 * z,
   )

   total = LuminosityFunction.central_satellite_conditional(
       central_mean_absolute_mag=lambda z: -20.8 - 0.6 * (z - 0.1),
       central_sigma_log_luminosity=0.18,
       central_amplitude=lambda z: 8.0e-4 * (1.0 + z) ** 0.4,
       satellite_phi_star=lambda z: 1.2e-3 * (1.0 + z) ** 0.5,
       satellite_alpha=lambda z: -1.05 - 0.10 * z,
       satellite_m_star=lambda z: -19.9 - 0.5 * (z - 0.1),
   )

   n_central = central.integrated_number_density(
       redshift,
       m_bright=-24.0,
       m_faint=-14.0,
       n_m=800,
   )
   n_satellite = satellite.integrated_number_density(
       redshift,
       m_bright=-24.0,
       m_faint=-14.0,
       n_m=800,
   )
   n_total = total.integrated_number_density(
       redshift,
       m_bright=-24.0,
       m_faint=-14.0,
       n_m=800,
   )

   fig, ax = plt.subplots(figsize=(7.0, 5.0))

   ax.plot(redshift, n_central, lw=3, color=colors_blue[1], label="Central")
   ax.plot(redshift, n_satellite, lw=3, color=colors_red[1], label="Satellite")
   ax.plot(redshift, n_total, lw=3, color=c_mid, label="Central + satellite")

   ax.set_yscale("log")
   ax.set_xlabel("Redshift $z$", fontsize=LABEL_SIZE)
   ax.set_ylabel(r"Integrated number density [$\mathrm{Mpc}^{-3}$]", fontsize=LABEL_SIZE)
   ax.set_title(r"Integrated conditional LF over $-24 \leq M \leq -14$", fontsize=TITLE_SIZE)
   ax.tick_params(axis="both", labelsize=TICK_SIZE)
   ax.legend(frameon=True, fontsize=LEGEND_SIZE, loc="best")
   plt.tight_layout()


Central fraction
----------------

The relative contribution of central and satellite galaxies can be summarized
as a fraction of the integrated central-plus-satellite luminosity function.

This example computes the central fraction over a fixed absolute-magnitude
range. The fraction changes with redshift because the central and satellite
components have different conditional parameter dependence.

This is a compact diagnostic for checking whether the selected population is
central-dominated, satellite-dominated, or mixed.

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

   redshift = np.linspace(0.05, 1.5, 180)

   central = LuminosityFunction.central_lognormal_conditional(
       mean_absolute_mag=lambda z: -20.8 - 0.6 * (z - 0.1),
       sigma_log_luminosity=0.18,
       amplitude=lambda z: 8.0e-4 * (1.0 + z) ** 0.4,
   )

   satellite = LuminosityFunction.satellite_modified_schechter_conditional(
       phi_star=lambda z: 1.2e-3 * (1.0 + z) ** 0.5,
       m_star=lambda z: -19.9 - 0.5 * (z - 0.1),
       alpha=lambda z: -1.05 - 0.10 * z,
   )

   n_central = central.integrated_number_density(
       redshift,
       m_bright=-24.0,
       m_faint=-14.0,
       n_m=800,
   )
   n_satellite = satellite.integrated_number_density(
       redshift,
       m_bright=-24.0,
       m_faint=-14.0,
       n_m=800,
   )

   central_fraction = n_central / (n_central + n_satellite)
   satellite_fraction = n_satellite / (n_central + n_satellite)

   fig, ax = plt.subplots(figsize=(7.0, 5.0))

   ax.plot(
       redshift,
       central_fraction,
       lw=3,
       color=colors_blue[1],
       label="Central fraction",
   )
   ax.plot(
       redshift,
       satellite_fraction,
       lw=3,
       color=colors_red[1],
       label="Satellite fraction",
   )

   ax.set_ylim(-0.05, 1.05)
   ax.set_xlabel("Redshift $z$", fontsize=LABEL_SIZE)
   ax.set_ylabel("Fraction of integrated LF", fontsize=LABEL_SIZE)
   ax.set_title(r"Central and satellite fractions over $-24 \leq M \leq -14$", fontsize=TITLE_SIZE)
   ax.tick_params(axis="both", labelsize=TICK_SIZE)
   ax.legend(frameon=True, fontsize=LEGEND_SIZE, loc="center right")
   plt.tight_layout()


Central-plus-satellite surface
------------------------------

The full central-plus-satellite conditional luminosity function can be shown as
a two-dimensional surface.

The filled colour scale shows :math:`\log_{10}\Phi_{\rm cen+sat}(M \mid z)`.
The white contours mark constant :math:`\log_{10}\Phi_{\rm cen+sat}(M \mid z)`
levels at -5, -4, -3, and -2.

This view is useful for checking whether the central peak, satellite tail, and
redshift dependence combine smoothly over the full magnitude-redshift range.

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

   absolute_mag = np.linspace(-24.0, -14.0, 220)
   redshift = np.linspace(0.0, 1.5, 180)

   mag_grid, z_grid = np.meshgrid(absolute_mag, redshift)

   lf = LuminosityFunction.central_satellite_conditional(
       central_mean_absolute_mag=lambda z: -20.8 - 0.6 * (z - 0.1),
       central_sigma_log_luminosity=0.18,
       central_amplitude=lambda z: 8.0e-4 * (1.0 + z) ** 0.4,
       satellite_phi_star=lambda z: 1.2e-3 * (1.0 + z) ** 0.5,
       satellite_alpha=lambda z: -1.05 - 0.10 * z,
       satellite_m_star=lambda z: -19.9 - 0.5 * (z - 0.1),
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
   ax.set_title("Central-plus-satellite conditional LF surface", fontsize=TITLE_SIZE)
   ax.tick_params(axis="both", labelsize=TICK_SIZE)

   cbar = fig.colorbar(mesh, ax=ax)
   cbar.set_label(
       r"$\log_{10}\Phi_{\rm cen+sat}(M \mid z)$ [$\log_{10}(\mathrm{Mpc}^{-3}\,\mathrm{mag}^{-1})$]",
       fontsize=LABEL_SIZE,
   )
   cbar.ax.tick_params(labelsize=TICK_SIZE)

   plt.tight_layout()
