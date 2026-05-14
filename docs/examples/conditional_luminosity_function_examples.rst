.. |lfkitlogo| image:: /_static/logos/lfkit_logo-icon.png
   :alt: LFKit logo
   :width: 50px

|lfkitlogo| Conditional luminosity functions
============================================

This page shows how to evaluate conditional luminosity functions with LFKit.

A conditional luminosity function has the form :math:`\Phi(M \mid x)`, where
:math:`M` is absolute magnitude and :math:`x` is an external conditioning
variable. The conditioning variable is generic: it can represent redshift, halo
mass, environment, galaxy type, richness, stellar mass, or another quantity.

The examples below use redshift as the conditioning variable because it is a
natural choice for luminosity function evolution. The same functions can be
used with any other conditioning variable by replacing ``z`` with the desired
quantity.

The examples include:

* a conditional Schechter luminosity function,
* a conditional Schechter model using LFKit parameter models,
* a lognormal component,
* a modified Schechter-like component,
* a two-component lognormal plus modified-Schechter model,
* integrated number densities and component fractions.

The number-density units follow the normalization supplied to the luminosity
function. For example, if the amplitudes are supplied in
:math:`{\rm Mpc}^{-3}\,{\rm mag}^{-1}`, then :math:`\Phi(M \mid z)` has units
of :math:`{\rm Mpc}^{-3}\,{\rm mag}^{-1}`.


Conditional Schechter luminosity function
-----------------------------------------

A conditional Schechter luminosity function allows the Schechter parameters to
depend on an external variable.

This example makes the normalization and characteristic magnitude depend on
redshift. The faint-end slope is kept fixed. At each redshift, the model is
still a Schechter luminosity function, but the curve evolves smoothly with the
conditioning variable.

.. plot::
   :include-source: True
   :width: 520

   import numpy as np
   import matplotlib.pyplot as plt
   import cmasher as cmr

   from lfkit.photometry.conditional_lf_models import (
       conditional_schechter,
   )

   LABEL_SIZE = 15
   TICK_SIZE = 13
   TITLE_SIZE = 17
   LEGEND_SIZE = 15

   colors = cmr.take_cmap_colors("cmr.guppy", 3, cmap_range=(0.72, 0.96))

   absolute_mag = np.linspace(-24.0, -14.0, 500)
   redshifts = [0.1, 0.6, 1.1]

   fig, ax = plt.subplots(figsize=(7.0, 5.0))

   for z_value, color in zip(redshifts, colors):
       z = np.full_like(absolute_mag, z_value)

       phi = conditional_schechter(
           absolute_mag,
           z,
           phi_star=lambda z: 1.0e-3 * (1.0 + z) ** 0.8,
           m_star=lambda z: -20.5 - 0.7 * (z - 0.1),
           alpha=-1.1,
       )

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
       r"$\Phi(M \mid z)$ [$\mathrm{Mpc}^{-3}\,\mathrm{mag}^{-1}$]",
       fontsize=LABEL_SIZE,
   )
   ax.set_title("Conditional Schechter luminosity function", fontsize=TITLE_SIZE)
   ax.tick_params(axis="both", labelsize=TICK_SIZE)
   ax.legend(frameon=True, fontsize=LEGEND_SIZE, loc="best")
   plt.tight_layout()


Conditional Schechter surface
-----------------------------

The same model can be shown across the full magnitude-redshift plane.

The filled colour scale shows :math:`\log_{10}\Phi(M \mid z)`. The contours
mark constant abundance levels. This is a useful diagnostic for checking that
the conditional model behaves smoothly across the region where it will be used.

.. plot::
   :include-source: True
   :width: 560

   import numpy as np
   import matplotlib.pyplot as plt
   import cmasher as cmr

   from lfkit.photometry.conditional_lf_models import (
       conditional_schechter,
   )

   LABEL_SIZE = 15
   TICK_SIZE = 13
   TITLE_SIZE = 17

   absolute_mag = np.linspace(-24.0, -16.0, 220)
   redshift = np.linspace(0.0, 1.5, 180)

   mag_grid, z_grid = np.meshgrid(absolute_mag, redshift)

   phi = conditional_schechter(
       mag_grid,
       z_grid,
       phi_star=lambda z: 1.0e-3 * (1.0 + z) ** 0.8,
       m_star=lambda z: -20.5 - 0.7 * (z - 0.1),
       alpha=-1.1,
   )

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
       r"$\log_{10}\Phi(M \mid z)$ "
       r"[$\log_{10}(\mathrm{Mpc}^{-3}\,\mathrm{mag}^{-1})$]",
       fontsize=LABEL_SIZE,
   )
   cbar.ax.tick_params(labelsize=TICK_SIZE)

   plt.tight_layout()


Conditional Schechter model with LFKit parameter models
-------------------------------------------------------

LFKit can also evaluate conditional Schechter models using its registered
parameter models. This is useful when the desired dependence follows one of the
standard LFKit parameterizations.

Here, the normalization and characteristic magnitude evolve with the
conditioning variable, while the faint-end slope is constant.

.. plot::
   :include-source: True
   :width: 520

   import numpy as np
   import matplotlib.pyplot as plt
   import cmasher as cmr

   from lfkit.photometry.conditional_lf_models import (
       conditional_schechter_evolving,
   )

   LABEL_SIZE = 15
   TICK_SIZE = 13
   TITLE_SIZE = 17
   LEGEND_SIZE = 15

   colors = cmr.take_cmap_colors("cmr.guppy", 3, cmap_range=(0.03, 0.26))

   absolute_mag = np.linspace(-24.0, -14.0, 500)
   redshifts = [0.1, 0.6, 1.1]

   fig, ax = plt.subplots(figsize=(7.0, 5.0))

   for z_value, color in zip(redshifts, colors):
       z = np.full_like(absolute_mag, z_value)

       phi = conditional_schechter_evolving(
           absolute_mag,
           z,
           phi_model="linear_p",
           phi_kwargs={"phi_0_star": 1.0e-3, "p": 0.7},
           m_star_model="linear_q",
           m_star_kwargs={"m_0_star": -20.5, "q": 0.8, "z_ref": 0.1},
           alpha_model="constant",
           alpha_kwargs={"alpha": -1.1},
       )

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
       r"$\Phi(M \mid z)$ [$\mathrm{Mpc}^{-3}\,\mathrm{mag}^{-1}$]",
       fontsize=LABEL_SIZE,
   )
   ax.set_title("Conditional evolving Schechter model", fontsize=TITLE_SIZE)
   ax.tick_params(axis="both", labelsize=TICK_SIZE)
   ax.legend(frameon=True, fontsize=LEGEND_SIZE, loc="best")
   plt.tight_layout()


Lognormal conditional component
-------------------------------

A narrow lognormal component can represent a population concentrated around a
characteristic luminosity at fixed condition.

This example uses a mean absolute magnitude that becomes brighter with
redshift. The scatter is fixed, so the peak shifts while retaining a similar
width.

.. plot::
   :include-source: True
   :width: 520

   import numpy as np
   import matplotlib.pyplot as plt
   import cmasher as cmr

   from lfkit.photometry.conditional_lf_models import (
       lognormal_conditional_lf,
   )

   LABEL_SIZE = 15
   TICK_SIZE = 13
   TITLE_SIZE = 17
   LEGEND_SIZE = 15

   colors = cmr.take_cmap_colors("cmr.guppy", 3, cmap_range=(0.72, 0.96))

   absolute_mag = np.linspace(-24.0, -16.0, 500)
   redshifts = [0.1, 0.6, 1.1]

   fig, ax = plt.subplots(figsize=(7.0, 5.0))

   for z_value, color in zip(redshifts, colors):
       z = np.full_like(absolute_mag, z_value)

       phi = lognormal_conditional_lf(
           absolute_mag,
           z,
           mean_absolute_mag=lambda z: -20.8 - 0.6 * (z - 0.1),
           sigma_log_luminosity=0.18,
           amplitude=lambda z: 8.0e-4 * (1.0 + z) ** 0.4,
       )

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
       r"$\Phi_{\rm lognormal}(M \mid z)$ "
       r"[$\mathrm{Mpc}^{-3}\,\mathrm{mag}^{-1}$]",
       fontsize=LABEL_SIZE,
   )
   ax.set_title("Lognormal conditional LF component", fontsize=TITLE_SIZE)
   ax.tick_params(axis="both", labelsize=TICK_SIZE)
   ax.legend(frameon=True, fontsize=LEGEND_SIZE, loc="best")
   plt.tight_layout()


Modified Schechter conditional component
----------------------------------------

The modified Schechter component uses a squared exponential cutoff in luminosity
ratio. It is broader than the lognormal component and contributes over a wider
range of faint magnitudes.

.. plot::
   :include-source: True
   :width: 520

   import numpy as np
   import matplotlib.pyplot as plt
   import cmasher as cmr

   from lfkit.photometry.conditional_lf_models import (
       modified_schechter_conditional_lf,
   )

   LABEL_SIZE = 15
   TICK_SIZE = 13
   TITLE_SIZE = 17
   LEGEND_SIZE = 15

   colors = cmr.take_cmap_colors("cmr.guppy", 3, cmap_range=(0.03, 0.26))

   absolute_mag = np.linspace(-24.0, -14.0, 500)
   redshifts = [0.1, 0.6, 1.1]

   fig, ax = plt.subplots(figsize=(7.0, 5.0))

   for z_value, color in zip(redshifts, colors):
       z = np.full_like(absolute_mag, z_value)

       phi = modified_schechter_conditional_lf(
           absolute_mag,
           z,
           phi_star=lambda z: 1.2e-3 * (1.0 + z) ** 0.5,
           m_star=lambda z: -19.9 - 0.5 * (z - 0.1),
           alpha=lambda z: -1.05 - 0.10 * z,
       )

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
       r"$\Phi_{\rm modSch}(M \mid z)$ "
       r"[$\mathrm{Mpc}^{-3}\,\mathrm{mag}^{-1}$]",
       fontsize=LABEL_SIZE,
   )
   ax.set_title("Modified Schechter conditional LF component", fontsize=TITLE_SIZE)
   ax.tick_params(axis="both", labelsize=TICK_SIZE)
   ax.legend(frameon=True, fontsize=LEGEND_SIZE, loc="best")
   plt.tight_layout()


Standard, modified, and lognormal component shapes
--------------------------------------------------

It is useful to compare the component shapes at fixed condition. The standard
Schechter form has the usual exponential cutoff in luminosity ratio. The
modified Schechter component uses a squared exponential cutoff, making the
bright-end suppression sharper. The lognormal component is localized around a
mean luminosity and is useful for narrow populations.

.. plot::
   :include-source: True
   :width: 520

   import numpy as np
   import matplotlib.pyplot as plt
   import cmasher as cmr

   from lfkit.photometry.conditional_lf_models import (
       conditional_schechter,
       lognormal_conditional_lf,
       modified_schechter_conditional_lf,
   )

   LABEL_SIZE = 15
   TICK_SIZE = 13
   TITLE_SIZE = 17
   LEGEND_SIZE = 15

   colors = cmr.take_cmap_colors("cmr.guppy", 3, cmap_range=(0.08, 0.92))

   absolute_mag = np.linspace(-24.0, -14.0, 600)
   z_value = 0.6
   z = np.full_like(absolute_mag, z_value)

   phi_schechter = conditional_schechter(
       absolute_mag,
       z,
       phi_star=1.0e-3,
       m_star=-20.5,
       alpha=-1.1,
   )

   phi_modified = modified_schechter_conditional_lf(
       absolute_mag,
       z,
       phi_star=1.0e-3,
       m_star=-20.5,
       alpha=-1.1,
   )

   phi_lognormal = lognormal_conditional_lf(
       absolute_mag,
       z,
       mean_absolute_mag=-20.5,
       sigma_log_luminosity=0.20,
       amplitude=1.0e-3,
   )

   fig, ax = plt.subplots(figsize=(7.0, 5.0))

   ax.plot(
       absolute_mag,
       phi_schechter,
       lw=3,
       color=colors[0],
       label="Standard Schechter",
   )
   ax.plot(
       absolute_mag,
       phi_modified,
       lw=3,
       color=colors[1],
       label="Modified Schechter",
   )
   ax.plot(
       absolute_mag,
       phi_lognormal,
       lw=3,
       color=colors[2],
       label="Lognormal",
   )

   ax.set_yscale("log")
   ax.invert_xaxis()
   ax.set_xlabel("Absolute magnitude $M$", fontsize=LABEL_SIZE)
   ax.set_ylabel(
       r"$\Phi(M \mid z=0.6)$ "
       r"[$\mathrm{Mpc}^{-3}\,\mathrm{mag}^{-1}$]",
       fontsize=LABEL_SIZE,
   )
   ax.set_title("Conditional LF component shapes", fontsize=TITLE_SIZE)
   ax.tick_params(axis="both", labelsize=TICK_SIZE)
   ax.legend(frameon=True, fontsize=LEGEND_SIZE, loc="best")

   plt.tight_layout()


Two-component conditional luminosity function
---------------------------------------------

The lognormal and modified Schechter components can be combined into a
two-component conditional luminosity function.

This plot separates the lognormal component, the modified Schechter component,
and their sum at a fixed redshift. This is a useful way to check which component
dominates different magnitude ranges.

.. plot::
   :include-source: True
   :width: 520

   import numpy as np
   import matplotlib.pyplot as plt
   import cmasher as cmr

   from lfkit.photometry.conditional_lf_models import (
       lognormal_conditional_lf,
       modified_schechter_conditional_lf,
       two_component_conditional_lf,
   )

   LABEL_SIZE = 15
   TICK_SIZE = 13
   TITLE_SIZE = 17
   LEGEND_SIZE = 15

   colors_blue = cmr.take_cmap_colors("cmr.guppy", 3, cmap_range=(0.72, 0.96))
   colors_red = cmr.take_cmap_colors("cmr.guppy", 3, cmap_range=(0.03, 0.26))
   total_color = 0.5 * (np.array(colors_blue[1]) + np.array(colors_red[1]))

   absolute_mag = np.linspace(-24.0, -14.0, 500)
   z_value = 0.6
   z = np.full_like(absolute_mag, z_value)

   lognormal_phi = lognormal_conditional_lf(
       absolute_mag,
       z,
       mean_absolute_mag=lambda z: -20.8 - 0.6 * (z - 0.1),
       sigma_log_luminosity=0.18,
       amplitude=lambda z: 8.0e-4 * (1.0 + z) ** 0.4,
   )

   modified_phi = modified_schechter_conditional_lf(
       absolute_mag,
       z,
       phi_star=lambda z: 1.2e-3 * (1.0 + z) ** 0.5,
       m_star=lambda z: -19.9 - 0.5 * (z - 0.1),
       alpha=lambda z: -1.05 - 0.10 * z,
   )

   total_phi = two_component_conditional_lf(
       absolute_mag,
       z,
       lognormal_mean_absolute_mag=lambda z: -20.8 - 0.6 * (z - 0.1),
       lognormal_sigma_log_luminosity=0.18,
       lognormal_amplitude=lambda z: 8.0e-4 * (1.0 + z) ** 0.4,
       modified_phi_star=lambda z: 1.2e-3 * (1.0 + z) ** 0.5,
       modified_m_star=lambda z: -19.9 - 0.5 * (z - 0.1),
       modified_alpha=lambda z: -1.05 - 0.10 * z,
   )

   fig, ax = plt.subplots(figsize=(7.0, 5.0))

   ax.plot(
       absolute_mag,
       lognormal_phi,
       lw=3,
       color=colors_blue[1],
       label="Lognormal component",
   )
   ax.plot(
       absolute_mag,
       modified_phi,
       lw=3,
       color=colors_red[1],
       label="Modified Schechter component",
   )
   ax.plot(
       absolute_mag,
       total_phi,
       lw=3,
       color=total_color,
       label="Two-component total",
   )

   ax.set_yscale("log")
   ax.invert_xaxis()
   ax.set_xlabel("Absolute magnitude $M$", fontsize=LABEL_SIZE)
   ax.set_ylabel(
       r"$\Phi(M \mid z)$ [$\mathrm{Mpc}^{-3}\,\mathrm{mag}^{-1}$]",
       fontsize=LABEL_SIZE,
   )
   ax.set_title(r"Two-component conditional LF at $z=0.6$", fontsize=TITLE_SIZE)
   ax.tick_params(axis="both", labelsize=TICK_SIZE)
   ax.legend(frameon=True, fontsize=LEGEND_SIZE, loc="best")
   plt.tight_layout()


Two-component evolution
-----------------------

The two-component conditional luminosity function can be evaluated across
several redshifts. This example shows how the full model changes when both
components depend on the conditioning variable.

.. plot::
   :include-source: True
   :width: 520

   import numpy as np
   import matplotlib.pyplot as plt
   import cmasher as cmr

   from lfkit.photometry.conditional_lf_models import (
       two_component_conditional_lf,
   )

   LABEL_SIZE = 15
   TICK_SIZE = 13
   TITLE_SIZE = 17
   LEGEND_SIZE = 15

   colors = cmr.take_cmap_colors("cmr.guppy", 3, cmap_range=(0.72, 0.96))

   absolute_mag = np.linspace(-24.0, -14.0, 500)
   redshifts = [0.1, 0.6, 1.1]

   fig, ax = plt.subplots(figsize=(7.0, 5.0))

   for z_value, color in zip(redshifts, colors):
       z = np.full_like(absolute_mag, z_value)

       phi = two_component_conditional_lf(
           absolute_mag,
           z,
           lognormal_mean_absolute_mag=lambda z: -20.8 - 0.6 * (z - 0.1),
           lognormal_sigma_log_luminosity=0.18,
           lognormal_amplitude=lambda z: 8.0e-4 * (1.0 + z) ** 0.4,
           modified_phi_star=lambda z: 1.2e-3 * (1.0 + z) ** 0.5,
           modified_m_star=lambda z: -19.9 - 0.5 * (z - 0.1),
           modified_alpha=lambda z: -1.05 - 0.10 * z,
       )

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
       r"$\Phi_{\rm total}(M \mid z)$ "
       r"[$\mathrm{Mpc}^{-3}\,\mathrm{mag}^{-1}$]",
       fontsize=LABEL_SIZE,
   )
   ax.set_title("Two-component conditional LF", fontsize=TITLE_SIZE)
   ax.tick_params(axis="both", labelsize=TICK_SIZE)
   ax.legend(frameon=True, fontsize=LEGEND_SIZE, loc="best")
   plt.tight_layout()


Integrated conditional number density
-------------------------------------

A conditional luminosity function can be integrated over absolute magnitude at
each value of the conditioning variable.

This example uses LFKit's conditional luminosity-function integration helper to
integrate the lognormal component, the modified Schechter component, and the
two-component total over a fixed absolute-magnitude range. The result shows how
the selected number density changes with redshift.

.. plot::
   :include-source: True
   :width: 520

   import numpy as np
   import matplotlib.pyplot as plt
   import cmasher as cmr

   from lfkit.photometry.conditional_lf_integrals import (
       integrate_conditional_luminosity_function,
   )
   from lfkit.photometry.conditional_lf_models import (
       lognormal_conditional_lf,
       modified_schechter_conditional_lf,
       two_component_conditional_lf,
   )

   LABEL_SIZE = 15
   TICK_SIZE = 13
   TITLE_SIZE = 17
   LEGEND_SIZE = 15

   colors_blue = cmr.take_cmap_colors("cmr.guppy", 3, cmap_range=(0.72, 0.96))
   colors_red = cmr.take_cmap_colors("cmr.guppy", 3, cmap_range=(0.03, 0.26))
   total_color = 0.5 * (np.array(colors_blue[1]) + np.array(colors_red[1]))

   redshift = np.linspace(0.05, 1.5, 180)
   absolute_mag = np.linspace(-24.0, -14.0, 800)

   _, z_grid = np.meshgrid(absolute_mag, redshift)

   n_lognormal = integrate_conditional_luminosity_function(
       absolute_mag=absolute_mag,
       condition=z_grid,
       conditional_lf=lambda absolute_mag, condition: lognormal_conditional_lf(
           absolute_mag,
           condition,
           mean_absolute_mag=lambda z: -20.8 - 0.6 * (z - 0.1),
           sigma_log_luminosity=0.18,
           amplitude=lambda z: 8.0e-4 * (1.0 + z) ** 0.4,
       ),
       axis=1,
   )

   n_modified = integrate_conditional_luminosity_function(
       absolute_mag=absolute_mag,
       condition=z_grid,
       conditional_lf=lambda absolute_mag, condition: modified_schechter_conditional_lf(
           absolute_mag,
           condition,
           phi_star=lambda z: 1.2e-3 * (1.0 + z) ** 0.5,
           m_star=lambda z: -19.9 - 0.5 * (z - 0.1),
           alpha=lambda z: -1.05 - 0.10 * z,
       ),
       axis=1,
   )

   n_total = integrate_conditional_luminosity_function(
       absolute_mag=absolute_mag,
       condition=z_grid,
       conditional_lf=lambda absolute_mag, condition: two_component_conditional_lf(
           absolute_mag,
           condition,
           lognormal_mean_absolute_mag=lambda z: -20.8 - 0.6 * (z - 0.1),
           lognormal_sigma_log_luminosity=0.18,
           lognormal_amplitude=lambda z: 8.0e-4 * (1.0 + z) ** 0.4,
           modified_phi_star=lambda z: 1.2e-3 * (1.0 + z) ** 0.5,
           modified_m_star=lambda z: -19.9 - 0.5 * (z - 0.1),
           modified_alpha=lambda z: -1.05 - 0.10 * z,
       ),
       axis=1,
   )

   fig, ax = plt.subplots(figsize=(7.0, 5.0))

   ax.plot(
       redshift,
       n_lognormal,
       lw=3,
       color=colors_blue[1],
       label="Lognormal component",
   )
   ax.plot(
       redshift,
       n_modified,
       lw=3,
       color=colors_red[1],
       label="Modified Schechter component",
   )
   ax.plot(
       redshift,
       n_total,
       lw=3,
       color=total_color,
       label="Two-component total",
   )

   ax.set_yscale("log")
   ax.set_xlabel("Redshift $z$", fontsize=LABEL_SIZE)
   ax.set_ylabel(r"Integrated number density [$\mathrm{Mpc}^{-3}$]", fontsize=LABEL_SIZE)
   ax.set_title(r"Integrated conditional LF over $-24 \leq M \leq -14$", fontsize=TITLE_SIZE)
   ax.tick_params(axis="both", labelsize=TICK_SIZE)
   ax.legend(frameon=True, fontsize=LEGEND_SIZE, loc="best")
   plt.tight_layout()


Component fractions
-------------------

The relative contribution of each component can be summarized as a fraction of
the integrated two-component luminosity function.

This example uses LFKit's conditional luminosity-function integration helper to
compute the integrated lognormal and modified Schechter components. This is a
compact diagnostic for checking whether the selected population is dominated by
the lognormal component, the modified Schechter component, or a mixture of both.

.. plot::
   :include-source: True
   :width: 520

   import numpy as np
   import matplotlib.pyplot as plt
   import cmasher as cmr

   from lfkit.photometry.conditional_lf_integrals import (
       integrate_conditional_luminosity_function,
   )
   from lfkit.photometry.conditional_lf_models import (
       lognormal_conditional_lf,
       modified_schechter_conditional_lf,
   )

   LABEL_SIZE = 15
   TICK_SIZE = 13
   TITLE_SIZE = 17
   LEGEND_SIZE = 15

   colors_blue = cmr.take_cmap_colors("cmr.guppy", 3, cmap_range=(0.72, 0.96))
   colors_red = cmr.take_cmap_colors("cmr.guppy", 3, cmap_range=(0.03, 0.26))

   redshift = np.linspace(0.05, 1.5, 180)
   absolute_mag = np.linspace(-24.0, -14.0, 800)

   _, z_grid = np.meshgrid(absolute_mag, redshift)

   n_lognormal = integrate_conditional_luminosity_function(
       absolute_mag=absolute_mag,
       condition=z_grid,
       conditional_lf=lambda absolute_mag, condition: lognormal_conditional_lf(
           absolute_mag,
           condition,
           mean_absolute_mag=lambda z: -20.8 - 0.6 * (z - 0.1),
           sigma_log_luminosity=0.18,
           amplitude=lambda z: 8.0e-4 * (1.0 + z) ** 0.4,
       ),
       axis=1,
   )

   n_modified = integrate_conditional_luminosity_function(
       absolute_mag=absolute_mag,
       condition=z_grid,
       conditional_lf=lambda absolute_mag, condition: modified_schechter_conditional_lf(
           absolute_mag,
           condition,
           phi_star=lambda z: 1.2e-3 * (1.0 + z) ** 0.5,
           m_star=lambda z: -19.9 - 0.5 * (z - 0.1),
           alpha=lambda z: -1.05 - 0.10 * z,
       ),
       axis=1,
   )

   n_total = n_lognormal + n_modified

   lognormal_fraction = n_lognormal / n_total
   modified_fraction = n_modified / n_total

   fig, ax = plt.subplots(figsize=(7.0, 5.0))

   ax.plot(
       redshift,
       lognormal_fraction,
       lw=3,
       color=colors_blue[1],
       label="Lognormal fraction",
   )
   ax.plot(
       redshift,
       modified_fraction,
       lw=3,
       color=colors_red[1],
       label="Modified Schechter fraction",
   )

   ax.set_ylim(-0.05, 1.05)
   ax.set_xlabel("Redshift $z$", fontsize=LABEL_SIZE)
   ax.set_ylabel("Fraction of integrated LF", fontsize=LABEL_SIZE)
   ax.set_title(r"Component fractions over $-24 \leq M \leq -14$", fontsize=TITLE_SIZE)
   ax.tick_params(axis="both", labelsize=TICK_SIZE)
   ax.legend(frameon=True, fontsize=LEGEND_SIZE, loc="center right")
   plt.tight_layout()


Two-component LF surface
-----------------------------------------

The full two-component conditional luminosity function can be shown as a
surface in the magnitude-redshift plane.

The filled colour scale shows :math:`\log_{10}\Phi_{\rm total}(M \mid z)`.
The contours mark constant abundance levels. This view is useful for checking
whether the narrow component, broad component, and redshift dependence combine
smoothly.

.. plot::
   :include-source: True
   :width: 560

   import numpy as np
   import matplotlib.pyplot as plt
   import cmasher as cmr

   from lfkit.photometry.conditional_lf_models import (
       two_component_conditional_lf,
   )

   LABEL_SIZE = 15
   TICK_SIZE = 13
   TITLE_SIZE = 17

   absolute_mag = np.linspace(-24.0, -14.0, 220)
   redshift = np.linspace(0.0, 1.5, 180)

   mag_grid, z_grid = np.meshgrid(absolute_mag, redshift)

   phi = two_component_conditional_lf(
       mag_grid,
       z_grid,
       lognormal_mean_absolute_mag=lambda z: -20.8 - 0.6 * (z - 0.1),
       lognormal_sigma_log_luminosity=0.18,
       lognormal_amplitude=lambda z: 8.0e-4 * (1.0 + z) ** 0.4,
       modified_phi_star=lambda z: 1.2e-3 * (1.0 + z) ** 0.5,
       modified_m_star=lambda z: -19.9 - 0.5 * (z - 0.1),
       modified_alpha=lambda z: -1.05 - 0.10 * z,
   )

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
   ax.set_title("Two-component conditional LF surface", fontsize=TITLE_SIZE)
   ax.tick_params(axis="both", labelsize=TICK_SIZE)

   cbar = fig.colorbar(mesh, ax=ax)
   cbar.set_label(
       r"$\log_{10}\Phi_{\rm total}(M \mid z)$ "
       r"[$\log_{10}(\mathrm{Mpc}^{-3}\,\mathrm{mag}^{-1})$]",
       fontsize=LABEL_SIZE,
   )
   cbar.ax.tick_params(labelsize=TICK_SIZE)

   plt.tight_layout()


Halo-mass conditional luminosity function
-----------------------------------------

The conditioning variable does not need to be redshift. In halo-model
applications, a conditional luminosity function is often written as
:math:`\Phi(M \mid M_h)`, where :math:`M_h` is halo mass.

This example uses log halo mass as the conditioning variable and lets the
lognormal mean magnitude become brighter in more massive halos.

.. plot::
   :include-source: True
   :width: 520

   import numpy as np
   import matplotlib.pyplot as plt
   import cmasher as cmr

   from lfkit.photometry.conditional_lf_models import lognormal_conditional_lf

   LABEL_SIZE = 15
   TICK_SIZE = 13
   TITLE_SIZE = 17
   LEGEND_SIZE = 15

   colors = cmr.take_cmap_colors("cmr.guppy", 4, cmap_range=(0.08, 0.92))

   absolute_mag = np.linspace(-24.0, -16.0, 600)
   log_halo_masses = [11.5, 12.0, 12.5, 13.0]

   fig, ax = plt.subplots(figsize=(7.0, 5.0))

   for log_mh, color in zip(log_halo_masses, colors):
       condition = np.full_like(absolute_mag, log_mh)

       phi = lognormal_conditional_lf(
           absolute_mag,
           condition,
           mean_absolute_mag=lambda log_mh: -20.0 - 0.8 * (log_mh - 12.0),
           sigma_log_luminosity=0.18,
           amplitude=lambda log_mh: 5.0e-4 * 10.0 ** (0.3 * (log_mh - 12.0)),
       )

       ax.plot(
           absolute_mag,
           phi,
           lw=3,
           color=color,
           label=rf"$\log_{{10}} M_h={log_mh}$",
       )

   ax.set_yscale("log")
   ax.invert_xaxis()
   ax.set_xlabel("Absolute magnitude $M$", fontsize=LABEL_SIZE)
   ax.set_ylabel(
       r"$\Phi(M \mid M_h)$ [$\mathrm{Mpc}^{-3}\,\mathrm{mag}^{-1}$]",
       fontsize=LABEL_SIZE,
   )
   ax.set_title("Halo-mass conditional lognormal LF", fontsize=TITLE_SIZE)
   ax.tick_params(axis="both", labelsize=TICK_SIZE)
   ax.legend(frameon=True, fontsize=LEGEND_SIZE, loc="best")

   plt.tight_layout()


Mean magnitude from a conditional luminosity function
-----------------------------------------------------

Weighted integrals can be used to compute summary statistics of a conditional
luminosity function. For example, the mean absolute magnitude at fixed condition
is

:math:`\langle M \rangle(x) = \int M \Phi(M \mid x)\,dM / \int \Phi(M \mid x)\,dM`.

.. plot::
   :include-source: True
   :width: 520

   import numpy as np
   import matplotlib.pyplot as plt
   import cmasher as cmr

   from lfkit.photometry.conditional_lf_integrals import (
       integrate_conditional_luminosity_function,
       integrate_weighted_conditional_luminosity_function,
   )
   from lfkit.photometry.conditional_lf_models import two_component_conditional_lf

   LABEL_SIZE = 15
   TICK_SIZE = 13
   TITLE_SIZE = 17

   redshift = np.linspace(0.05, 1.5, 180)
   absolute_mag = np.linspace(-24.0, -14.0, 800)

   mag_grid, z_grid = np.meshgrid(absolute_mag, redshift)

   number_density = integrate_conditional_luminosity_function(
       absolute_mag=absolute_mag,
       condition=z_grid,
       conditional_lf=lambda absolute_mag, condition: two_component_conditional_lf(
           absolute_mag,
           condition,
           lognormal_mean_absolute_mag=lambda z: -20.8 - 0.6 * (z - 0.1),
           lognormal_sigma_log_luminosity=0.18,
           lognormal_amplitude=lambda z: 8.0e-4 * (1.0 + z) ** 0.4,
           modified_phi_star=lambda z: 1.2e-3 * (1.0 + z) ** 0.5,
           modified_m_star=lambda z: -19.9 - 0.5 * (z - 0.1),
           modified_alpha=lambda z: -1.05 - 0.10 * z,
       ),
       axis=1,
   )

   weighted_magnitude = integrate_weighted_conditional_luminosity_function(
       absolute_mag=absolute_mag,
       condition=z_grid,
       conditional_lf=lambda absolute_mag, condition: two_component_conditional_lf(
           absolute_mag,
           condition,
           lognormal_mean_absolute_mag=lambda z: -20.8 - 0.6 * (z - 0.1),
           lognormal_sigma_log_luminosity=0.18,
           lognormal_amplitude=lambda z: 8.0e-4 * (1.0 + z) ** 0.4,
           modified_phi_star=lambda z: 1.2e-3 * (1.0 + z) ** 0.5,
           modified_m_star=lambda z: -19.9 - 0.5 * (z - 0.1),
           modified_alpha=lambda z: -1.05 - 0.10 * z,
       ),
       weight=lambda absolute_mag, condition: absolute_mag,
       axis=1,
   )

   mean_magnitude = weighted_magnitude / number_density

   fig, ax = plt.subplots(figsize=(7.0, 5.0))

   ax.plot(
       redshift,
       mean_magnitude,
       lw=3,
       color=cmr.take_cmap_colors("cmr.guppy", 1, cmap_range=(0.7, 0.9))[0],
   )

   ax.invert_yaxis()
   ax.set_xlabel("Redshift $z$", fontsize=LABEL_SIZE)
   ax.set_ylabel(r"Mean absolute magnitude $\langle M \rangle$", fontsize=LABEL_SIZE)
   ax.set_title("Mean magnitude of the conditional LF", fontsize=TITLE_SIZE)
   ax.tick_params(axis="both", labelsize=TICK_SIZE)

   plt.tight_layout()


Selection-limited conditional number density
--------------------------------------------

Instead of integrating over a fixed absolute-magnitude range by hand, LFKit can
integrate a luminosity function callable over finite magnitude bounds. This is
useful for survey-like selections where only galaxies brighter than a limiting
absolute magnitude contribute to the selected sample.

Here, the limiting absolute magnitude becomes brighter with redshift. The
example compares the full number density over a fixed magnitude range with the
number density brighter than the redshift-dependent limit.

.. plot::
   :include-source: True
   :width: 520

   import numpy as np
   import matplotlib.pyplot as plt
   import cmasher as cmr

   from lfkit.photometry.conditional_lf_models import two_component_conditional_lf
   from lfkit.photometry.lf_integrals import integrated_number_density

   LABEL_SIZE = 15
   TICK_SIZE = 13
   TITLE_SIZE = 17
   LEGEND_SIZE = 15

   redshift = np.linspace(0.05, 1.5, 180)

   def lf(absolute_mag, z):
       return two_component_conditional_lf(
           absolute_mag,
           z,
           lognormal_mean_absolute_mag=lambda z: -20.8 - 0.6 * (z - 0.1),
           lognormal_sigma_log_luminosity=0.18,
           lognormal_amplitude=lambda z: 8.0e-4 * (1.0 + z) ** 0.4,
           modified_phi_star=lambda z: 1.2e-3 * (1.0 + z) ** 0.5,
           modified_m_star=lambda z: -19.9 - 0.5 * (z - 0.1),
           modified_alpha=lambda z: -1.05 - 0.10 * z,
       )

   limiting_mag = -18.5 - 1.2 * redshift

   n_total = integrated_number_density(
       redshift,
       lf,
       m_bright=-24.0,
       m_faint=-14.0,
       n_m=800,
   )

   n_selected = integrated_number_density(
       redshift,
       lf,
       m_bright=-24.0,
       m_faint=limiting_mag,
       n_m=800,
   )

   colors = cmr.take_cmap_colors("cmr.guppy", 3, cmap_range=(0.12, 0.9))

   fig, ax = plt.subplots(figsize=(7.0, 5.0))

   ax.plot(
       redshift,
       n_total,
       lw=3,
       color=colors[0],
       label="Full magnitude range",
   )
   ax.plot(
       redshift,
       n_selected,
       lw=3,
       color=colors[2],
       label="Brighter than limit",
   )

   ax.set_yscale("log")
   ax.set_xlabel("Redshift $z$", fontsize=LABEL_SIZE)
   ax.set_ylabel(
       r"Integrated number density [$\mathrm{Mpc}^{-3}$]",
       fontsize=LABEL_SIZE,
   )
   ax.set_title("Selection-limited conditional number density", fontsize=TITLE_SIZE)
   ax.tick_params(axis="both", labelsize=TICK_SIZE)
   ax.legend(frameon=True, fontsize=LEGEND_SIZE, loc="best")

   plt.tight_layout()


Selection fraction
------------------

The selected fraction is the ratio between the number density brighter than the
redshift-dependent limiting magnitude and the number density over the full
reference magnitude range.

.. plot::
   :include-source: True
   :width: 520

   import numpy as np
   import matplotlib.pyplot as plt
   import cmasher as cmr

   from lfkit.photometry.conditional_lf_models import two_component_conditional_lf
   from lfkit.photometry.lf_integrals import integrated_number_density

   LABEL_SIZE = 15
   TICK_SIZE = 13
   TITLE_SIZE = 17

   redshift = np.linspace(0.05, 1.5, 180)

   def lf(absolute_mag, z):
       return two_component_conditional_lf(
           absolute_mag,
           z,
           lognormal_mean_absolute_mag=lambda z: -20.8 - 0.6 * (z - 0.1),
           lognormal_sigma_log_luminosity=0.18,
           lognormal_amplitude=lambda z: 8.0e-4 * (1.0 + z) ** 0.4,
           modified_phi_star=lambda z: 1.2e-3 * (1.0 + z) ** 0.5,
           modified_m_star=lambda z: -19.9 - 0.5 * (z - 0.1),
           modified_alpha=lambda z: -1.05 - 0.10 * z,
       )

   limiting_mag = -18.5 - 1.2 * redshift

   n_total = integrated_number_density(
       redshift,
       lf,
       m_bright=-24.0,
       m_faint=-14.0,
       n_m=800,
   )

   n_selected = integrated_number_density(
       redshift,
       lf,
       m_bright=-24.0,
       m_faint=limiting_mag,
       n_m=800,
   )

   selected_fraction = n_selected / n_total

   color = cmr.take_cmap_colors("cmr.guppy", 1, cmap_range=(0.72, 0.9))[0]

   fig, ax = plt.subplots(figsize=(7.0, 5.0))

   ax.plot(
       redshift,
       selected_fraction,
       lw=3,
       color=color,
   )

   ax.set_ylim(-0.05, 1.05)
   ax.set_xlabel("Redshift $z$", fontsize=LABEL_SIZE)
   ax.set_ylabel("Selected fraction", fontsize=LABEL_SIZE)
   ax.set_title("Fraction brighter than the limiting magnitude", fontsize=TITLE_SIZE)
   ax.tick_params(axis="both", labelsize=TICK_SIZE)

   plt.tight_layout()
