.. |lfkitlogo| image:: /_static/logos/lfkit_logo-icon.png
   :alt: LFKit logo
   :width: 50px

|lfkitlogo| Available luminosity function models
================================================

LFKit exposes registered model names so users can inspect which luminosity
functions and redshift-dependent parameter models are available from the public
API.

This is useful in notebooks, examples, and tests because it lets users discover
valid model names without looking through the implementation modules.


Available luminosity function models
------------------------------------

.. code-block:: python

   from lfkit import LuminosityFunction

   LuminosityFunction.available_models()


Available apparent magnitude models
-----------------------------------

.. code-block:: python

   from lfkit import LuminosityFunction

   LuminosityFunction.available_from_m_models()


Available parameter models
--------------------------

.. code-block:: python

   from lfkit import LuminosityFunction

   LuminosityFunction.available_parameter_models()


Typical use
-----------

The registered names can be passed to constructors such as
:meth:`lfkit.LuminosityFunction.evolving_schechter`.

.. code-block:: python

   from lfkit import LuminosityFunction

   lf = LuminosityFunction.evolving_schechter(
       phi_model="linear_p",
       phi_kwargs={"phi_0_star": 1.0e-3, "p": 0.7},
       m_star_model="linear_q",
       m_star_kwargs={"m_0_star": -20.5, "q": 0.8, "z_ref": 0.1},
       alpha_model="constant",
       alpha_kwargs={"alpha": -1.1},
   )
