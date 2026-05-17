.. _contributing:

.. |lfkitlogo| image:: /_static/logos/lfkit_logo.png
   :alt: LFKit logo black
   :width: 50px


|lfkitlogo| Contributing
========================

Contributions in any shape or form are appreciated.
Below are some minimal guidelines to get started.

Development of ``lfkit`` is organised around `the Github repository <https://github.com/cosmology-kit/lfkit>`__.
Contributing usually requires `setting up an account <https://github.com/signup>`__ on Github.
No worries, it is free of charge!

When submitting contributions, please write in clear and correct English using full sentences.
Be concise and avoid unnecessarily formulaic descriptions.

***********************
Submitting bugs or code
***********************

Submitting a bug report or feature request can be done by `opening an issue on Github <https://github.com/cosmology-kit/lfkit/issues/new/choose>`__.
In the case of a bug report please make sure to

- describe the expected behaviour,
- describe the actual behaviour,
- the version(s) of ``lfkit`` that produce the bug,
- any specific environment details.

In the case of a feature request, please make sure to

- describe the feature,
- describe why the feature is useful.

Submitting a code contribution can be done by `opening a pull request on Github <https://github.com/cosmology-kit/lfkit/compare>`__.
In the pull request, please make sure of the following:

- The description of the pull request contains a high-level overview of what is implemented in the contribution.
- The description describes the reason for the addition.
  Specifically, it describes what problem it is supposed to solve.
  If the pull request resolves a bug or implements a feature for which an issue exists, make sure to refer to this.
- The ``lfkit`` workflows complete successfully.
  The workflows will activate when a pull request is created, but can also be run locally on your computer as described below.

***************************
Running ``lfkit`` workflows
***************************

``lfkit`` uses `tox <https://tox.wiki>`__ to run its workflows.
For development, clone the repository and install the development dependencies::

   git clone https://github.com/cosmology-kit/lfkit.git
   cd lfkit
   pip install -e ".[dev]"

All workflows can be run consecutively by calling tox from the project root directory::

   tox

Specific workflows can also be run in isolation.
The following workflows are provided:

=======
Linting
=======

Code for ``lfkit`` must comply with `PEP-8 <https://peps.python.org/pep-0008/>`__.
Comments and docstrings must be compatible with `Google-style comments and docstrings <https://google.github.io/styleguide/pyguide.html#s3.8-comments-and-docstrings>`__.

The linting workflow can be run using::

   tox -e lint

Note that tox uses `ruff <https://docs.astral.sh/ruff/>`__ as the actual linter.
Options can be passed to `ruff check` by supplying them as command-line arguments to tox.
For example, to address fixable linting errors, use::

   tox -e lint -- --fix

=============
Documentation
=============

Documentation is written in `reStructuredText <https://docutils.sourceforge.io/rst.html>`__.
Code documentation is generated from docstrings, and the documentation can be built locally with::

   tox -e docs

The new documentation will be placed in the ``docs/_build`` directory.
Newly created rST files may need to be added to the appropriate table of contents files by hand.

Note that ``tox -e docs`` will use the ``html`` builder of sphinx-build.
A different builder can be selected by passing it as a command-line argument to tox.
For example, to run the doctest build::

   tox -e docs -- doctest

A list of supported options can be found on the `Builders <https://www.sphinx-doc.org/en/master/usage/builders/index.html#builders>`__ page of Sphinx.

The entire documentation for the head of the main branch and all release tags can be generated using::

   tox -e docs-releases

=======
Testing
=======

Contributions that contain new code must include tests in the appropriate files in the ``tests`` directory.
The test suite can be run locally with::

   tox -m test

Note that this will attempt to run the test suite for all supported Python versions.
It will automatically skip versions which are not locally available.

If the test suite must be run for a specific version of Python, that specific environment can be called.
For example, to test against Python 3.13::

   tox -e py313

For development it is sometimes useful to run a single test, as this is much faster than running the entire test suite.
To do so, add it as a command-line argument separated from the ``tox`` invocation by a ``--``.
For example::

   tox -m test -- tests/test_luminosity_function.py
