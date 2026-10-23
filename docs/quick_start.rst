.. _sec_quick_start:

Quick Start
===========

.. meta::
    :property=og:title: Quick Start
    :property=og:description: Installation and first steps.
    :property=og:image: https://spey.readthedocs.io/en/main/_static/spey-logo.png
    :property=og:url: https://spey-roofit.readthedocs.io/en/main/quick_start.html

Installation
------------

``ROOT`` and ``combine`` are only distributed through ``conda-forge``, so the environment is best created with
``conda``:

.. code-block:: bash

    conda env create -f environment.yml
    conda activate spey-roofit
    pip install -e .

``environment.yml`` is part of the `repository <https://github.com/SpeysideHEP/spey-roofit>`_. The ``pip`` step
installs ``spey`` and ``spey-hs3`` (the HS3 interpreter) as dependencies. ``Python >=3.10`` is required.
Once the package is installed ``Spey`` detects it automatically:

.. code-block:: python3

    >>> import spey
    >>> print(spey.AvailableBackends())
    >>> # [..., 'root.cms-combine', 'root.hs3', 'root.roofit', ...]

The published material used by the examples of this documentation belongs to the records that publish it and is
not part of the repository. Fetch it once with

.. code-block:: bash

    ./atlas-exot-2018-06/download.sh    # ATLAS monojet, RooFit workspaces
    ./atlas-topq-2018-24/download.sh    # ATLAS ttbar+gamma, HS3 documents
    ./cms-exo-24-011/download.sh        # CMS dijet angular, CMS Combine datacards

Workflow
--------

All three backends follow the same four steps, which is also how the documentation pages are organised:

1. **Inspect** the likelihood with the interpreter: ``channels``, ``bin_map``, ``observed_data``,
   ``expected_background_yields``, ``signal_samples``, ``summary()``.
2. **Remove** the signal that a signal-plus-background likelihood carries with ``strip_signal()`` and write the
   result to disk with ``save_patch(...)``.
3. **Add** your own signal yields, one number per bin of a region, with ``inject_signal`` or
   ``inject_signals``.
4. **Infer** with the backend: ``exclusion_confidence_level()``, ``poi_upper_limit()``, ``significance()``,
   ``maximize_likelihood()`` and everything else the :class:`~spey.StatisticalModel` provides.

The backends accept the signal directly, so steps 2 and 3 do not have to be done by hand:

.. code-block:: python3

    >>> import spey
    >>> model = spey.get_backend("root.roofit")(
    ...     workspace="bOnlyFit_postFit.root",
    ...     signal_yields={"SR_200_cuts": [1349.3]},   # {region: [one yield per bin]}
    ...     analysis="my-reinterpretation",
    ... )
    >>> model.exclusion_confidence_level()   # 1 - CLs
    >>> model.poi_upper_limit()              # 95% CL upper limit on the signal strength

Pick the page that matches the format the likelihood is published in:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Format
     - Page
   * - ``RooFit`` / ``HistFactory`` ``.root`` workspace
     - :ref:`sec_roofit`
   * - HS3 ``JSON`` document
     - :ref:`sec_hs3`
   * - CMS Combine datacard
     - :ref:`sec_combine`

.. note::

    The plug-in returns ``RooFit``'s own negative log-likelihood, with no rescaling. ``RooFit`` adds terms that
    depend on the data but not on the parameters, so the absolute value of ``model.likelihood()`` is **not**
    comparable with that of another backend for the same model. Likelihood ratios — fits, profiles, test
    statistics, :math:`CL_s` and limits — are.

.. note::

    ``Spey`` minimises with ``scipy`` unless told otherwise; ``spey.set_optimiser("minuit")`` selects ``iminuit``
    for every following call. For a model with many nuisance parameters the choice matters: on the 150-parameter
    ATLAS monojet model the fits of the Asimov dataset at small signal strengths do not converge with ``scipy``, and
    :math:`CL_s` is then reported as one there, so the ``RooFit`` and HS3 pages use ``iminuit``. The small models
    of the CMS Combine page are fine with the default. When in doubt, run both and compare.

.. note::

    The numbers on these pages are rounded and were obtained with ``spey`` 0.2.8; the last digits of a
    fit depend on the tolerance of the minimiser and can differ between platforms.

.. note::

    ``RooFit`` provides no analytic gradient, so ``Spey`` differentiates numerically; a likelihood evaluation
    costs one model evaluation per free parameter. A hypothesis test of a model with a few hundred nuisance
    parameters takes minutes, and a ``poi_upper_limit`` — a root search made of many such tests — can take
    much longer. The timings quoted on the following pages were measured on a laptop.
