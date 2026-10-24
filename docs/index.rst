.. spey-roofit documentation master file

Welcome to the documentation of the ``RooFit`` plug-ins for Spey
================================================================

.. meta::
    :property=og:title: spey-roofit plug-in
    :property=og:image: https://spey.readthedocs.io/en/main/_static/spey-logo.png
    :property=og:url: https://spey-roofit.readthedocs.io/en/main/

.. image:: https://img.shields.io/github/issues/SpeysideHEP/spey-roofit?style=plastic&logo=github
   :alt: GitHub issues
   :target: https://github.com/SpeysideHEP/spey-roofit/issues

.. image:: https://img.shields.io/static/v1?style=plastic&label=DOI&message=10.21468/SciPostPhys.16.1.032&color=blue
    :alt: SciPost Phys DOI
    :target: https://scipost.org/10.21468/SciPostPhys.16.1.032

``spey-roofit`` turns a binned ``RooFit`` model into a `Spey <https://spey.readthedocs.io>`_
:class:`~spey.StatisticalModel`, so that exclusion limits, significances, upper limits and model
combination are computed by ``Spey`` exactly as they are for any other likelihood prescription. It provides
three backends, one for each way a published likelihood is distributed:

.. list-table::
   :header-rows: 1
   :widths: 22 28 50

   * - Backend
     - Input
     - Documentation
   * - ``root.roofit``
     - a ``RooFit`` workspace
     - :ref:`sec_roofit`
   * - ``root.hs3``
     - an HS3 ``JSON`` document
     - :ref:`sec_hs3`
   * - ``root.cms-combine``
     - a CMS Combine datacard
     - :ref:`sec_combine`

Every backend comes with an **interpreter**: a small object that tells you what the analysis contains (regions,
bins, samples, yields), removes the signal a published likelihood already carries and attaches your own signal
yields to it. The likelihood is never reimplemented; ``ROOT`` evaluates the ``RooFit`` and HS3 models and
``combine`` builds the datacard ones.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   Spey Documentation <https://spey.readthedocs.io/>
   quick_start
   roofit
   hs3
   cms_combine
   validation
   api
   citations

* :ref:`genindex`
