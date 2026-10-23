.. _sec_hs3:

HS3 Workspaces
==============

.. meta::
    :property=og:title: HS3 workspaces
    :property=og:description: Inspect, edit and use HS3 JSON likelihoods within Spey.
    :property=og:image: https://spey.readthedocs.io/en/main/_static/spey-logo.png
    :property=og:url: https://spey-roofit.readthedocs.io/en/main/hs3.html

The workspace
-------------

The *HEP Statistics Serialization Standard* (HS3) describes a statistical model as a plain ``JSON`` document that
is independent of the software evaluating it. A document is made of a few top-level lists that refer to each
other by name:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Key
     - Contents
   * - ``distributions``
     - the model. A ``histfactory_dist`` is one analysis region: its ``axes`` give the binning and its ``samples``
       the per-bin ``contents`` and ``errors`` of every process, each with its ``modifiers`` (``normfactor``,
       ``normsys``, ``histosys``, ``staterror``, ...)
   * - ``data``
     - observed (and Asimov) counts, one ``binned`` entry per region
   * - ``likelihoods``
     - which distributions are paired with which datasets
   * - ``analyses``
     - a likelihood together with its ``parameters_of_interest`` and ``domains``
   * - ``parameter_points``, ``domains``
     - default parameter values and their ranges

The ``root.hs3`` backend compiles the document into a ``RooFit`` workspace with ``ROOT``'s own reader and then
evaluates it exactly as the :ref:`RooFit backend <sec_roofit>` does, so the same analysis published in either format
gives the same numbers. The interpreter works on the ``JSON`` itself, without ``ROOT``; it is provided by
`spey-hs3 <https://github.com/SpeysideHEP/spey-hs3>`_, which is installed with this plug-in.

**External resources**

* `HS3 specification <https://hep-statistics-serialization-standard.github.io/>`_ and its
  `repository <https://github.com/hep-statistics-serialization-standard/hep-statistics-serialization-standard>`_
* `RooJSONFactoryWSTool <https://root.cern/doc/master/classRooJSONFactoryWSTool.html>`_: the ``ROOT`` reader and
  writer of HS3
* :xref:`HEPData`, where HS3 likelihoods are published

The examples use the ATLAS monojet search (`arXiv:2102.10874 <https://arxiv.org/abs/2102.10874>`_,
`HEPData ins1847779 <https://www.hepdata.net/record/ins1847779>`_) — the same likelihood as on the
:ref:`RooFit page <sec_roofit>`, which makes the two backends easy to compare. HEPData publishes it as a ``RooFit``
workspace, so the first step is to write it as an HS3 document with ``ROOT``; analyses that publish HS3 directly, such as
the ATLAS :math:`t\bar{t}\gamma` measurement
(`HEPData ins2768921 <https://www.hepdata.net/record/ins2768921>`_,
``./atlas-topq-2018-24/download.sh``), need no conversion.

.. code-block:: python3

    >>> import ROOT
    >>> from spey_roofit import load_document, load_workspace, repair_document

    >>> workspace = load_workspace("atlas-exot-2018-06/sPlusBFit_postFit.root")
    >>> ROOT.RooJSONFactoryWSTool(workspace).exportJSON("monojet_HS3.json")
    True

    >>> document, repaired = repair_document(load_document("monojet_HS3.json"))
    >>> len(repaired)
    102

.. note::

    ``ROOT`` 6.36 writes the Gaussian constraints of a ``HistFactory`` model without the ``x`` key that names the
    parameter they constrain, which the standard requires. ``repair_document`` restores it from the naming
    convention (here for 102 constraints); the backend does this itself, but a document that is handed to the
    interpreter first should be repaired first, otherwise stripping the signal prunes the parameters that only the
    signal uses while their constraints are left behind.

Inspecting a document
---------------------

The interpreter is that of ``spey-hs3`` and works on the ``JSON`` (a ``dict``) alone; ``ROOT`` is not involved.

.. code-block:: python3

    >>> from spey_hs3.helper_functions import WorkspaceInterpreter

    >>> interpreter = WorkspaceInterpreter(document)

    >>> len(interpreter.bin_map), interpreter.bin_map["model_SR_200_cuts"]   # regions, bins of a region
    (78, 1)
    >>> interpreter.guess_signal_regions()[:3]                               # from the names of the regions
    ['model_SR_1000_cuts', 'model_SR_1100_cuts', 'model_SR_1200_cuts']
    >>> interpreter.observed_data["model_SR_200_cuts"]
    [1791624.0]
    >>> interpreter.signal_samples["model_SR_200_cuts"]                      # samples the parameter of interest scales
    ['DMA_1_2000_gq0p25_SR_200_cuts_overallSyst_x_StatUncert']
    >>> interpreter.poi_parameters
    ['mu_SIG']
    >>> interpreter.expected_signal_yields["model_SR_200_cuts"]              # nominal template contents
    [1349.1658935546875]

``interpreter.summary()`` prints the analyses, regions, samples and modifiers.

.. note::

    The interpreter reports the nominal contents of the templates: the sum of the samples before any
    normalisation factor or systematic variation is applied. The background of ``model_SR_200_cuts`` is
    1.57e6 events by this measure, whereas the ``RooFit`` interpreter, which evaluates the model at its post-fit
    parameters, reports 1.79e6.

Removing the signal and saving the document
-------------------------------------------

``strip_signal()`` queues the removal of every signal sample and returns what it will remove; the parameter of
interest that becomes orphaned is removed from the document as well (``cleanup_parameters=True``, the default).
``save_patch`` applies what has been queued and writes a new ``JSON`` file; the original is not modified. First keep the
published signal, to be injected again below:

.. code-block:: python3

    >>> signal = {
    ...     region: yields
    ...     for region, yields in interpreter.expected_signal_yields.items()
    ...     if any(yields)
    ... }
    >>> len(signal)
    13

    >>> interpreter.strip_signal()["model_SR_200_cuts"]
    ['DMA_1_2000_gq0p25_SR_200_cuts_overallSyst_x_StatUncert']
    >>> interpreter.save_patch("background_only.json")

    >>> background = WorkspaceInterpreter(load_document("background_only.json"))
    >>> background.signal_samples, background.poi_parameters
    ({}, [])

``interpreter.patch`` returns the same document as a ``dict`` instead of writing it. The ``RooFit`` interpreter
writes HS3 as well, ``WorkspaceInterpreter(...).save_patch("workspace.json")``, which is how the document above is
obtained from a ``.root`` workspace without calling ``ROOT`` yourself.

Adding new signal events
------------------------

``inject_signals`` adds a new sample to a region, as ``{region: {sample name: yields}}`` with one yield per bin. The
sample is scaled by the parameter of interest ``poi_name`` and carries no systematic uncertainty; per-bin
statistical errors can be given with ``{"contents": [...], "errors": [...]}`` instead of the list. Here the removed
signal is put back:

.. code-block:: python3

    >>> background.inject_signals(
    ...     {region: {"my_signal": yields} for region, yields in signal.items()},
    ...     poi_name="mu",
    ... )
    >>> background.save_patch("reinterpretation.json")

    >>> WorkspaceInterpreter(load_document("reinterpretation.json")).signal_samples["model_SR_200_cuts"]
    ['my_signal']

A single region is ``background.inject_signal("model_SR_200_cuts", "my_signal", yields, poi_name="mu")``. An unknown
region, a wrong number of bins or a sample name that already exists raises a ``ValueError``, and ``reset_signal()``
clears every queued change.

Using the backend
-----------------

The ``root.hs3`` backend takes the document — a path, a ``JSON`` string, a ``dict`` or an interpreter — and
performs the stripping and the injection itself when asked to. The following build equivalent models (their
:math:`1-CL_s` agree to :math:`10^{-5}`; the last one keeps the name ``mu_SIG`` for the parameter of interest):

.. code-block:: python3

    >>> import spey

    >>> wrapper = spey.get_backend("root.hs3")

    >>> # the saved reinterpretation; an interpreter with queued changes works as well
    >>> model = wrapper(workspace="reinterpretation.json", analysis="atlas-exot-2018-06")

    >>> # the saved background-only document and the signal
    >>> model = wrapper(
    ...     workspace="background_only.json", signal_yields=signal, analysis="atlas-exot-2018-06"
    ... )

    >>> # straight from the published signal-plus-background document
    >>> model = wrapper(
    ...     workspace="monojet_HS3.json",
    ...     signal_yields=signal,
    ...     strip_existing_signal=True,
    ...     analysis="atlas-exot-2018-06",
    ... )

**Arguments:**

* ``workspace``: path, ``JSON`` text, ``dict`` or ``spey_hs3`` ``WorkspaceInterpreter``.
* ``signal_yields`` (optional): ``{region: [yield per bin]}``, or ``{region: {sample: yields}}`` to name the sample.
* ``strip_existing_signal`` (optional): remove the signal the document already carries before injecting.
* ``poi_name`` (optional): parameter of interest, by default the one the document declares, and ``"mu"`` when it
  declares none.
* ``poi_bounds`` (optional): range given to a parameter of interest the document does not declare, ``(-10, 10)`` by
  default.
* ``model_config_name``, ``data_name`` (optional): when the automatic choice of the ``ModelConfig`` and of the
  observed dataset (the one that is not Asimov) does not apply.
* ``repair`` (optional): restore the ``x`` key of constraints that lack it, ``True`` by default.
* ``analysis`` (optional): unique identifier for the analysis.

A document that already carries a signal is used as it is when ``signal_yields`` is omitted.

Exclusion limits and upper limits
---------------------------------

The values are for the first of the models above, ``reinterpretation.json``. This model has 150 nuisance parameters; as
explained in :ref:`sec_quick_start`, ``iminuit`` is the minimiser to use for it.

.. code-block:: python3

    >>> spey.set_optimiser("minuit")
    >>> model = wrapper(workspace="reinterpretation.json", analysis="atlas-exot-2018-06")

    >>> model.maximize_likelihood()      # (mu_hat, -log L at the maximum)
    (0.1254, 581.57)

    >>> model.exclusion_confidence_level()                                          # observed, 1 - CLs at mu = 1
    [0.98541]
    >>> model.exclusion_confidence_level(expected=spey.ExpectationType.apriori)     # expected, -2 ... +2 sigma
    [0.99995, 0.99938, 0.99351, 0.94945, 0.75941]

    >>> model.poi_upper_limit()                                                     # observed, 95% CL
    0.81486
    >>> model.poi_upper_limit(expected=spey.ExpectationType.apriori)                # expected, 95% CL
    0.72044

    >>> model.exclusion_confidence_level(poi_test=model.poi_upper_limit())          # 1 - CLs at the limit
    [0.94997]

The signal as it stands, :math:`\mu=1`, is excluded at 95% CL: :math:`1-CL_s=0.985` is above 0.95, which is the same
statement as an upper limit of 0.815 on :math:`\mu` that is below one. The upper limit is the value of :math:`\mu` at
which :math:`1-CL_s=0.95`; the last call returns 0.94997 rather than 0.95 because of the tolerance of the minimiser.

.. note::

    Compiling the document takes about three seconds. On an Apple M-series laptop, with ``iminuit``, the best fit takes
    five seconds, ``exclusion_confidence_level`` about half a minute and ``poi_upper_limit`` about
    four and a half minutes for this 150-parameter model. The model is the one of the ``RooFit`` page; the numbers differ
    slightly from the ones there because the signal injected here is the nominal template of the HS3 document (1349.2
    events in ``model_SR_200_cuts``), whereas the ``RooFit`` interpreter evaluates it at the post-fit nuisance
    parameters (1351.3 events).
