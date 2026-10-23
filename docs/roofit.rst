.. _sec_roofit:

RooFit / RooStats Workspaces
============================

.. meta::
    :property=og:title: RooFit / RooStats workspaces
    :property=og:description: Inspect, edit and use RooFit workspaces within Spey.
    :property=og:image: https://spey.readthedocs.io/en/main/_static/spey-logo.png
    :property=og:url: https://spey-roofit.readthedocs.io/en/main/roofit.html

The workspace
-------------

A ``RooWorkspace`` is the container ``RooFit`` uses to store a statistical model: the probability density
function, the variables it depends on, the datasets, and a ``RooStats::ModelConfig`` that says which variables are
the parameters of interest, the nuisance parameters and the global observables (the auxiliary measurements of the
constraint terms). A workspace is saved in a ``.root`` file, and this is how many ATLAS analyses publish their
likelihoods on `HEPData <https://www.hepdata.net>`_.

Binned analyses build their workspace with ``HistFactory``. The model is a ``RooSimultaneous`` over the analysis
regions (channels); each channel is a product of a ``RooRealSumPdf``, which sums the samples
(signal and backgrounds) bin by bin, and of the constraint terms of its nuisance parameters:

.. math::

    \mathcal{L}(\mu,\theta) = \prod_{c\in{\rm channels}}\prod_{i\in{\rm bins}}
        {\rm Pois}\left(n_{ci}\,\middle\vert\,\mu\,s_{ci}(\theta)+b_{ci}(\theta)\right)
        \cdot\prod_{k} p_k(a_k\vert\theta_k)\ .

The ``root.roofit`` backend reads this structure and exposes it as a ``Spey`` likelihood.

**External resources**

* `RooFit manual <https://root.cern/manual/roofit/>`_ and `RooStats documentation
  <https://root.cern/doc/master/group__Roostats.html>`_; the papers :xref:`physics/0306116` and :xref:`1009.1003`
* `HistFactory <https://root.cern/doc/master/group__HistFactory.html>`_: the tool that builds the workspaces
  (`CERN-OPEN-2012-016 <https://cds.cern.ch/record/1456844>`_)
* :xref:`HEPData`, where the workspaces of the analyses used below are published

The examples use the ATLAS monojet search (`arXiv:2102.10874 <https://arxiv.org/abs/2102.10874>`_,
`HEPData ins1847779 <https://www.hepdata.net/record/ins1847779>`_). The workspace has 78 regions of a single bin each,
150 nuisance parameters, and a signal-plus-background version that contains the ``DMA_1_2000_gq0p25`` benchmark.
Fetch the files with ``./atlas-exot-2018-06/download.sh``.

Inspecting a workspace
----------------------

:class:`~spey_roofit.WorkspaceInterpreter` reads the region structure of a workspace without requiring any
knowledge of ``RooFit``. It accepts a path or a ``ROOT.RooWorkspace`` and never modifies it.

.. code-block:: python3

    >>> from spey_roofit import WorkspaceInterpreter

    >>> interpreter = WorkspaceInterpreter("atlas-exot-2018-06/sPlusBFit_postFit.root")
    >>> interpreter
    WorkspaceInterpreter(channels=78, bins=78, poi=['mu_SIG'], injected=[])

    >>> interpreter.channels[:3]                  # regions, in the order of the data vector
    ['SR_200_cuts', 'CR1mu0b_200_cuts', 'CR1e0b_200_cuts']
    >>> interpreter.bin_map["SR_200_cuts"]        # number of bins of a region
    1
    >>> interpreter.guess_signal_regions()[:3]    # from the names of the regions
    ['SR_200_cuts', 'SR_250_cuts', 'SR_300_cuts']
    >>> interpreter.observed_data["SR_200_cuts"]
    [1791624.0]
    >>> interpreter.expected_background_yields["SR_200_cuts"]
    [1790292.0657108685]
    >>> interpreter.signal_samples["SR_200_cuts"] # samples scaled by the parameter of interest
    ['DMA_1_2000_gq0p25']
    >>> interpreter.poi_names
    ['mu_SIG']

``summary()`` prints all of it at once, with one block per region:

.. code-block:: python3

    >>> interpreter.summary(max_channels=1)
    RooFit workspace 'w'
      channels            : 78
      bins                : 78
      parameters of interest : mu_SIG
      nuisance parameters : 150
      global observables  : 179
      [SR] SR_200_cuts (1 bin)
          observed   : [1.792e+06]
          background : [1.79e+06]
          signal     : [171.4]
            - NCB_200: [7334]
            - multijet_200: [2.57e+04]
            ...
            - DMA_1_2000_gq0p25 (signal): [171.4]
      ... 77 more channel(s)

.. note::

    Yields are evaluated at the parameter values stored in the workspace. In this post-fit workspace
    :math:`\mu_{\rm SIG}=0.1268`, so ``expected_signal_yields`` is the signal at that value and not at
    :math:`\mu=1`; the signal at :math:`\mu=1` is the yield divided by the stored value, 171.4 / 0.1268 = 1351.3 for
    ``SR_200_cuts``.

Removing the signal and saving the workspace
--------------------------------------------

``strip_signal()`` drops every sample that is scaled by a parameter of interest, which turns a signal-plus-background
workspace into a background-only one. ``save_patch`` writes the result to a new ``.root`` file; the interpreter
never touches the original.

.. code-block:: python3

    >>> mu = interpreter.workspace.var("mu_SIG").getVal()   # 0.1268, the value stored in the file
    >>> signal_at_mu_1 = {      # keep the published signal, for the next section
    ...     region: [y / mu for y in yields]
    ...     for region, yields in interpreter.expected_signal_yields.items()
    ...     if any(yields)
    ... }

    >>> interpreter.strip_signal().save_patch("background_only.root")
    'background_only.root'

    >>> background = WorkspaceInterpreter("background_only.root")
    >>> background.signal_samples["SR_200_cuts"]
    []
    >>> background.poi_names, background.is_alive
    ([], False)

The parameter of interest is gone with the signal; the nuisance parameters, the constraint terms, the data and the
post-fit values of the signal-plus-background fit are kept. A background-only workspace published on ``HEPData``
(``bOnlyFit_postFit.root`` for this analysis) is, when available, the better starting point, because its nuisance
parameters were fitted without a signal.

Adding new signal events
------------------------

``inject_signal`` attaches signal yields to a region, one number per bin, and ``inject_signals`` does the same for
several regions at once. The signal is added to the sum over the samples of the region as :math:`\mu\,s_i`; it is
scaled by the parameter of interest, which is created as ``mu`` when the workspace declares none, and is not
affected by any nuisance parameter. Here the signal removed above is put back, from the yields at :math:`\mu=1`:

.. code-block:: python3

    >>> background.inject_signals(signal_at_mu_1)        # {region: [one yield per bin]}
    >>> background.signal_per_channel["SR_200_cuts"]
    [1351.294638407221]
    >>> background.is_alive
    True

A single region is ``background.inject_signal("SR_200_cuts", [1351.3])``. Injecting into a region that already has
an injected signal, giving the wrong number of yields or an unknown region raises a ``ValueError``;
``reset_signal()`` forgets every injection. The patched workspace is ``background.patch``, and it can be saved
with ``background.save_patch("reinterpretation.root")``.

.. note::

    The injected signal has no systematic uncertainties, whereas the published one carries its own, so re-injecting the
    published yields reproduces the published model closely but not exactly. The best fit signal strength is 0.1308,
    against 0.1309 for the published workspace.

Using the backend
-----------------

The ``root.roofit`` backend takes the workspace — a path, a ``RooWorkspace`` or an interpreter — and does steps
2 and 3 itself when asked to:

.. code-block:: python3

    >>> import spey

    >>> model = spey.get_backend("root.roofit")(
    ...     workspace=background,        # or "background_only.root" with signal_yields=signal_at_mu_1
    ...     analysis="atlas-exot-2018-06",
    ... )

    >>> # equivalently, straight from the published signal-plus-background file
    >>> model = spey.get_backend("root.roofit")(
    ...     workspace="atlas-exot-2018-06/sPlusBFit_postFit.root",
    ...     strip_existing_signal=True,
    ...     signal_yields=signal_at_mu_1,
    ...     analysis="atlas-exot-2018-06",
    ... )

**Arguments:**

* ``workspace``: path of the ``.root`` file, a ``ROOT.RooWorkspace`` or a
  :class:`~spey_roofit.WorkspaceInterpreter`.
* ``signal_yields`` (optional): ``{region: [yield per bin]}``, see above.
* ``strip_existing_signal`` (optional): remove the signal the workspace already carries before injecting.
* ``poi_name`` (optional): parameter of interest, by default the one the model configuration declares.
* ``model_config_name``, ``data_name``, ``workspace_name`` (optional): names of the ``ModelConfig``, of the observed
  dataset and of the workspace inside the file, when the defaults (``"ModelConfig"``, ``"obsData"``, the first
  workspace of the file) do not apply.
* ``analysis`` (optional): unique identifier for the analysis.

A workspace that already contains a signal, such as ``sPlusBFit_postFit.root``, is used directly without
``signal_yields``.

Exclusion limits and upper limits
---------------------------------

All the :class:`~spey.StatisticalModel` methods are available; the ones below are what a reinterpretation needs.
``exclusion_confidence_level`` returns :math:`1-CL_s` at :math:`\mu=1`, and ``poi_upper_limit`` the :math:`\mu`
excluded at 95% CL. The values are for the model built above, the published signal re-injected into the
background-only workspace. This model has 150 nuisance parameters; as explained in :ref:`sec_quick_start`,
``iminuit`` is the minimiser to use for it.

.. code-block:: python3

    >>> spey.set_optimiser("minuit")

    >>> model.maximize_likelihood()      # (mu_hat, -log L at the maximum)
    (0.1308, -164.82)

    >>> model.exclusion_confidence_level()                                          # observed, 1 - CLs at mu = 1
    [0.98467]
    >>> model.exclusion_confidence_level(expected=spey.ExpectationType.apriori)     # expected, -2 ... +2 sigma
    [0.99995, 0.99936, 0.99336, 0.94863, 0.75701]

    >>> model.poi_upper_limit()                                                     # observed, 95% CL
    0.82070
    >>> model.poi_upper_limit(expected=spey.ExpectationType.apriori)                # expected, 95% CL
    0.72202

The signal as it stands, :math:`\mu=1`, is excluded at 95% CL: :math:`1-CL_s=0.985` is above 0.95, which is the same
statement as an upper limit of 0.821 on :math:`\mu` that is below one. For comparison, the published workspace, with
the signal systematic uncertainties it carries, gives :math:`1-CL_s=0.982`. ``-log L`` is ``RooFit``'s own value,
whose offset is not comparable with that of another backend, see :ref:`sec_quick_start`.

.. note::

    The model has 150 free nuisance parameters and no analytic gradient, so these calls are slow: on an Apple M-series
    laptop, with ``iminuit``, the best fit takes about 25 seconds, ``exclusion_confidence_level`` two to three minutes
    and ``poi_upper_limit`` about twenty minutes, because it is a root search made of many hypothesis tests.
    Loading a workspace and saving a patched one take a few seconds. A model with a handful of nuisance parameters
    takes less than a second for all of them.
