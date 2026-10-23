.. _sec_combine:

CMS Combine Workspaces
======================

.. meta::
    :property=og:title: CMS Combine workspaces
    :property=og:description: Inspect, edit and use CMS Combine datacards within Spey.
    :property=og:image: https://spey.readthedocs.io/en/main/_static/spey-logo.png
    :property=og:url: https://spey-roofit.readthedocs.io/en/main/cms_combine.html

The workspace
-------------

CMS Combine is the statistical tool of the CMS collaboration. A model is described by a **datacard**, a plain text
file that lists the analysis channels (``bin``), the processes that contribute to each of them, their yields
(``rate``), the observed counts (``observation``) and the systematic uncertainties, one line per nuisance
parameter, with one entry per process:

.. code-block:: text

    imax 2  number of channels
    jmax *  number of backgrounds
    kmax *  number of nuisance parameters
    ------------
    bin         SR      CR
    observation 14      220
    ------------
    bin         SR      SR      CR      CR
    process     signal  bkg     signal  bkg
    process     0       1       0       1
    rate        6.0     10.0    1.0     200.0
    ------------
    lumi    lnN  1.025  1.025  1.025  1.025
    bkg_xs  lnN  -      1.15   -      1.05

The second ``process`` line numbers the processes: **signal processes have an index** :math:`\leq 0` and
backgrounds a positive one. A card without a ``shapes`` directive is a counting experiment, one bin per channel; with
``shapes`` lines, the yields come from histograms stored in ``ROOT`` files and a channel has as many bins as its
histograms. Systematic uncertainties are ``lnN`` (log-normal), ``shape`` (template morphing), ``gmN``, ``autoMCStats``,
``rateParam`` and others. ``combine`` turns the card into a ``RooFit`` workspace with ``text2workspace.py``; the
``root.cms-combine`` backend runs ``combine``'s own parser and model builder in process, so the model is built by the
same code as the published one, and then hands it to the ``RooFit`` machinery of this plug-in.

``combine`` is only distributed through ``conda-forge`` (``conda install -c conda-forge cms-combine``), see
:ref:`Quick Start <sec_quick_start>`.

**External resources**

* `Combine manual <https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/>`_ and in particular its
  `datacard format <https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/latest/part2/settinguptheanalysis/>`_
* `Combine repository <https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit>`_ and the paper
  :xref:`2404.06614`
* :xref:`HEPData`, where CMS publishes datacards; the shipped example is the CMS dijet angular search
  `CMS-EXO-24-011 <https://repository.cern/records/wqfww-n3j76>`_
  (`HEPData ins3136278 <https://www.hepdata.net/record/ins3136278>`_), see the last section

The examples below use the card above, which is small enough to be written out; it is passed to the interpreter as
text, a file path works the same way.

Inspecting a datacard
---------------------

:class:`~spey_roofit.DatacardInterpreter` reads the card with ``combine``'s parser, before anything is compiled.

.. code-block:: python3

    >>> from spey_roofit import DatacardInterpreter

    >>> interpreter = DatacardInterpreter("counting_card.txt")   # a path, or the text of the card
    >>> interpreter
    DatacardInterpreter(2 channels, 2 bins, 2 systematics)

    >>> interpreter.channels, interpreter.bin_map
    (['SR', 'CR'], {'SR': 1, 'CR': 1})
    >>> interpreter.processes, interpreter.signal_processes
    (['signal', 'bkg'], ['signal'])
    >>> interpreter.observed_data
    {'SR': [14.0], 'CR': [220.0]}
    >>> interpreter.expected_background_yields
    {'SR': [10.0], 'CR': [200.0]}
    >>> interpreter.expected_signal_yields
    {'SR': [6.0], 'CR': [1.0]}
    >>> interpreter.nuisance_parameters
    ['lumi', 'bkg_xs']

``summary()`` prints all of it, one block per channel:

.. code-block:: python3

    >>> interpreter.summary()
    CMS Combine datacard (counting)
      channels            : 2
      bins                : 2
      processes           : signal, bkg
      signal processes    : signal
      nuisance parameters : 2 (2 lnN)
      rate parameters     : 0
      autoMCStats         : none
      SR (1 bin)
          observed   : [14]
          background : [10]
          signal     : [6]
            - signal (signal): [6]
            - bkg: [10]
      CR (1 bin)
          observed   : [220]
          background : [200]
          signal     : [1]
            - signal (signal): [1]
            - bkg: [200]

Removing the signal and saving the card
---------------------------------------

``strip_signal()`` removes every process declared with an index :math:`\leq 0` from the card ``combine`` is given,
and ``save_patch`` writes the card to a file. The card is rewritten rather than the workspace: only the columns of
the table that belong to the signal are removed, and every other directive survives verbatim.

.. code-block:: python3

    >>> interpreter.strip_signal().save_patch("background_only.txt")
    'background_only.txt'

.. code-block:: text

    imax *
    jmax *
    kmax *
    ------------
    bin         SR      CR
    observation 14      220
    ------------
    bin SR CR
    process bkg bkg
    process 1 1
    rate 10.0 200.0
    ------------
    lumi lnN 1.025 1.025
    bkg_xs lnN 1.15 1.05

.. code-block:: python3

    >>> background = DatacardInterpreter("background_only.txt")
    >>> background.signal_processes, background.is_alive
    ([], False)

The ``imax``, ``jmax`` and ``kmax`` counts become wildcards, so that adding or removing a process does not make the card
disagree with its header. A card whose signal removal would leave a channel without any process raises a
``ValueError``. With a ``.root`` suffix (or ``fmt="root"``), ``save_patch`` writes the workspace ``combine`` builds
from the patched card instead; an HS3 document is not possible, because ``combine``'s own classes have no HS3
serialisation, and ``save_patch`` says so.

Adding new signal events
------------------------

``inject_signal`` attaches signal yields to a channel, one number per bin, and ``inject_signals`` does the same for
several channels at once. The signal enters as one extra process, ``spey_signal``, with index 0, scaled by the parameter of
interest and affected by no nuisance parameter. ``save_patch`` writes the card, with the new process, to a file:

.. code-block:: python3

    >>> background.inject_signal("SR", [3.0])          # or inject_signals({"SR": [3.0]})
    >>> background.signal_per_channel
    {'SR': [3.0]}
    >>> background.save_patch("reinterpretation.txt")

.. code-block:: text

    imax *
    jmax *
    kmax *
    ------------
    bin         SR      CR
    observation 14      220
    ------------
    bin SR CR SR
    process bkg bkg spey_signal
    process 1 1 0
    rate 10.0 200.0 3
    ------------
    lumi lnN 1.025 1.025 -
    bkg_xs lnN 1.15 1.05 -

For a card with ``shapes`` directives, a channel takes one yield per bin of its histograms. The interpreter then writes
the signal template to a ``ROOT`` file and adds a ``shapes`` line for it; ``save_patch`` copies that file next to the
saved card, and the templates of the original stay where they are, with absolute paths, so the saved card keeps working
wherever it is moved to. A channel that already has an injected signal, a wrong number of yields or an unknown channel
raises a ``ValueError``, and ``reset_signal()`` forgets every injection.

Using the backend
-----------------

The ``root.cms-combine`` backend takes the card — a path or its text — or an interpreter, and does the stripping and
the injection itself when asked to. The following build the same model:

.. code-block:: python3

    >>> import spey

    >>> wrapper = spey.get_backend("root.cms-combine")

    >>> # the saved reinterpretation
    >>> model = wrapper(datacard="reinterpretation.txt", analysis="counting-card")

    >>> # the saved background-only card and the signal
    >>> model = wrapper(
    ...     datacard="background_only.txt", signal_yields={"SR": [3.0]}, analysis="counting-card"
    ... )

    >>> # straight from the card that declares the signal
    >>> model = wrapper(
    ...     datacard="counting_card.txt",
    ...     signal_yields={"SR": [3.0]},
    ...     strip_existing_signal=True,
    ...     analysis="counting-card",
    ... )

**Arguments:**

* ``datacard``: path of the ``.txt`` card, its text, or a :class:`~spey_roofit.DatacardInterpreter`.
* ``signal_yields`` (optional): ``{channel: [yield per bin]}``.
* ``strip_existing_signal`` (optional): remove the processes the card declares as signal before injecting.
* ``mass`` (optional): value substituted for ``$MASS``, the ``-m`` option of ``text2workspace.py``.
* ``physics_model``, ``physics_options`` (optional): the ``-P`` and ``--PO`` options of ``text2workspace.py``;
  ``combine``'s default model, in which the parameter of interest :math:`r` scales every signal process, is used when
  omitted.
* ``directory`` (optional): directory the relative paths of the ``shapes`` lines are resolved against, by default
  the directory of the card.
* ``options`` (optional): further ``combine`` options, see :func:`~spey_roofit.combine_options`.
* ``analysis`` (optional): unique identifier for the analysis.

A card that already declares its signal is used as it is when ``signal_yields`` is omitted.

Exclusion limits and upper limits
---------------------------------

The values are for the reinterpretation above: the three events added to ``SR`` of the background-only card.

.. code-block:: python3

    >>> model = wrapper(datacard="reinterpretation.txt", analysis="counting-card")

    >>> model.maximize_likelihood()      # (mu_hat, -log L at the maximum)
    (1.0109068, 2.4336279)

    >>> model.exclusion_confidence_level()                                          # observed, 1 - CLs at mu = 1
    [0.3611939]
    >>> model.exclusion_confidence_level(expected=spey.ExpectationType.apriori)     # expected, -2 ... +2 sigma
    [0.8977181, 0.7880934, 0.5934981, 0.3255211, 0.1005510]

    >>> model.poi_upper_limit()                                                     # observed, 95% CL
    3.6672538
    >>> model.poi_upper_limit(expected=spey.ExpectationType.apriori)                # expected, 95% CL
    2.6022773

    >>> model.exclusion_confidence_level(poi_test=model.poi_upper_limit())          # 1 - CLs at the limit
    [0.95]

The signal as it stands (:math:`\mu=1`, three events in a region with ten expected) is excluded at 36%, and it takes a
signal strength of 3.67 to exclude it at 95% CL; the median expected limit, 2.60, is tighter than the observed one
because the data show an excess of four events in ``SR``. For the original card, with its own signal of six and one
events, :math:`1-CL_s=0.6818` and the upper limit on :math:`\mu` is 1.869.

The ``lnN`` uncertainties of this card are the log-normal constraints that ``pyhf`` implements as ``normsys``
modifiers, so the same model can be built in ``spey-pyhf`` as a check. It gives :math:`1-CL_s=0.6818` and an upper
limit of 1.869 for the original card, and 0.3612 and 3.667 for the reinterpretation, in agreement to four significant digits.

Published two-hypothesis cards
------------------------------

Some published cards do not follow the signal-plus-background convention above. CMS-EXO-24-011 declares two signal
processes per channel, ``QCDADD12000`` (QCD plus the ADD signal) and ``QCDADD12000_ALT`` (QCD alone), and its
parameter of interest :math:`x` interpolates between them: :math:`x=0` is the Standard Model and :math:`x=1` the ADD
hypothesis. Stripping the signal of such a card removes both hypotheses, so it is not a reinterpretation tool; the card is
used as it is, with the physics model that gives :math:`x` its meaning.

.. code-block:: python3

    >>> from spey_roofit.combine import SWAPPED_TWO_HYPOTHESIS_MODEL

    >>> card = "cms-exo-24-011/datacards/input/chi_datacard13TeV3_ADD12000_run2.txt"
    >>> interpreter = DatacardInterpreter(card)
    >>> interpreter.bin_map                       # eight mass bins, twelve angular bins each
    {'bin0': 12, 'bin1': 12, 'bin2': 12, 'bin3': 12, 'bin4': 12, 'bin5': 12, 'bin6': 12, 'bin7': 12}
    >>> interpreter.signal_processes
    ['QCDADD12000', 'QCDADD12000_ALT']

    >>> model = wrapper(
    ...     datacard=card,
    ...     physics_model=SWAPPED_TWO_HYPOTHESIS_MODEL,   # the -P option of text2workspace.py
    ...     analysis="cms-exo-24-011",
    ... )
    >>> model.maximize_likelihood()               # x_hat = 0.13, close to the Standard Model at x = 0
    (0.1319372, 175.7730939)
    >>> model.exclusion_confidence_level(poi_test=1.0)
    [0.9999977]
    >>> model.exclusion_confidence_level(poi_test=1.0, expected=spey.ExpectationType.apriori)
    [1.0, 1.0, 0.9999998, 0.9999822, 0.9992295]

The ADD hypothesis with a scale of 12 TeV is excluded with :math:`1-CL_s>0.99999`. Fetch the cards with ``./cms-exo-24-011/download.sh``. The 149 shape uncertainties are
compiled by ``combine`` in about four seconds, and ``exclusion_confidence_level`` takes about twenty seconds.
