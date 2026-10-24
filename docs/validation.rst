.. _sec_validation:

Validation
==========

.. meta::
    :property=og:title: Validation
    :property=og:description: spey-roofit against a pure RooFit and RooStats workflow on three published analyses.
    :property=og:image: https://spey.readthedocs.io/en/main/_static/spey-logo.png
    :property=og:url: https://spey-roofit.readthedocs.io/en/main/validation.html

The plug-in is validated on one published analysis per backend, by comparing ``Spey`` with ``RooStats``.
Both sides compute the same two things on the same grid of signal strengths.

* **The Brazil band plots.** The exclusion confidence level, observed and expected with its :math:`\pm1\sigma` and
  :math:`\pm2\sigma` bands. ``RooStats`` uses ``HypoTestInverter`` with an ``AsymptoticCalculator`` and the one-sided
  test statistic :math:`\tilde{q}_\mu` (the configuration of ``StandardHypoTestInvDemo.C``); ``Spey`` uses its
  defaults, :math:`\tilde{q}_\mu` and expected bands from the post-fit Asimov data, which is what ``RooStats``
  uses as well.
* **The** :math:`\chi^2` **curve.** The profile likelihood ratio :math:`-2\Delta\log\mathcal{L}(\mu)`, from
  ``RooMinimizer`` with ``Minuit2`` on the negative log-likelihood that ``createNLL`` builds from the ``ModelConfig``,
  against ``Spey``'s own profile. This isolates the likelihood and the minimiser from the asymptotic formulae.

The reference runs in a separate process that imports neither ``Spey`` nor this plug-in, since ``ROOT`` hands back the
same ``RooWorkspace`` when a file that is already open is read again and the two sides would otherwise share their
parameters, ranges and caches.

In the left plots the lines and the filled bands are ``Spey`` and the markers are ``RooStats``: open circles for the
expected, squares for the observed and crosses for the :math:`\pm1\sigma` and :math:`\pm2\sigma` bands. The vertical axis
is labelled :math:`\mathrm{CL}_s` but is the exclusion confidence level, :math:`1-\mathrm{CL}_s` in the notation of the
other pages: it rises with the signal strength, and a signal is excluded where the curve is above the horizontal 95% line.
The dashed vertical line, where drawn, is the observed 95% CL limit. In the right plots the black line is ``RooFit`` with ``Minuit2``,
the red circles are ``Spey``, and the dotted lines mark :math:`-2\Delta\log\mathcal{L}=1` and the 95% CL value.

ATLAS-EXOT-2018-06: monojet
---------------------------

The analysis
~~~~~~~~~~~~

A search for new phenomena in events with an energetic jet and missing transverse momentum, with the full Run-2
dataset at 13 TeV (`arXiv:2102.10874 <https://arxiv.org/abs/2102.10874>`_,
`HEPData ins1847779 <https://www.hepdata.net/record/ins1847779>`_). The benchmark is the axial-vector
dark-matter model ``DMA_1_2000_gq0p25`` and the parameter of interest is the signal strength :math:`\mu_{\rm SIG}`.

The likelihood
~~~~~~~~~~~~~~

A binned ``HistFactory`` likelihood published as a ``RooFit`` workspace and read by the ``root.roofit`` backend. It has
78 regions of a single bin each, 13 signal regions in missing-transverse-momentum thresholds and five control regions
for each, and 150 nuisance parameters with their constraint terms. Each bin is a Poisson term, so the model is the
:ref:`RooFit likelihood <sec_roofit>` in its simplest form, only large: the ``RooFit`` negative log-likelihood is of
order :math:`2.5\times10^{7}`.

``Spey`` against ``RooStats``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :class: figure-row
   :widths: 50 50

   * - .. figure:: figs/ATLAS-EXOT-2018-06-cls.png
          :alt: Brazil band plots, ATLAS-EXOT-2018-06

          Observed and expected exclusion confidence level.
     - .. figure:: figs/ATLAS-EXOT-2018-06-chi2.png
          :alt: Profile likelihood, ATLAS-EXOT-2018-06

          Profile likelihood ratio; the lower panel is the difference of the two, in per cent.

The markers sit on the lines and bands: the observed and expected exclusion, and the limit, :math:`\mu^{\rm obs}_{95\%}\approx0.85`
below one, agree between the two workflows. The upper limits that the scan crosses match to about :math:`10^{-3}`, and
the largest difference in the exclusion level anywhere in the scan is about :math:`10^{-2}`. The profile
curves agree to :math:`2\times10^{-2}` in :math:`-2\Delta\log\mathcal{L}`, below 0.1% as the lower panel shows, with
a best fit of about 0.13. What is left is the precision of two minimisers on 150 parameters.

Two things were needed for this agreement. ``Spey``'s default ``scipy`` optimiser stops its fits of this model early and
reports an exclusion level of one at small signal strengths, so the comparison uses ``iminuit`` with a larger budget of
function calls (``{"minimizer": "minuit", "maxiter": 200_000}``). And the likelihood is evaluated with ``RooFit``'s
bin-by-bin offset, without which one evaluation is too imprecise for finite-difference gradients.

ATLAS-TOPQ-2018-24: :math:`t\bar{t}\gamma`
-------------------------------------------

The analysis
~~~~~~~~~~~~

A measurement of inclusive and differential :math:`t\bar{t}\gamma` cross-sections at 13 TeV
(`arXiv:2403.09452 <https://arxiv.org/abs/2403.09452>`_,
`HEPData ins2768921 <https://www.hepdata.net/record/ins2768921>`_). The shown likelihood is the dilepton channel with
the inclusive normalisation (``dilep_1NF``); the parameter of interest is the :math:`t\bar{t}\gamma` signal strength.
It is a measurement, not a search: the data prefer :math:`\mu\simeq1.1`, so the expected curves describe a world without
:math:`t\bar{t}\gamma` and the observed one the measured cross-section.

The likelihood
~~~~~~~~~~~~~~

A binned ``HistFactory`` likelihood published by HEPData as an HS3 document and read by the ``root.hs3`` backend:
one region of ten bins and 226 free parameters. The Monte Carlo statistics parameters have Poisson constraints, as in
``HistFactory`` (the ``pyhf`` twin of the same record uses Gaussian ones, which is why it is not used as the reference).
The :ref:`HS3 reader <sec_hs3>` compiles the document into a ``RooFit`` workspace, and the reference reads that same
workspace saved to a ``.root`` file, so that it is pure ``RooStats`` on the same model.

``Spey`` against ``RooStats``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :class: figure-row
   :widths: 50 50

   * - .. figure:: figs/ATLAS-TOPQ-2018-24-cls.png
          :alt: Brazil band plots, ATLAS-TOPQ-2018-24

          Observed and expected exclusion confidence level.
     - .. figure:: figs/ATLAS-TOPQ-2018-24-chi2.png
          :alt: Profile likelihood, ATLAS-TOPQ-2018-24

          Profile likelihood ratio, with the 95% CL interval of the signal strength.

The expected curve rises quickly because a world without :math:`t\bar{t}\gamma` is far from the measurement. The observed
curve levels off at 0.5: the tested signal strengths are below the fitted one, :math:`\hat\mu\simeq1.1`, where
the one-sided test statistic vanishes. The markers lie on the curves everywhere, and the profile likelihood on the right
has its minimum at :math:`\mu\simeq1.1` with a 95% CL interval of [0.97, 1.27], the same for both. Every upper limit —
observed, expected and the four band edges — agrees to the fourth digit, the exclusion level to better than
:math:`10^{-3}` and the profile curves to :math:`1.4\times10^{-3}` in :math:`-2\Delta\log\mathcal{L}`. The
expected curve stops where :math:`\mathrm{CL}_s` underflows to zero, in both tools.

CMS-EXO-24-011: dijet angular distributions
-------------------------------------------

The analysis
~~~~~~~~~~~~

A search for new physics in the dijet angular variable :math:`\chi=\exp|y_1-y_2|`, in eight bins of dijet mass
(`CMS-EXO-24-011 <https://repository.cern/records/wqfww-n3j76>`_,
`HEPData ins3136278 <https://www.hepdata.net/record/ins3136278>`_). The figure is the ADD model with
:math:`\Lambda=12` TeV. The signal is a pure *shape*: the card is a two-hypothesis fit whose mixing parameter :math:`x`
interpolates between QCD alone (:math:`x=0`) and QCD plus ADD (:math:`x=1`), and the scale is excluded when the
upper limit on :math:`x` falls below one.

The likelihood
~~~~~~~~~~~~~~

A binned likelihood given as a CMS Combine datacard and read by the ``root.cms-combine`` backend: eight channels of
twelve bins (96 bins) and 149 ``shape`` uncertainties, each a template morphing with a Gaussian constraint. ``combine``
builds the workspace itself and ``Spey`` reads it, with the physics model that gives :math:`x` the published meaning (see
:ref:`sec_combine`). The reference is the workspace of ``text2workspace.py --no-optimize-pdfs``, which writes
plain ``RooFit`` constraint terms instead of ``combine``'s optimised ones; with the default the asymptotic
calculator of ``RooStats`` returns wrong expected bands.

``Spey`` against ``RooStats``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :class: figure-row
   :widths: 50 50

   * - .. figure:: figs/CMS-EXO-24-011-cls.png
          :alt: Brazil band plots, CMS-EXO-24-011

          Observed and expected exclusion confidence level.
     - .. figure:: figs/CMS-EXO-24-011-chi2.png
          :alt: Profile likelihood, CMS-EXO-24-011

          Profile likelihood ratio, with the 95% CL interval of the mixing parameter.

The scan stays inside :math:`x\in[0,1]`, the range the physics model allows. The observed limit, :math:`x^{\rm obs}_{95\%}\approx0.45`,
is the same in both workflows, as is the median expected one (0.354 against 0.355); all six upper limits agree to
:math:`10^{-3}` and the exclusion level anywhere in the scan to :math:`1.2\times10^{-3}`. The profile likelihood has its
minimum near :math:`x\simeq0.13`, compatible with the Standard Model, and gives the interval [−0.19, 0.48] at 95% CL; the two curves
differ by up to :math:`8\times10^{-3}`. That remainder is a difference in the likelihood, not in the minimiser, and has the
size of the plug-in's one deliberate change: it widens the range of the auxiliary measurements, which ``combine`` truncates at
:math:`\pm4` and thereby cuts a few parts in a thousand off a constraint once its nuisance parameter is displaced.
