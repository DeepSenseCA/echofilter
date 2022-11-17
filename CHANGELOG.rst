Changelog
=========

All notable changes to echofilter will be documented here.

The format is based on `Keep a Changelog`_, and this project adheres to
`Semantic Versioning`_.

.. _Keep a Changelog: https://keepachangelog.com/en/1.0.0/
.. _Semantic Versioning: https://semver.org/spec/v2.0.0.html

Categories for changes are: Added, Changed, Deprecated, Removed, Fixed,
Security.


Version `1.1.1 <https://github.com/DeepSenseCA/echofilter/tree/1.1.1>`__
------------------------------------------------------------------------

Release date: 2022-11-16.
`Full commit changelog <https://github.com/DeepSenseCA/echofilter/compare/1.1.0...1.1.1>`__.


.. _v1.1.1 Fixed:

Fixed
~~~~~~~

.. _v1.1.1 Fixed Inference:

Inference
^^^^^^^^^

-   EVL final value pad was for a timestamp in between the preceding two, not extending forward in time by half a timepoint.
    (`#300 <https://github.com/DeepSenseCA/echofilter/pull/300>`__)

.. _v1.1.1 Fixed Metadata:

Metadata
^^^^^^^^

-   Declare ``python_requires<3.11`` requirement.
    (`#302 <https://github.com/DeepSenseCA/echofilter/pull/302>`__)
-   Declare ``torch<1.12.0`` requirement.
    (`#302 <https://github.com/DeepSenseCA/echofilter/pull/302>`__)


Version `1.1.0 <https://github.com/DeepSenseCA/echofilter/tree/1.1.0>`__
------------------------------------------------------------------------

Release date: 2022-11-12.
`Full commit changelog <https://github.com/DeepSenseCA/echofilter/compare/1.0.2...1.1.0>`__.


.. _v1.1.0 Changed:

Changed
~~~~~~~

.. _v1.1.0 Changed Inference:

Inference
^^^^^^^^^

-   Disable logit smoothing by default. The previous behaviour can be restored
    by setting ``--logit-smoothing-sigma=1`` at the CLI.
    (`#293 <https://github.com/DeepSenseCA/echofilter/pull/293>`__)


.. _v1.1.0 Fixed:

Fixed
~~~~~

.. _v1.1.0 Fixed Inference:

Inference
^^^^^^^^^

-   Fix bug where joined segments of data would have their first ping dropped.
    (`#272 <https://github.com/DeepSenseCA/echofilter/pull/272>`__)

.. _v1.1.0 Fixed Training:

Training
^^^^^^^^

-   Make the number of channels in the first block respect the ``initial_channels`` argument.
    (`#271 <https://github.com/DeepSenseCA/echofilter/pull/271>`__)

.. _v1.1.0 Fixed Miscellaneous:

Miscellaneous
^^^^^^^^^^^^^

-   Fix unseen internal bugs, including in ``generate_shards``.
    (`#283 <https://github.com/DeepSenseCA/echofilter/pull/283>`__)


.. _v1.1.0 Added:

Added
~~~~~

.. _v1.1.0 Added Inference:

Inference
^^^^^^^^^

-   Add support for using a config file to provide arguments to the CLI.
    (`#294 <https://github.com/DeepSenseCA/echofilter/pull/294>`__)
-   Add ``--continue-on-error`` argument to inference routine, which will
    capture an error when processing an individual file and continue running
    the rest.
    (`#245 <https://github.com/DeepSenseCA/echofilter/pull/245>`__)
-   Break up large files into more manageable chunks of at most 1280 pings,
    to reduce out-of-memory errors.
    (`#245 <https://github.com/DeepSenseCA/echofilter/pull/245>`__)
-   Reduce GPU memory consumption during inference by moving outputs to CPU
    memory sooner.
    (`#245 <https://github.com/DeepSenseCA/echofilter/pull/245>`__)
-   Fill in missing values in the input file through 2d linear interpolation.
    (`#246 <https://github.com/DeepSenseCA/echofilter/pull/246>`__)
-   Pad Sv data in timestamp dimension during inference to ensure the data is fully within the network's effective receptive field.
    (`#277 <https://github.com/DeepSenseCA/echofilter/pull/277>`__)
-   Add ``--prenorm-nan-value`` and ``--postnorm-nan-value`` options to control what value NaN values in the input are mapped to.
    (`#274 <https://github.com/DeepSenseCA/echofilter/pull/274>`__)
-   Add support for providing a single path as a string to the run_inference API.
    (Note that the CLI already supported this and so is unchanged).
    (`#288 <https://github.com/DeepSenseCA/echofilter/pull/288>`__)
-   Add more verbosity messages.
    (`#276 <https://github.com/DeepSenseCA/echofilter/pull/276>`__,
    `#278 <https://github.com/DeepSenseCA/echofilter/pull/278>`__,
    `#292 <https://github.com/DeepSenseCA/echofilter/pull/292>`__)

.. _v1.1.0 Added ev2csv:

ev2csv
^^^^^^

-   Add ``--keep-thresholds`` option which allow for exporting Sv data with thresholds and exclusions enabled (set as they currently are in the EV file).
    The default behaviour is still to export raw Sv data (disabling all thresholds).
    The default file name for the CSV file depends on whether the export is of raw or thresholded data.
    (`#275 <https://github.com/DeepSenseCA/echofilter/pull/275>`__)
-   Add ``--keep-ext`` argument to ev2csv, which allows the existing
    extension on the input path to be kept preceding the new file extension.
    (`#242 <https://github.com/DeepSenseCA/echofilter/pull/242>`__)

.. _v1.1.0 Added Tests:

Tests
^^^^^

-   Add tests which check that inference commands run, whether checking their outputs.
    (`#289 <https://github.com/DeepSenseCA/echofilter/pull/289>`__)


.. _v1.1.0 Added Internal:

Internal
^^^^^^^^

-   Add EVR reader ``echofilter.raw.loader.evr_reader``.
    (`#280 <https://github.com/DeepSenseCA/echofilter/pull/280>`__)


Version `1.0.3 <https://github.com/DeepSenseCA/echofilter/tree/1.0.3>`__
------------------------------------------------------------------------

Release date: 2022-11-15.
`Full commit changelog <https://github.com/DeepSenseCA/echofilter/compare/1.0.2...1.0.3>`__.

This minor patch fix addresses package metadata.

.. _v1.0.3 Fixed:

Fixed
~~~~~

.. _v1.0.3 Fixed Metadata:

Metadata
^^^^^^^^

-   Declare ``python_requires>=3.6,<3.11`` requirement.
    (`#264 <https://github.com/DeepSenseCA/echofilter/pull/264>`__,
    `#302 <https://github.com/DeepSenseCA/echofilter/pull/302>`__)
-   Declare ``torch<1.12.0`` requirement.
    (`#302 <https://github.com/DeepSenseCA/echofilter/pull/302>`__)


Version `1.0.2 <https://github.com/DeepSenseCA/echofilter/tree/1.0.2>`__
------------------------------------------------------------------------

Release date: 2022-11-06.
`Full commit changelog <https://github.com/DeepSenseCA/echofilter/compare/1.0.1...1.0.2>`__.

This minor patch fix addresses github dependencies so the package can be pushed to PyPI.

.. _v1.0.2 Changed:

Changed
~~~~~~~

.. _v1.0.2 Changed Requirements:

Requirements
^^^^^^^^^^^^

-   Change ``torch_lr_finder`` train requirement from a specific github commit ref to >=0.2.0.
    (`#260 <https://github.com/DeepSenseCA/echofilter/pull/260>`__)
-   Remove `ranger <https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer>`__ from train requirements.
    (`#261 <https://github.com/DeepSenseCA/echofilter/pull/261>`__)

.. _v1.0.2 Changed Training:

Training
^^^^^^^^

-   Default optimizer changed from ``"rangerva"`` to ``"adam"``.
    If you have manually installed `ranger <https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer>`__ you can still use the ``"rangerva"`` optimizer if you specify it.
    (`#261 <https://github.com/DeepSenseCA/echofilter/pull/261>`__)


Version `1.0.1 <https://github.com/DeepSenseCA/echofilter/tree/1.0.1>`__
------------------------------------------------------------------------

Release date: 2022-11-06.
`Full commit changelog <https://github.com/DeepSenseCA/echofilter/compare/1.0.0...1.0.1>`__.

This patch fix addresses requirement inconsistencies and documentation building.
This release is provided under the `AGPLv3 <https://www.gnu.org/licenses/agpl-3.0.en.html>`__ license.

.. _v1.0.1 Changed:

Changed
~~~~~~~

.. _v1.0.1 Changed Requirements:

Requirements
^^^^^^^^^^^^

-   Add a vendorized copy of functions from
    `torchutils <https://github.com/scottclowe/pytorch-utils>`__
    and remove it from the requirements.
    (`#249 <https://github.com/DeepSenseCA/echofilter/pull/249>`__)

.. _v1.0.1 Fixed:

Fixed
~~~~~

.. _v1.0.1 Fixed Release:

Release
^^^^^^^

-   Added checkpoints.yaml file to package_data.
    (`#255 <https://github.com/DeepSenseCA/echofilter/pull/255>`__)
-   Added appdirs package, required for caching model checkpoints.
    (`#240 <https://github.com/DeepSenseCA/echofilter/pull/240>`__)
-   Support for pytorch>=1.11 by dropping import of ``torch._six.container_abcs``.
    (`#250 <https://github.com/DeepSenseCA/echofilter/pull/250>`__)


Version `1.0.0 <https://github.com/DeepSenseCA/echofilter/tree/1.0.0>`__
------------------------------------------------------------------------

Release date: 2020-10-18.
`Full commit changelog <https://github.com/DeepSenseCA/echofilter/compare/1.0.0rc3...1.0.0>`__.

This is the first major release of echofilter.

.. _v1.0.0 Added:

Added
~~~~~

.. _v1.0.0 Added Inference:

Inference
^^^^^^^^^

-   Add support for loading checkpoints shipped as part of the package.
    (`#228 <https://github.com/DeepSenseCA/echofilter/pull/228>`__)
-   More detailed error messages when unable to download or load a model
    i.e. due to a problem with the Internet connection, a 404 error,
    or because the hard disk is out of space.
    (`#228 <https://github.com/DeepSenseCA/echofilter/pull/228>`__)

.. _v1.0.0 Added Documentation:

Documentation
^^^^^^^^^^^^^

-   Add Usage Guide source and sphinx documentation PDF generation routines
    (`#232 <https://github.com/DeepSenseCA/echofilter/pull/232>`__,
    `#233 <https://github.com/DeepSenseCA/echofilter/pull/233>`__,
    `#234 <https://github.com/DeepSenseCA/echofilter/pull/234>`__,
    `#235 <https://github.com/DeepSenseCA/echofilter/pull/235>`__)


Version `1.0.0rc3 <https://github.com/DeepSenseCA/echofilter/tree/1.0.0rc3>`__
------------------------------------------------------------------------------

Release date: 2020-09-23.
`Full commit changelog <https://github.com/DeepSenseCA/echofilter/compare/1.0.0rc2...1.0.0rc3>`__.

This is the third release candidate for the forthcoming v1.0.0 major release.

.. _v1.0.0rc3 Fixed:

Fixed
~~~~~~~

.. _v1.0.0rc3 Fixed Inference:

Inference
^^^^^^^^^

-   Include extension in temporary EVL file, fixing issue importing it into Echoview.
    (`#224 <https://github.com/DeepSenseCA/echofilter/pull/224>`__)


Version `1.0.0rc2 <https://github.com/DeepSenseCA/echofilter/tree/1.0.0rc2>`__
------------------------------------------------------------------------------

Release date: 2020-09-23.
`Full commit changelog <https://github.com/DeepSenseCA/echofilter/compare/1.0.0rc1...1.0.0rc2>`__.

This is the second release candidate for the forthcoming v1.0.0 major release.

.. _v1.0.0rc2 Fixed:

Fixed
~~~~~~~

.. _v1.0.0rc2 Fixed Inference:

Inference
^^^^^^^^^

-   Fix reference to ``echofilter.raw.loader.evl_loader`` when loading EVL files into Echoview.
    (`#222 <https://github.com/DeepSenseCA/echofilter/pull/222>`__)


Version `1.0.0rc1 <https://github.com/DeepSenseCA/echofilter/tree/1.0.0rc1>`__
------------------------------------------------------------------------------

Release date: 2020-09-23.
`Full commit changelog <https://github.com/DeepSenseCA/echofilter/compare/1.0.0b4...1.0.0rc1>`__.

This is a release candidate for the forthcoming v1.0.0 major release.

.. _v1.0.0rc1 Changed:

Changed
~~~~~~~

.. _v1.0.0rc1 Changed Inference:

Inference
^^^^^^^^^

-   Import lines into Echoview twice, once with and once without offset.
    (`#218 <https://github.com/DeepSenseCA/echofilter/pull/218>`__)
-   EVL outputs now indicate raw depths, before any offset or clipping is applied.
    (`#218 <https://github.com/DeepSenseCA/echofilter/pull/218>`__)
-   Change default ``--lines-during-passive`` value from ``"predict"`` to ``"interpolate-time"``.
    (`#216 <https://github.com/DeepSenseCA/echofilter/pull/216>`__)
-   Disable all bad data region outputs by default.
    (`#217 <https://github.com/DeepSenseCA/echofilter/pull/217>`__)
-   Change default nearfield cut-off behaviour to only clip the bottom line (upfacing data) and not the turbulence line (downfacing data).
    (`#219 <https://github.com/DeepSenseCA/echofilter/pull/219>`__)

.. _v1.0.0rc1 Changed Training:

Training
^^^^^^^^

-   Reduce minimum distance by which surface line must be above turbulence line from 0.25m to 0m.
    (`#212 <https://github.com/DeepSenseCA/echofilter/pull/212>`__)
-   Reduce minimum distance by which bottom line must be above surface line from 0.5m to 0.02m.
    (`#212 <https://github.com/DeepSenseCA/echofilter/pull/212>`__)

.. _v1.0.0rc1 Fixed:

Fixed
~~~~~

.. _v1.0.0rc1 Fixed Inference:

Inference
^^^^^^^^^

-   Change nearfield line for downfacing recordings to be nearfield distance below the shallowest recording depth, not at a depth equal to the nearfield distance.
    (`#214 <https://github.com/DeepSenseCA/echofilter/pull/214>`__)

.. _v1.0.0rc1 Added:

Added
~~~~~

.. _v1.0.0rc1 Added Inference:

Inference
^^^^^^^^^

-   Add new checkpoints: v2.0, v2.1 for stationary model; v2.0, v2.1, v2.2 for conditional hybrid model.
    (`#213 <https://github.com/DeepSenseCA/echofilter/pull/213>`__)
-   Add notes to lines imported into Echoview.
    (`#215 <https://github.com/DeepSenseCA/echofilter/pull/215>`__)
-   Add arguments controlling color and thickness of offset lines (``--color-surface-offset``, etc).
    (`#218 <https://github.com/DeepSenseCA/echofilter/pull/218>`__)
-   Add argument ``--cutoff-at-nearfield`` which re-enables clipping of the turbulence line at nearfield depth with downfacing data.
    (`#219 <https://github.com/DeepSenseCA/echofilter/pull/219>`__)



Version `1.0.0b4 <https://github.com/DeepSenseCA/echofilter/tree/1.0.0b4>`__
----------------------------------------------------------------------------

Release date: 2020-07-05.
`Full commit changelog <https://github.com/DeepSenseCA/echofilter/compare/1.0.0b3...1.0.0b4>`__.

This is a beta pre-release of v1.0.0.

.. _v1.0.0b4 Changed:

Changed
~~~~~~~

.. _v1.0.0b4 Changed Inference:

Inference
^^^^^^^^^

-   Arguments relating to top are renamed to turbulence, and "top" outputs are renamed "turbulence".
    (`#190 <https://github.com/DeepSenseCA/echofilter/pull/190>`__)
-   Change default checkpoint from ``conditional_mobile-stationary2_effunet6x2-1_lc32_v1.0`` to ``conditional_mobile-stationary2_effunet6x2-1_lc32_v2.0``.
    (`#208 <https://github.com/DeepSenseCA/echofilter/pull/208>`__)
-   Status value in EVL outputs extends to final sample (as per specification, not observed EVL files).
    (`#201 <https://github.com/DeepSenseCA/echofilter/pull/201>`__)
-   Rename ``--nearfield-cutoff`` argument to ``--nearfield``, add ``--no-cutoff-at-nearfield`` argument to control whether the turbulence/bottom line can extend closer to the echosounder that the nearfield line.
    (`#203 <https://github.com/DeepSenseCA/echofilter/pull/203>`__)
-   Improved UI help and verbosity messages.
    (`#187 <https://github.com/DeepSenseCA/echofilter/pull/187>`__,
    `#188 <https://github.com/DeepSenseCA/echofilter/pull/188>`__,
    `#203 <https://github.com/DeepSenseCA/echofilter/pull/203>`__,
    `#204 <https://github.com/DeepSenseCA/echofilter/pull/204>`__,
    `#207 <https://github.com/DeepSenseCA/echofilter/pull/207>`__)

.. _v1.0.0b4 Changed Training:

Training
^^^^^^^^

-   Use 0m as target for surface line for downfacing, not the top of the echogram.
    (`#191 <https://github.com/DeepSenseCA/echofilter/pull/191>`__)
-   Don't include periods where the surface line is below the bottom line in the training loss.
    (`#191 <https://github.com/DeepSenseCA/echofilter/pull/191>`__)
-   Bottom line target during nearfield is now the bottom of the echogram, not 0.5m above the bottom.
    (`#191 <https://github.com/DeepSenseCA/echofilter/pull/191>`__)
-   Normalise training samples separately, based on their own Sv intensity distribution after augmentation.
    (`#192 <https://github.com/DeepSenseCA/echofilter/pull/192>`__)
-   Record echofilter version number in checkpoint file.
    (`#193 <https://github.com/DeepSenseCA/echofilter/pull/193>`__)
-   Change "optimal" depth zoom augmentation, used for validation, to cover a slightly wider depth range past the deepest bottom and shallowest surface line.
    (`#194 <https://github.com/DeepSenseCA/echofilter/pull/194>`__)
-   Don't record fraction of image which is active during training.
    (`#206 <https://github.com/DeepSenseCA/echofilter/pull/206>`__)

.. _v1.0.0b4 Changed Miscellaneous:

Miscellaneous
^^^^^^^^^^^^^

-   Rename top->turbulence, bot->bottom surf->surface, throughout all code.
    (`#190 <https://github.com/DeepSenseCA/echofilter/pull/190>`__)
-   Convert undefined value -10000.99 to NaN when loading lines from EVL files.
    (`#191 <https://github.com/DeepSenseCA/echofilter/pull/191>`__)
-   Include surface line in transect plots.
    (`#191 <https://github.com/DeepSenseCA/echofilter/pull/191>`__)
-   Move argparser and colour styling into ui subpackage.
    (`#198 <https://github.com/DeepSenseCA/echofilter/pull/198>`__)
-   Move inference command line interface to its own module to increase responsiveness for non-processing actions (``--help``, ``--version``, ``--list-checkpoints``, ``--list-colors``).
    (`#199 <https://github.com/DeepSenseCA/echofilter/pull/199>`__)

.. _v1.0.0b4 Fixed:

Fixed
~~~~~

.. _v1.0.0b4 Fixed Inference:

Inference
^^^^^^^^^

-   Fix depth extent of region boxes.
    (`#186 <https://github.com/DeepSenseCA/echofilter/pull/186>`__)
-   EVL and EVR outputs extend half a timestamp interval so it is clear what is inside their extent.
    (`#200 <https://github.com/DeepSenseCA/echofilter/pull/200>`__)

.. _v1.0.0b4 Fixed Training:

Training
^^^^^^^^

-   Labels for passive collection times in Minas Passage and Grand Passage datasets are manually set for samples where automatic labeling failed.
    (`#191 <https://github.com/DeepSenseCA/echofilter/pull/191>`__)
-   Interpolate surface depths during passive periods.
    (`#191 <https://github.com/DeepSenseCA/echofilter/pull/191>`__)
-    Smooth out anomalies in the surface line, and exclude the smoothed version from the training loss.
    (`#191 <https://github.com/DeepSenseCA/echofilter/pull/191>`__)
-    Use a looser nearfield removal process when removing the nearfield zone from the bottom line targets, so nearfield is removed from all samples where it needs to be.
    (`#191 <https://github.com/DeepSenseCA/echofilter/pull/191>`__)
-   When reshaping samples, don't use higher order interpolation than first for the bottom line with upfacing data, as the boundaries are rectangular
    (`#191 <https://github.com/DeepSenseCA/echofilter/pull/191>`__)
-   The precision criterion's measurement value when there are no predicted positives equals 1 and if there are no true positives and 0 otherwise (previously 0.5 regardless of target).
    (`#195 <https://github.com/DeepSenseCA/echofilter/pull/195>`__)

.. _v1.0.0b4 Added:

Added
~~~~~

.. _v1.0.0b4 Added Inference:

Inference
^^^^^^^^^

-   Add nearfield line to EV file when importing lines, and add ``--no-nearfield-line`` argument to disable this.
    (`#203 <https://github.com/DeepSenseCA/echofilter/pull/203>`__)
-   Add arguments to control display of nearfield line, ``--color-nearfield`` and ``--thickness-nearfield``.
    (`#203 <https://github.com/DeepSenseCA/echofilter/pull/203>`__)
-   Add ``-r`` and ``-R`` short-hand arguments for recursive and non-recursive directory search.
    (`#189 <https://github.com/DeepSenseCA/echofilter/pull/189>`__)
-   Add ``-s`` short-hand argument for ``--skip``
    (`#189 <https://github.com/DeepSenseCA/echofilter/pull/189>`__)
-   Add two new model checkpoints to list of available checkpoints, ``conditional_mobile-stationary2_effunet6x2-1_lc32_v1.1`` and ``conditional_mobile-stationary2_effunet6x2-1_lc32_v2.0``.
    (`#208 <https://github.com/DeepSenseCA/echofilter/pull/208>`__)
-   Use YAML file to define list of available checkpoints.
    (`#208 <https://github.com/DeepSenseCA/echofilter/pull/208>`__,
    `#209 <https://github.com/DeepSenseCA/echofilter/pull/209>`__)
-   Default checkpoint is shown with an asterisk in checkpoint list.
    (`#202 <https://github.com/DeepSenseCA/echofilter/pull/202>`__)

.. _v1.0.0b4 Added Training:

Training
^^^^^^^^

-   Add cold/warm restart option, for training a model with initial weights from the output of a previously trained model.
    (`#196 <https://github.com/DeepSenseCA/echofilter/pull/196>`__)
-   Add option to manually specify training and validation partitions.
    (`#205 <https://github.com/DeepSenseCA/echofilter/pull/205>`__)



Version `1.0.0b3 <https://github.com/DeepSenseCA/echofilter/tree/1.0.0b3>`__
----------------------------------------------------------------------------

Release date: 2020-06-25.
`Full commit changelog <https://github.com/DeepSenseCA/echofilter/compare/1.0.0b2...1.0.0b3>`__.

This is a beta pre-release of v1.0.0.

.. _v1.0.0b3 Changed:

Changed
~~~~~~~

.. _v1.0.0b3 Changed Inference:

Inference
^^^^^^^^^

-   Rename ``--crop-depth-min`` argument to ``--crop-min-depth``, and ``--crop-depth-max`` argument to ``--crop-max-depth``.
    (`#174 <https://github.com/DeepSenseCA/echofilter/pull/174>`__)
-   Rename ``--force_unconditioned`` argument to ``--force-unconditioned``.
    (`#166 <https://github.com/DeepSenseCA/echofilter/pull/166>`__)
-   Default offset of surface line is now 1m.
    (`#168 <https://github.com/DeepSenseCA/echofilter/pull/168>`__)
-   Change default ``--checkpoint`` so it is always the same (the conditional model), independent of the ``--facing`` argument.
    (`#177 <https://github.com/DeepSenseCA/echofilter/pull/177>`__)
-   Change default ``--lines-during-passive`` from ``"redact"`` to ``"predict"``.
    (`#176 <https://github.com/DeepSenseCA/echofilter/pull/176>`__)
-   Change ``--sufix-csv`` behaviour so it should no longer include ``".csv"`` extension, matching how ``--suffix-file`` is handled.
    (`#171 <https://github.com/DeepSenseCA/echofilter/pull/171>`__,
    `#175 <https://github.com/DeepSenseCA/echofilter/pull/175>`__)
-   Change handling of ``--suffix-var`` and ``--sufix-csv`` to prepend with ``"-"`` as a delimiter if none is included in the string, as was already the case for ``--sufix-file``.
    (`#170 <https://github.com/DeepSenseCA/echofilter/pull/170>`__,
    `#171 <https://github.com/DeepSenseCA/echofilter/pull/171>`__)
-   Include ``--suffix-var`` string in region names.
    (`#173 <https://github.com/DeepSenseCA/echofilter/pull/173>`__)
-   Improved UI help and verbosity messages.
    (`#166 <https://github.com/DeepSenseCA/echofilter/pull/166>`__,
    `#167 <https://github.com/DeepSenseCA/echofilter/pull/167>`__,
    `#170 <https://github.com/DeepSenseCA/echofilter/pull/170>`__,
    `#179 <https://github.com/DeepSenseCA/echofilter/pull/179>`__,
    `#180 <https://github.com/DeepSenseCA/echofilter/pull/180>`__,
    `#182 <https://github.com/DeepSenseCA/echofilter/pull/182>`__)
-   Increase default verbosity level from 1 to 2.
    (`#179 <https://github.com/DeepSenseCA/echofilter/pull/179>`__)

.. _v1.0.0b3 Fixed:

Fixed
~~~~~

.. _v1.0.0b3 Fixed Inference:

Inference
^^^^^^^^^

-   Autocrop with upward facing was running with reflected data as its input, resulting in the data being processed upside down and by the wrong conditional model.
    (`#172 <https://github.com/DeepSenseCA/echofilter/pull/172>`__)
-   Remove duplicate leading byte order mark character from evr file output, which was preventing the file from importing into Echoview.
    (`#178 <https://github.com/DeepSenseCA/echofilter/pull/178>`__)
-   Fix \\r\\n line endings being mapped to \\r\\r\\n on Windows in evl and evr output files.
    (`#178 <https://github.com/DeepSenseCA/echofilter/pull/178>`__)
-   Show error message when importing the evr file into the ev file fails.
    (`#169 <https://github.com/DeepSenseCA/echofilter/pull/169>`__)
-   Fix duplicated Segments tqdm progress bar.
    (`#180 <https://github.com/DeepSenseCA/echofilter/pull/180>`__)

.. _v1.0.0b3 Added:

Added
~~~~~

.. _v1.0.0b3 Added Inference:

Inference
^^^^^^^^^

-   Add ``--offset-surface`` argument, which allows the surface line to be adjusted by a fixed distance.
    (`#168 <https://github.com/DeepSenseCA/echofilter/pull/168>`__)


Version `1.0.0b2 <https://github.com/DeepSenseCA/echofilter/tree/1.0.0b2>`__
----------------------------------------------------------------------------

Release date: 2020-06-18.
`Full commit changelog <https://github.com/DeepSenseCA/echofilter/compare/1.0.0b1...1.0.0b2>`__.

This is a beta pre-release of v1.0.0.

.. _v1.0.0b2 Changed:

Changed
~~~~~~~

.. _v1.0.0b2 Changed Inference:

Inference
^^^^^^^^^

-   Change default value of ``--offset`` to 1m.
    (`#159 <https://github.com/DeepSenseCA/echofilter/pull/159>`__)
-   Use a default ``--nearfield-cutoff`` of 1.7m.
    (`#159 <https://github.com/DeepSenseCA/echofilter/pull/159>`__,
    `#161 <https://github.com/DeepSenseCA/echofilter/pull/161>`__)
-   Show total run time when inference is finished.
    (`#156 <https://github.com/DeepSenseCA/echofilter/pull/156>`__)
-   Only ever report number of skipped regions if there were some which were skipped.
    (`#156 <https://github.com/DeepSenseCA/echofilter/pull/156>`__)

.. _v1.0.0b2 Fixed:

Fixed
~~~~~

.. _v1.0.0b2 Fixed Inference:

Inference
^^^^^^^^^

-   When using the "redact" method for ``--lines-during-passive`` (the default option), depths were redacted but the timestamps were not, resulting in a temporal offset which accumulated with each passive region.
    (`#155 <https://github.com/DeepSenseCA/echofilter/pull/155>`__)
-   Fix behaviour with ``--suffix-file``, so files are written to the filename with the suffix.
    (`#160 <https://github.com/DeepSenseCA/echofilter/pull/160>`__)
-   Fix type of ``--offset-top`` and ``--offset-bottom`` arguments from ``int`` to ``float``.
    (`#159 <https://github.com/DeepSenseCA/echofilter/pull/155>`__)
-   Documentation for ``--overwrite-ev-lines`` argument.
    (`#157 <https://github.com/DeepSenseCA/echofilter/pull/157>`__)

.. _v1.0.0b2 Added:

Added
~~~~~

.. _v1.0.0b2 Added Inference:

Inference
^^^^^^^^^

-   Add ability to specify whether to use recursive search through subdirectory tree, or just files in the specified directory, to both inference.py and ev2csv.py.
    Add ``--no-recursive-dir-search`` argument to enable the non-recursive mode.
    (`#158 <https://github.com/DeepSenseCA/echofilter/pull/158>`__)
-   Add option to cap the top or bottom line (depending on orientation) so it cannot go too close to the echosounder, with ``--nearfield-cutoff`` argument.
    (`#159 <https://github.com/DeepSenseCA/echofilter/pull/159>`__)
-   Add option to skip outputting individual evl lines, with ``--no-top-line``, ``--no-bottom-line``, ``--no-surface-line`` arguments.
    (`#162 <https://github.com/DeepSenseCA/echofilter/pull/162>`__)


Version `1.0.0b1 <https://github.com/DeepSenseCA/echofilter/tree/1.0.0b1>`__
----------------------------------------------------------------------------

Release date: 2020-06-17.
`Full commit changelog <https://github.com/DeepSenseCA/echofilter/compare/0.1.4...1.0.0b1>`__.

This is a beta pre-release of v1.0.0.

.. _v1.0.0b1 Changed:

Changed
~~~~~~~

.. _v1.0.0b1 Changed Training:

Training
^^^^^^^^

-   Built-in line offsets and nearfield line are removed from training targets.
    (`#82 <https://github.com/DeepSenseCA/echofilter/pull/82>`__)
-   Training validation is now against data which is cropped by depth to zoom in on only the "optimal" range of depths (from the shallowest ground truth surface line to the deepest bottom line), using ``echofilter.data.transforms.OptimalCropDepth``.
    (`#83 <https://github.com/DeepSenseCA/echofilter/pull/83>`__,
    `#109 <https://github.com/DeepSenseCA/echofilter/pull/109>`__)
-   Training augmentation stack.
    (`#79 <https://github.com/DeepSenseCA/echofilter/pull/79>`__,
    `#83 <https://github.com/DeepSenseCA/echofilter/pull/83>`__,
    `#106 <https://github.com/DeepSenseCA/echofilter/pull/106>`__,
    `#124 <https://github.com/DeepSenseCA/echofilter/pull/124>`__)
-   Train using normalisation based on the 10th percentile as the zero point and standard deviation robustly estimated from the interdecile range.
    (`#80 <https://github.com/DeepSenseCA/echofilter/pull/80>`__)
-   Use log-avg-exp for ``logit_is_passive`` and ``logit_is_removed``.
    (`#97 <https://github.com/DeepSenseCA/echofilter/pull/97>`__)
-   Exclude data during removed blocks from top and bottom line targets.
    (`#92 <https://github.com/DeepSenseCA/echofilter/pull/92>`__,
    `#110 <https://github.com/DeepSenseCA/echofilter/pull/110>`__,
    `#136 <https://github.com/DeepSenseCA/echofilter/pull/136>`__)
-   Seeding of workers and random state during training.
    (`#93 <https://github.com/DeepSenseCA/echofilter/pull/93>`__,
    `#126 <https://github.com/DeepSenseCA/echofilter/pull/126>`__)
-   Change names of saved checkpoints and log.
    (`#122 <https://github.com/DeepSenseCA/echofilter/pull/122>`__,
    `#132 <https://github.com/DeepSenseCA/echofilter/pull/132>`__)
-   Save UNet state to checkpoint, not the wrapped model.
    (`#133 <https://github.com/DeepSenseCA/echofilter/pull/133>`__)
-   Change and reduce number of images generated when training.
    (`#95 <https://github.com/DeepSenseCA/echofilter/pull/95>`__,
    `#98 <https://github.com/DeepSenseCA/echofilter/pull/98>`__,
    `#99 <https://github.com/DeepSenseCA/echofilter/pull/99>`__,
    `#101 <https://github.com/DeepSenseCA/echofilter/pull/101>`__,
    `#108 <https://github.com/DeepSenseCA/echofilter/pull/108>`__,
    `#112 <https://github.com/DeepSenseCA/echofilter/pull/112>`__,
    `#114 <https://github.com/DeepSenseCA/echofilter/pull/114>`__,
    `#127 <https://github.com/DeepSenseCA/echofilter/pull/127>`__)

.. _v1.0.0b1 Changed Inference:

Inference
^^^^^^^^^

-   Change checkpoints available to be used for inference.
    (`#147 <https://github.com/DeepSenseCA/echofilter/pull/147>`__)
-   Change default checkpoint to be dependent on the ``--facing`` argument.
    (`#147 <https://github.com/DeepSenseCA/echofilter/pull/147>`__)
-   Default line status of output lines changed from ``1`` to ``3``.
    (`#135 <https://github.com/DeepSenseCA/echofilter/pull/135>`__)
-   Default handling of lines during passive data collection changed from implicit ``"predict"`` to ``"redact"``.
    (`#138 <https://github.com/DeepSenseCA/echofilter/pull/138>`__)
-   By default, output logits are smoothed using a Gaussian with width of 1 pixel (relative to the model's latent output space) before being converted into output probibilities.
    (`#144 <https://github.com/DeepSenseCA/echofilter/pull/144>`__)
-   By default, automatically cropping to zoom in on the depth range of interest if the fraction of the depth which could be removed is at least 35% of the original depth.
    (`#149 <https://github.com/DeepSenseCA/echofilter/pull/149>`__)
-   Change default normalisation behaviour to be based on the current input's distribution of Sv values instead of the statistics used for training.
    (`#80 <https://github.com/DeepSenseCA/echofilter/pull/80>`__)
-   Output surface line as an evl file.
    (`f829cb7 <https://github.com/DeepSenseCA/echofilter/commit/f829cb76b1e7ba93062cdc737016ae8aac00a519>`__)
-   Output regions as an evr file.
    (`#141 <https://github.com/DeepSenseCA/echofilter/pull/141>`__,
    `#142 <https://github.com/DeepSenseCA/echofilter/pull/142>`__,
    `#143 <https://github.com/DeepSenseCA/echofilter/pull/143>`__)
-   By default, when running on a .ev file, the generated lines and regions are imported into the file.
    (`#152 <https://github.com/DeepSenseCA/echofilter/pull/152>`__)
-   Renamed ``--csv-suffix`` argument to ``--suffix-csv``.
    (`#152 <https://github.com/DeepSenseCA/echofilter/pull/152>`__)
-   Improved UI help and verbosity messages.
    (`#81 <https://github.com/DeepSenseCA/echofilter/pull/81>`__,
    `#129 <https://github.com/DeepSenseCA/echofilter/pull/129>`__,
    `#137 <https://github.com/DeepSenseCA/echofilter/pull/137>`__,
    `#145 <https://github.com/DeepSenseCA/echofilter/pull/145>`__)

.. _v1.0.0b1 Changed Miscellaneous:

Miscellaneous
^^^^^^^^^^^^^

-   Set Sv values outside the range (-1e37, 1e37) to be NaN (previously values lower than -1e6 were set to NaN).
    (`#140 <https://github.com/DeepSenseCA/echofilter/pull/140>`__)
-   Move modules into subpackages.
    (`#104 <https://github.com/DeepSenseCA/echofilter/pull/104>`__,
    `#130 <https://github.com/DeepSenseCA/echofilter/pull/130>`__)
-   General code tidy up and refactoring.
    (`#85 <https://github.com/DeepSenseCA/echofilter/pull/85>`__,
    `#88 <https://github.com/DeepSenseCA/echofilter/pull/88>`__,
    `#89 <https://github.com/DeepSenseCA/echofilter/pull/89>`__,
    `#94 <https://github.com/DeepSenseCA/echofilter/pull/94>`__,
    `#96 <https://github.com/DeepSenseCA/echofilter/pull/96>`__,
    `#146 <https://github.com/DeepSenseCA/echofilter/pull/146>`__)
-   Change code to use the black style.
    (`#86 <https://github.com/DeepSenseCA/echofilter/pull/86>`__,
    `#87 <https://github.com/DeepSenseCA/echofilter/pull/87>`__)

.. _v1.0.0b1 Fixed:

Fixed
~~~~~

.. _v1.0.0b1 Fixed Training:

Training
^^^^^^^^

-   Edge-cases when resizing data such as lines crossing; surface lines marked as undefined with value ``-10000.99``.
    (`#90 <https://github.com/DeepSenseCA/echofilter/pull/90>`__)
-   Seeding numpy random state for dataloader workers during training.
    (`#93 <https://github.com/DeepSenseCA/echofilter/pull/93>`__)
-   Resume train schedule when resuming training from existing checkpoint.
    (`#120 <https://github.com/DeepSenseCA/echofilter/pull/120>`__)
-   Setting state for RangerVA when resuming training from existing checkpoint.
    (`#121 <https://github.com/DeepSenseCA/echofilter/pull/121>`__)
-   Running LRFinder after everything else is set up for the model.
    (`#131 <https://github.com/DeepSenseCA/echofilter/pull/131>`__)

.. _v1.0.0b1 Fixed Inference:

Inference
^^^^^^^^^

-   Exporting raw data in ev2csv required more Echoview parameters to be disabled, such as the minimum value threshold.
    (`#100 <https://github.com/DeepSenseCA/echofilter/pull/100>`__)

.. _v1.0.0b1 Fixed Miscellaneous:

Miscellaneous
^^^^^^^^^^^^^

-   Fixed behaviour when loading data from CSVs with different number of depth samples and range of depths for different rows in the CSV file.
    (`#102 <https://github.com/DeepSenseCA/echofilter/pull/102>`__,
    `#103 <https://github.com/DeepSenseCA/echofilter/pull/103>`__)

.. _v1.0.0b1 Added:

Added
~~~~~

.. _v1.0.0b1 Added Training:

Training
^^^^^^^^

-   New augmentations: RandomCropDepth, RandomGrid, ElasticGrid,
    (`#83 <https://github.com/DeepSenseCA/echofilter/pull/83>`__,
    `#105 <https://github.com/DeepSenseCA/echofilter/pull/105>`__,
    `#124 <https://github.com/DeepSenseCA/echofilter/pull/124>`__)
-   Add outputs and loss terms for auxiliary targets: original top and bottom line, variants of the patches mask.
    (`#91 <https://github.com/DeepSenseCA/echofilter/pull/91>`__)
-   Add option to exclude passive and removed blocks from line targets.
    (`#92 <https://github.com/DeepSenseCA/echofilter/pull/92>`__)
-   Interpolation method option added to Rescale, randomly selected for training.
    (`#79 <https://github.com/DeepSenseCA/echofilter/pull/79>`__)
-   More input scaling options.
    (`#80 <https://github.com/DeepSenseCA/echofilter/pull/80>`__)
-   Add option to specify pooling operation for ``logit_is_passive`` and ``logit_is_removed``.
    (`#97 <https://github.com/DeepSenseCA/echofilter/pull/97>`__)
-   Support training on Grand Passage dataset.
    (`#101 <https://github.com/DeepSenseCA/echofilter/pull/101>`__)
-   Support training on multiple datasets.
    (`#111 <https://github.com/DeepSenseCA/echofilter/pull/111>`__,
    `#113 <https://github.com/DeepSenseCA/echofilter/pull/113>`__)
-   Add ``stationary2`` dataset which contains both MinasPassage and two copies of GrandPassage with different augmentations, and ``mobile+stationary2`` dataset.
    (`#111 <https://github.com/DeepSenseCA/echofilter/pull/111>`__,
    `#113 <https://github.com/DeepSenseCA/echofilter/pull/113>`__)
-   Add conditional model architecture training wrapper.
    (`#116 <https://github.com/DeepSenseCA/echofilter/pull/116>`__)
-   Add outputs for conditional targets to tensorboard.
    (`#125 <https://github.com/DeepSenseCA/echofilter/pull/125>`__,
    `#134 <https://github.com/DeepSenseCA/echofilter/pull/134>`__)
-   Add stratified data sampler, which preserves the balance between datasets in each training batch.
    (`#117 <https://github.com/DeepSenseCA/echofilter/pull/117>`__)
-   Training process error catching.
    (`#119 <https://github.com/DeepSenseCA/echofilter/pull/119>`__)
-   Training on multiple GPUs on the same node for a single model.
    (`#123 <https://github.com/DeepSenseCA/echofilter/pull/123>`__,
    `#133 <https://github.com/DeepSenseCA/echofilter/pull/133>`__)

.. _v1.0.0b1 Added Inference:

Inference
^^^^^^^^^

-   Add ``--line-status`` argument, which controls the status to use in the evl output for the lines.
    (`#135 <https://github.com/DeepSenseCA/echofilter/pull/135>`__)
-   Add multiple methods of how to handle lines during passive data, and argument ``--lines-during-passive`` to control which method to use.
    (`#138 <https://github.com/DeepSenseCA/echofilter/pull/138>`__,
    `#148 <https://github.com/DeepSenseCA/echofilter/pull/148>`__)
-   Add ``--offset``, ``--offset-top``, ``--offset-bottom`` arguments, which allows the top and bottom lines to be adjusted by a fixed distance.
    (`#139 <https://github.com/DeepSenseCA/echofilter/pull/139>`__)
-   Write regions to evr file.
    (`#141 <https://github.com/DeepSenseCA/echofilter/pull/141>`__,
    `#142 <https://github.com/DeepSenseCA/echofilter/pull/142>`__,
    `#143 <https://github.com/DeepSenseCA/echofilter/pull/143>`__)
-   Add ``--logit-smoothing-sigma`` argument, which controls the kernel width for Gaussian smoothing applied to the logits before converting to predictions.
    (`#144 <https://github.com/DeepSenseCA/echofilter/pull/144>`__)
-   Generating outputs from conditional models, adding ``--unconditioned`` argument to disable usage of conditional probability outputs.
    (`#147 <https://github.com/DeepSenseCA/echofilter/pull/147>`__)
-   Add automatic cropping to zoom in on the depth range of interest.
    Add ``--auto-crop-threshold`` argument, which controls the threshold for when this occurs.
    (`#149 <https://github.com/DeepSenseCA/echofilter/pull/149>`__)
-   Add ``--list-checkpoints`` action, which lists the available checkpoints.
    (`#150 <https://github.com/DeepSenseCA/echofilter/pull/150>`__)
-   Fast fail if outputs already exist before processing already begins (and overwrite mode is not enabled).
    (`#151 <https://github.com/DeepSenseCA/echofilter/pull/151>`__)
-   Import generated line and region predictions from the .evl and .evr files into the .ev file and save it with the new lines and regions included.
    The ``--no-ev-import`` argument prevents this behaviour.
    (`#152 <https://github.com/DeepSenseCA/echofilter/pull/152>`__)
-   Add customisation of imported lines.
    The ``--suffix-var`` argument controls the suffix append to the name of the line variable.
    The ``--overwrite-ev-lines`` argument controls whether lines are overwritten if lines already exist with the same name.
    Also add arguments to customise the colour and thickness of the lines.
    (`#152 <https://github.com/DeepSenseCA/echofilter/pull/152>`__)
-   Add ``--suffix-file`` argument, will allows a suffix common to all the output files to be set.
    (`#152 <https://github.com/DeepSenseCA/echofilter/pull/152>`__)

.. _v1.0.0b1 Added Miscellaneous:

Miscellaneous
^^^^^^^^^^^^^

-   Add ``-V`` alias for ``--version`` to all command line interfaces.
    (`#84 <https://github.com/DeepSenseCA/echofilter/pull/84>`__)
-   Loading data from CSV files which contain invalid characters outside the UTF-8 set (seen in the Grand Passage dataset's csv files).
    (`#101 <https://github.com/DeepSenseCA/echofilter/pull/101>`__)
-   Handle raw and masked CSV data of different sizes (occuring in Grand Passage's csv files due to dropped rows containing invalid chararcters).
    (`#101 <https://github.com/DeepSenseCA/echofilter/pull/101>`__)
-   Add seed argument to separation script.
    (`#56 <https://github.com/DeepSenseCA/echofilter/pull/56>`__)
-   Add sample script to extract raw training data from ev files.
    (`#55 <https://github.com/DeepSenseCA/echofilter/pull/55>`__)


Version `0.1.4 <https://github.com/DeepSenseCA/echofilter/tree/0.1.4>`__
------------------------------------------------------------------------

Release date: 2020-05-19.
`Full commit changelog <https://github.com/DeepSenseCA/echofilter/compare/0.1.3...0.1.4>`__.

.. _v0.1.4 Added:

Added
~~~~~

-   Add ability to set orientation of echosounder with ``--facing`` argument
    (`#77 <https://github.com/DeepSenseCA/echofilter/pull/77>`__)
    The orientation is shown to the user if it was automatically detected as upward-facing
    (`#76 <https://github.com/DeepSenseCA/echofilter/pull/76>`__)


Version `0.1.3 <https://github.com/DeepSenseCA/echofilter/tree/0.1.3>`__
------------------------------------------------------------------------

Release date: 2020-05-16.
`Full commit changelog <https://github.com/DeepSenseCA/echofilter/compare/0.1.2...0.1.3>`__.

.. _v0.1.3 Fixed:

Fixed
~~~~~

-   EVL writer needs to output time to nearest 0.1ms.
    (`#72 <https://github.com/DeepSenseCA/echofilter/pull/72>`__)

.. _v0.1.3 Added:

Added
~~~~~

-   Add ``--suffix`` argument to the command line interface of ``ev2csv``.
    (`#71 <https://github.com/DeepSenseCA/echofilter/pull/71>`__)
-   Add ``--variable-name`` argument to ``inference.py`` (the main command line interface).
    (`#74 <https://github.com/DeepSenseCA/echofilter/pull/74>`__)


Version `0.1.2 <https://github.com/DeepSenseCA/echofilter/tree/0.1.2>`__
------------------------------------------------------------------------

Release date: 2020-05-14.
`Full commit changelog <https://github.com/DeepSenseCA/echofilter/compare/0.1.1...0.1.2>`__.

.. _v0.1.2 Fixed:

Fixed
~~~~~

-   In ``ev2csv``, the files generator needed to be cast as a list to measure the number of files.
    (`#66 <https://github.com/DeepSenseCA/echofilter/pull/66>`__)
-   Echoview is no longer opened during dry-run mode.
    (`#66 <https://github.com/DeepSenseCA/echofilter/pull/66>`__)
-   In ``parse_files_in_folders`` (affecting ``ev2csv``), string inputs were not being handled correctly.
    (`#66 <https://github.com/DeepSenseCA/echofilter/pull/66>`__)
-   Relative paths need to be converted to absolute paths before using them in Echoview.
    (`#68 <https://github.com/DeepSenseCA/echofilter/pull/68>`__,
    `#69 <https://github.com/DeepSenseCA/echofilter/pull/69>`__)

.. _v0.1.2 Added:

Added
~~~~~

-   Support hiding or minimizing Echoview while the script is running. The default behaviour is now to hide the window if it was created by the script. The same Echoview window is used throughout the the processing.
    (`#67 <https://github.com/DeepSenseCA/echofilter/pull/67>`__)


Version `0.1.1 <https://github.com/DeepSenseCA/echofilter/tree/0.1.1>`__
------------------------------------------------------------------------

Release date: 2020-05-12.
`Full commit changelog <https://github.com/DeepSenseCA/echofilter/compare/0.1.0...0.1.1>`__.

.. _v0.1.1 Fixed:

Fixed
~~~~~

-   Padding in echofilter.modules.pathing.FlexibleConcat2d when only one dim size doesn't match.
    (`#64 <https://github.com/DeepSenseCA/echofilter/pull/64>`__)


Version `0.1.0 <https://github.com/DeepSenseCA/echofilter/tree/0.1.0>`__
------------------------------------------------------------------------

Release date: 2020-05-12.
Initial release.
