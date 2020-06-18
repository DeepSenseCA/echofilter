Changelog
=========

All notable changes to echofilter will be documented here.

The format is based on `Keep a Changelog`_, and this project adheres to
`Semantic Versioning`_.

.. _Keep a Changelog: https://keepachangelog.com/en/1.0.0/
.. _Semantic Versioning: https://semver.org/spec/v2.0.0.html

Categories for changes are: Added, Changed, Deprecated, Removed, Fixed,
Security.


Unreleased
----------

`Full commit changelog <https://github.com/DeepSenseCA/echofilter/compare/0.1.4...master>`__.


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
    (`#82 <https://github.com/DeepSenseCA/echofilter/pull/82>`__).
-   Training validation is now against data which is cropped by depth to zoom in on only the "optimal" range of depths (from the shallowest ground truth surface line to the deepest bottom line), using ``echofilter.data.transforms.OptimalCropDepth``.
    (`#83 <https://github.com/DeepSenseCA/echofilter/pull/83>`__,
    `#109 <https://github.com/DeepSenseCA/echofilter/pull/109>`__).
-   Training augmentation stack.
    (`#79 <https://github.com/DeepSenseCA/echofilter/pull/79>`__,
    `#83 <https://github.com/DeepSenseCA/echofilter/pull/83>`__,
    `#106 <https://github.com/DeepSenseCA/echofilter/pull/106>`__).
-   Train using normalisation based on the 10th percentile as the zero point and standard deviation robustly estimated from the interdecile range.
    (`#80 <https://github.com/DeepSenseCA/echofilter/pull/80>`__).
-   Use log-avg-exp for ``logit_is_passive`` and ``logit_is_removed``.
    (`#97 <https://github.com/DeepSenseCA/echofilter/pull/97>`__).
-   Exclude data during removed blocks from top and bottom line targets.
    (`#92 <https://github.com/DeepSenseCA/echofilter/pull/92>`__,
    `#110 <https://github.com/DeepSenseCA/echofilter/pull/110>`__,
    `#136 <https://github.com/DeepSenseCA/echofilter/pull/136>`__).
-   Seeding of workers and random state during training.
    (`#93 <https://github.com/DeepSenseCA/echofilter/pull/93>`__,
    `#126 <https://github.com/DeepSenseCA/echofilter/pull/126>`__).
-   Change names of saved checkpoints and log.
    (`#122 <https://github.com/DeepSenseCA/echofilter/pull/122>`__,
    `#132 <https://github.com/DeepSenseCA/echofilter/pull/132>`__).
-   Save UNet state to checkpoint, not the wrapped model.
    (`#133 <https://github.com/DeepSenseCA/echofilter/pull/133>`__).
-   Change and reduce number of images generated when training.
    (`#95 <https://github.com/DeepSenseCA/echofilter/pull/95>`__,
    `#98 <https://github.com/DeepSenseCA/echofilter/pull/98>`__,
    `#99 <https://github.com/DeepSenseCA/echofilter/pull/99>`__,
    `#101 <https://github.com/DeepSenseCA/echofilter/pull/101>`__,
    `#108 <https://github.com/DeepSenseCA/echofilter/pull/108>`__,
    `#112 <https://github.com/DeepSenseCA/echofilter/pull/112>`__,
    `#114 <https://github.com/DeepSenseCA/echofilter/pull/114>`__,
    `#127 <https://github.com/DeepSenseCA/echofilter/pull/127>`__).

.. _v1.0.0b1 Changed Inference:

Inference
^^^^^^^^^

-   Change checkpoints available to be used for inference.
    (`#147 <https://github.com/DeepSenseCA/echofilter/pull/147>`__).
-   Change default checkpoint to be dependent on the ``--facing`` argument.
    (`#147 <https://github.com/DeepSenseCA/echofilter/pull/147>`__).
-   Default line status of output lines changed from ``1`` to ``3``.
    (`#135 <https://github.com/DeepSenseCA/echofilter/pull/135>`__).
-   Default handling of lines during passive data collection changed from implicit ``"predict"`` to ``"redact"``.
    (`#138 <https://github.com/DeepSenseCA/echofilter/pull/138>`__).
-   By default, output logits are smoothed using a Gaussian with width of 1 pixel (relative to the model's latent output space) before being converted into output probibilities.
    (`#144 <https://github.com/DeepSenseCA/echofilter/pull/144>`__)
-   By default, automatically cropping to zoom in on the depth range of interest if the fraction of the depth which could be removed is at least 35% of the original depth.
    (`#149 <https://github.com/DeepSenseCA/echofilter/pull/149>`__).
-   Change default normalisation behaviour to be based on the current input's distribution of Sv values instead of the statistics used for training.
    (`#80 <https://github.com/DeepSenseCA/echofilter/pull/80>`__).
-   Output surface line as an evl file.
    (`f829cb7 <https://github.com/DeepSenseCA/echofilter/commit/f829cb76b1e7ba93062cdc737016ae8aac00a519>`__)
-   Output regions as an evr file.
    (`#141 <https://github.com/DeepSenseCA/echofilter/pull/141>`__,
    `#142 <https://github.com/DeepSenseCA/echofilter/pull/142>`__,
    `#143 <https://github.com/DeepSenseCA/echofilter/pull/143>`__).
-   By default, when running on a .ev file, the generated lines and regions are imported into the file.
    (`#152 <https://github.com/DeepSenseCA/echofilter/pull/152>`__)
-   Renamed ``--csv-suffix`` argument to ``--suffix-csv``.
    (`#152 <https://github.com/DeepSenseCA/echofilter/pull/152>`__)
-   Improved UI help and verbosity messages.
    (`#81 <https://github.com/DeepSenseCA/echofilter/pull/81>`__,
    `#129 <https://github.com/DeepSenseCA/echofilter/pull/129>`__,
    `#137 <https://github.com/DeepSenseCA/echofilter/pull/137>`__,
    `#145 <https://github.com/DeepSenseCA/echofilter/pull/145>`__).

.. _v1.0.0b1 Changed General:

General
^^^^^^^

-   Set Sv values outside the range (-1e37, 1e37) to be NaN (previously values lower than -1e6 were set to NaN).
    (`#140 <https://github.com/DeepSenseCA/echofilter/pull/140>`__).
-   Move modules into subpackages.
    (`#104 <https://github.com/DeepSenseCA/echofilter/pull/104>`__,
    `#130 <https://github.com/DeepSenseCA/echofilter/pull/130>`__).
-   General code tidy up and refactoring.
    (`#85 <https://github.com/DeepSenseCA/echofilter/pull/85>`__,
    `#88 <https://github.com/DeepSenseCA/echofilter/pull/88>`__,
    `#89 <https://github.com/DeepSenseCA/echofilter/pull/89>`__,
    `#94 <https://github.com/DeepSenseCA/echofilter/pull/94>`__,
    `#96 <https://github.com/DeepSenseCA/echofilter/pull/96>`__,
    `#146 <https://github.com/DeepSenseCA/echofilter/pull/146>`__).
-   Change code to use the black style.
    (`#86 <https://github.com/DeepSenseCA/echofilter/pull/86>`__,
    `#87 <https://github.com/DeepSenseCA/echofilter/pull/87>`__).

.. _v1.0.0b1 Fixed:

Fixed
~~~~~

.. _v1.0.0b1 Fixed Training:

Training
^^^^^^^^

-   Edge-cases when resizing data such as lines crossing; surface lines marked as undefined with value ``-10000.99``.
    (`#90 <https://github.com/DeepSenseCA/echofilter/pull/90>`__).
-   Seeding numpy random state for dataloader workers during training.
    (`#93 <https://github.com/DeepSenseCA/echofilter/pull/93>`__).
-   Resume train schedule when resuming training from existing checkpoint.
    (`#120 <https://github.com/DeepSenseCA/echofilter/pull/120>`__).
-   Setting state for RangerVA when resuming training from existing checkpoint.
    (`#121 <https://github.com/DeepSenseCA/echofilter/pull/121>`__).
-   Running LRFinder after everything else is set up for the model.
    (`#131 <https://github.com/DeepSenseCA/echofilter/pull/131>`__).

.. _v1.0.0b1 Fixed Inference:

Inference
^^^^^^^^^

-   Exporting raw data in ev2csv required more EchoView parameters to be disabled, such as the minimum value threshold.
    (`#100 <https://github.com/DeepSenseCA/echofilter/pull/100>`__).

.. _v1.0.0b1 Fixed General:

General
^^^^^^^

-   Fixed behaviour when loading data from CSVs with different number of depth samples and range of depths for different rows in the CSV file.
    (`#102 <https://github.com/DeepSenseCA/echofilter/pull/102>`__, `#103 <https://github.com/DeepSenseCA/echofilter/pull/103>`__).


.. _v1.0.0b1 Added:

Added
~~~~~

.. _v1.0.0b1 Added Training:

Training
^^^^^^^^

-   New augmentations: RandomCropDepth, RandomGrid, ElasticGrid,
    (`#83 <https://github.com/DeepSenseCA/echofilter/pull/83>`__,
    `#105 <https://github.com/DeepSenseCA/echofilter/pull/105>`__,
    `#124 <https://github.com/DeepSenseCA/echofilter/pull/124>`__).
-   Add outputs and loss terms for auxiliary targets: original top and bottom line, variants of the patches mask.
    (`#91 <https://github.com/DeepSenseCA/echofilter/pull/91>`__).
-   Add option to exclude passive and removed blocks from line targets.
    (`#92 <https://github.com/DeepSenseCA/echofilter/pull/92>`__).
-   Interpolation method option added to Rescale, randomly selected for training.
    (`#79 <https://github.com/DeepSenseCA/echofilter/pull/79>`__,
-   More input scaling options.
    (`#80 <https://github.com/DeepSenseCA/echofilter/pull/80>`__).
-   Add option to specify pooling operation for ``logit_is_passive`` and ``logit_is_removed``.
    (`#97 <https://github.com/DeepSenseCA/echofilter/pull/97>`__).
-   Support training on Grand Passage dataset.
    (`#101 <https://github.com/DeepSenseCA/echofilter/pull/101>`__).
-   Support training on multiple datasets.
    (`#111 <https://github.com/DeepSenseCA/echofilter/pull/111>`__,
    `#113 <https://github.com/DeepSenseCA/echofilter/pull/113>`__).
-   Add ``stationary2`` dataset which contains both MinasPassage and two copies of GrandPassage with different augmentations, and ``mobile+stationary2`` dataset.
    (`#111 <https://github.com/DeepSenseCA/echofilter/pull/111>`__,
    `#113 <https://github.com/DeepSenseCA/echofilter/pull/113>`__).
-   Add conditional model architecture training wrapper.
    (`#116 <https://github.com/DeepSenseCA/echofilter/pull/116>`__).
-   Add outputs for conditional targets to tensorboard.
    (`#125 <https://github.com/DeepSenseCA/echofilter/pull/125>`__,
    `#134 <https://github.com/DeepSenseCA/echofilter/pull/134>`__).
-   Add stratified data sampler, which preserves the balance between datasets in each training batch.
    (`#117 <https://github.com/DeepSenseCA/echofilter/pull/117>`__).
-   Training process error catching.
    (`#119 <https://github.com/DeepSenseCA/echofilter/pull/119>`__).
-   Training on multiple GPUs on the same node for a single model.
    (`#123 <https://github.com/DeepSenseCA/echofilter/pull/123>`__,
    `#133 <https://github.com/DeepSenseCA/echofilter/pull/133>`__).

.. _v1.0.0b1 Added Inference:

Inference
^^^^^^^^^

-   Add ``--line-status`` argument, which controls the status to use in the evl output for the lines.
    (`#135 <https://github.com/DeepSenseCA/echofilter/pull/135>`__).
-   Add multiple methods of how to handle lines during passive data, and argument ``--lines-during-passive`` to control which method to use.
    (`#138 <https://github.com/DeepSenseCA/echofilter/pull/138>`__,
    `#148 <https://github.com/DeepSenseCA/echofilter/pull/148>`__).
-   Add ``--offset``, ``--offset-top``, ``--offset-bottom`` arguments, which allows the top and bottom lines to be adjusted by a fixed distance.
    (`#139 <https://github.com/DeepSenseCA/echofilter/pull/139>`__).
-   Write regions to evr file.
    (`#141 <https://github.com/DeepSenseCA/echofilter/pull/141>`__,
    `#142 <https://github.com/DeepSenseCA/echofilter/pull/142>`__,
    `#143 <https://github.com/DeepSenseCA/echofilter/pull/143>`__).
-   Add ``--logit-smoothing-sigma`` argument, which controls the kernel width for Gaussian smoothing applied to the logits before converting to predictions.
    (`#144 <https://github.com/DeepSenseCA/echofilter/pull/144>`__)
-   Generating outputs from conditional models, adding 
    (`#147 <https://github.com/DeepSenseCA/echofilter/pull/147>`__).
-   Add automatic cropping to zoom in on the depth range of interest.
    Add ``--auto-crop-threshold`` argument, which controls the threshold for when this occurs.
    (`#149 <https://github.com/DeepSenseCA/echofilter/pull/149>`__).
-   Add ``--list-checkpoints`` action, which lists the available checkpoints.
    (`#150 <https://github.com/DeepSenseCA/echofilter/pull/150>`__).
-   Fast fail if outputs already exist before processing already begins (and overwrite mode is not enabled).
    (`#151 <https://github.com/DeepSenseCA/echofilter/pull/151>`__).
-   Import generated line and region predictions from the .evl and .evr files into the .ev file and save it with the new lines and regions included.
    The ``--no-ev-import`` argument prevents this behaviour.
    (`#152 <https://github.com/DeepSenseCA/echofilter/pull/152>`__).
-   Add customisation of imported lines.
    The ``--suffix-var`` argument controls the suffix append to the name of the line variable.
    The ``--overwrite-ev-lines`` argument controls whether lines are overwritten if lines already exist with the same name.
    Also add arguments to customise the colour and thickness of the lines.
    (`#152 <https://github.com/DeepSenseCA/echofilter/pull/152>`__).
-   Add ``--suffix-file`` argument, will allows a suffix common to all the output files to be set.
    (`#152 <https://github.com/DeepSenseCA/echofilter/pull/152>`__).

.. _v1.0.0b1 Added General:

General
^^^^^^^

-   Add ``-V`` alias for ``--version`` to all command line interfaces.
    (`#84 <https://github.com/DeepSenseCA/echofilter/pull/84>`__).
-   Loading data from CSV files which contain invalid characters outside the UTF-8 set (seen in the Grand Passage dataset's csv files).
    (`#101 <https://github.com/DeepSenseCA/echofilter/pull/101>`__).
-   Handle raw and masked CSV data of different sizes (occuring in Grand Passage's csv files due to dropped rows containing invalid chararcters).
    (`#101 <https://github.com/DeepSenseCA/echofilter/pull/101>`__).
-   Add seed argument to separation script.
    (`#56 <https://github.com/DeepSenseCA/echofilter/pull/56>`__).
-   Add sample script to extract raw training data from ev files.
    (`#55 <https://github.com/DeepSenseCA/echofilter/pull/55>`__).


Version `0.1.4 <https://github.com/DeepSenseCA/echofilter/tree/0.1.4>`__
------------------------------------------------------------------------

Release date: 2020-05-19.
`Full commit changelog <https://github.com/DeepSenseCA/echofilter/compare/0.1.3...0.1.4>`__.

.. _v0.1.4 Added:

Added
~~~~~

-   Add ability to set orientation of echosounder with ``--facing`` argument
    (`#77 <https://github.com/DeepSenseCA/echofilter/pull/77>`__).
    The orientation is shown to the user if it was automatically detected as upward-facing
    (`#76 <https://github.com/DeepSenseCA/echofilter/pull/76>`__).


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
    (`#68 <https://github.com/DeepSenseCA/echofilter/pull/68>`__, `#69 <https://github.com/DeepSenseCA/echofilter/pull/69>`__)

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
