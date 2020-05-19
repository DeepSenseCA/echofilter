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


Version `0.1.4 <https://github.com/DeepSenseCA/echofilter/tree/0.1.4>`__
------------------------------------------------------------------------

.. _v0.1.4 Added:

Added
~~~~~

-   Add ability to set orientation of echosounder with ``--facing`` argument
    (`#77 <https://github.com/DeepSenseCA/echofilter/pull/77>`__).
    The orientation is shown to the user if it was automatically detected as upward-facing
    (`#76 <https://github.com/DeepSenseCA/echofilter/pull/76>`__).


Version `0.1.3 <https://github.com/DeepSenseCA/echofilter/tree/0.1.3>`__
------------------------------------------------------------------------

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

.. _v0.1.1 Fixed:

Fixed
~~~~~

-   Padding in echofilter.modules.pathing.FlexibleConcat2d when only one dim size doesn't match.
    (`#64 <https://github.com/DeepSenseCA/echofilter/pull/64>`__)


Version `0.1.0 <https://github.com/DeepSenseCA/echofilter/tree/0.1.0>`__
------------------------------------------------------------------------

Release date: 2020-05-12.
Initial release.
