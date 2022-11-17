echofilter
==========

+------------------+----------------------------------------------------------------------+
| Latest Release   | |PyPI badge|                                                         |
+------------------+----------------------------------------------------------------------+
| License          | |License|                                                            |
+------------------+----------------------------------------------------------------------+
| Documentation    | |readthedocs|                                                        |
+------------------+----------------------------------------------------------------------+
| Build Status     | |Documentation| |GHA tests| |Codecov| |pre-commit-status|            |
+------------------+----------------------------------------------------------------------+
| Code style       | |black| |pre-commit|                                                 |
+------------------+----------------------------------------------------------------------+
| Citation         | |DOI badge|                                                          |
+------------------+----------------------------------------------------------------------+

Echofilter is an application for segmenting an echogram. It takes as its
input an Echoview_ .EV file, and produces as its output several lines and
regions:

-  entrained air (turbulence) line

-  seafloor line

-  surface line

-  nearfield line

-  passive data regions

-  (unreliable) bad data regions for entirely removed periods of time, in the form
   of boxes covering the entire vertical depth

-  (unreliable) bad data regions for localised anomalies, in the form of polygonal
   contour patches

Echofilter uses a machine learning model to complete this task.
The machine learning model was trained on upfacing stationary and downfacing
mobile data provided by Fundy Ocean Research Centre for Energy (FORCE).
The training and evaluation data is
`available for download <https://data.fundyforce.ca/forceCloud/index.php/s/BzC87LpbGtnFsjT>`__.
Queries regarding dataset access should be directed to FORCE, info@fundyforce.ca.

The experimental methodology and results can be found in our
`companion paper <doi_>`_, published in Frontiers in Marine Science.

Full documentation of how to use echofilter can be viewed at `readthedocs`_.

If you encounter a specific problem please `open a new issue`_.

.. _Echoview: https://www.echoview.com/
.. _doi: https://www.doi.org/10.3389/fmars.2022.867857
.. _readthedocs: https://echofilter.readthedocs.io/en/1.1.1/usage/
.. _open a new issue: https://github.com/DeepSenseCA/echofilter/issues/new

Usage
-----

After installing, the model can be applied at the command prompt with:

.. code:: bash

    echofilter PATH PATH2 ...

Any number of paths can be specified. Each path can either be a path to
a single csv file to process (exported using the Echoview_ application),
or a directory containing csv files. If a directory is given, all csv files
within nested subfolders of the directory will be processed.

All optional parameters can be seen by running ``echofilter`` with the help
argument.

.. code:: bash

    echofilter --help

For more details, see the
`Usage Guide <https://echofilter.readthedocs.io/en/1.1.1/usage/>`__,
and the
`command line interface (CLI) reference <https://echofilter.readthedocs.io/en/1.1.1/programs/inference.html>`__
documentation.


Installation
------------

Installing as a stand-alone executable
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For your convenience, we provide a copy of Echofilter compiled as
a stand-alone executable for Windows.
To install this, download and unzip the echofilter-executable-M.N.P.zip file
from the latest release in the
`releases tab <https://github.com/DeepSenseCA/echofilter/releases>`__.
For example:
`echofilter-executable-1.1.1.zip <https://github.com/DeepSenseCA/echofilter/releases/download/1.1.1/echofilter-executable-1.1.1.zip>`__

For more details, see the step-by-step instructions in the
`Usage Guide <https://echofilter.readthedocs.io/en/1.1.1/usage/installation.html#installing-as-an-executable-file>`__.

Note: The precompiled executable has only CPU support, and does not support
running on GPU.

Installing in Python
^^^^^^^^^^^^^^^^^^^^

Alternatively, the echofilter package can be installed for Python 3.6 or 3.7
using pip as follows.

First, install torch.

Either with CPU-only capabilities:

.. code:: bash

    pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

Or with CUDA GPU support as well:

.. code:: bash

    pip install torch==1.4.0 torchvision==0.5.0 -f https://download.pytorch.org/whl/torch_stable.html

Then install the rest of the requirements.

.. code:: bash

    pip install -r frozen_requirements.txt
    pip install echofilter


Citing Echofilter
-----------------

For technical details about how the Echofilter model was trained, and our
findings about its empirical results, please consult our companion paper:

    SC Lowe, LP McGarry, J Douglas, J Newport, S Oore, C Whidden, DJ Hasselman (2022). Echofilter: A Deep Learning Segmention Model Improves the Automation, Standardization, and Timeliness for Post-Processing Echosounder Data in Tidal Energy Streams. *Front. Mar. Sci.*, **9**, 1–21.
    doi: |nbsp| `10.3389/fmars.2022.867857 <doi_>`_.

If you use Echofilter for your research, we would be grateful if you could cite
this paper in any resulting publications.

For your convenience, we provide a copy of this citation in `bibtex`_ format.

.. _bibtex: https://raw.githubusercontent.com/DeepSenseCA/echofilter/master/CITATION.bib

You can browse papers which utilise Echofilter `here <gscholarcitations_>`_.

.. _gscholarcitations: https://scholar.google.com/scholar?cites=18122679926970563847


License
-------

Copyright (C) 2020-2022  Scott C. Lowe and Offshore Energy Research Association (OERA)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, version 3.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.


.. |nbsp| unicode:: 0xA0
   :trim:
.. |PyPI badge| image:: https://img.shields.io/pypi/v/echofilter.svg
   :target: https://pypi.org/project/echofilter/
   :alt: Latest PyPI release
.. |GHA tests| image:: https://github.com/DeepSenseCA/echofilter/workflows/tests/badge.svg?branch=1.1.1
   :target: https://github.com/DeepSenseCA/echofilter/actions?query=workflow%3Atest
   :alt: GHA Status
.. |readthedocs| image:: https://img.shields.io/badge/docs-readthedocs-blue
   :target: readthedocs_
   :alt: Documentation
.. |Documentation| image:: https://readthedocs.org/projects/echofilter/badge/?version=1.1.1
   :target: readthedocs_
   :alt: Documentation Status
.. |Codecov| image:: https://codecov.io/gh/DeepSenseCA/echofilter/branch/v1.1.x/graph/badge.svg?token=BGX2EJ0SSI
   :target: https://codecov.io/gh/DeepSenseCA/echofilter
   :alt: Coverage
.. |DOI badge| image:: https://img.shields.io/badge/DOI-10.3389/fmars.2022.867857-blue.svg
   :target: doi_
   :alt: DOI
.. |License| image:: https://img.shields.io/pypi/l/echofilter
   :target: https://raw.githubusercontent.com/DeepSenseCA/echofilter/master/COPYING
   :alt: AGPLv3 License
.. |pre-commit| image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
   :target: https://github.com/pre-commit/pre-commit
   :alt: pre-commit enabled
.. |pre-commit-status| image:: https://results.pre-commit.ci/badge/github/DeepSenseCA/echofilter/master.svg
   :target: https://results.pre-commit.ci/1.1.1/github/DeepSenseCA/echofilter/master
   :alt: pre-commit.ci status
.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: black
