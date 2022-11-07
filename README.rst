echofilter
==========

Echofilter is an application for segmenting an echogram. It takes as its
input an Echoview_ .EV file, and produces as its output several lines and
regions:

-  turbulence (entrained air) line

-  bottom (seafloor) line

-  surface line

-  nearfield line

-  passive data regions

-  \*bad data regions for entirely removed periods of time, in the form
   of boxes covering the entire vertical depth

-  \*bad data regions for localised anomalies, in the form of polygonal
   contour patches

Echofilter uses a machine learning model to complete this task.
The machine learning model was trained on upfacing stationary and downfacing
mobile data provided by Fundy Ocean Research Centre for Energy (FORCE).

Full documentation can be viewed at `readthedocs <https://echofilter.readthedocs.io/en/stable/>`__.

.. _doi: https://www.doi.org/10.3389/fmars.2022.867857

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

    echofilter -h


Installation
------------

The package can be installed using pip as follows:

.. code:: bash

    pip install git+https://github.com/DeepSenseCA/echofilter


.. _Echoview: https://www.echoview.com/


Citing Echofilter
-----------------

If you use Echofilter for your research, we would be grateful if you could cite our
paper on echofilter in any resulting publications:

    SC Lowe, LP McGarry, J Douglas, J Newport, S Oore, C Whidden, DJ Hasselman (2022). Echofilter: A Deep Learning Segmention Model Improves the Automation, Standardization, and Timeliness for Post-Processing Echosounder Data in Tidal Energy Streams. *Front. Mar. Sci.*, **9**, 1–21.
    doi: |nbsp| `10.3389/fmars.2022.867857 <doi_>`_.

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
