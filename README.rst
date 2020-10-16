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
