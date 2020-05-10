echofilter
==========

Automatically identify entrained air in echosounder data.


Usage
-----

After installing, the model can be applied at the command prompt with:

.. code:: bash

    echofilter PATH PATH2 ...

Any number of paths can be specified. Each path can either be a path to
a single csv file to process (exported using the EchoView_ application),
or a directory containing csv files. If a directory is given, all csv files
within nested subfolders of the directory will be processed.

.. _EchoView: https://www.echoview.com/

All optional parameters can be seen by running ``echofilter`` with the help
argument.

.. code:: bash

    echofilter -h


Installation
------------

The package can be installed using pip as follows:

.. code:: bash

    pip install git+https://github.com/DeepSenseCA/echofilter
