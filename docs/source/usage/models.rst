Pre-trained models
------------------

The currently available model checkpoints can be seen by running the
command::

    echofilter --list-checkpoints

All current checkpoints were trained on data acquired by
`FORCE <http://fundyforce.ca>`__.

.. _Model checkpoints:

Model checkpoints
~~~~~~~~~~~~~~~~~

The architecture used for all current models is a U-Net with a backbone
of 6 EfficientNet blocks in each direction (encoding and decoding).
There are horizontal skip connections between compression and expansion
blocks at the same spatial scale and a latent space of 32 channels
throughout the network. The depth dimension of the input is halved
(doubled) after each block, whilst the time dimension is halved
(doubled) every other block.

For details about how the Echofilter models were trained, and our findings about
their empirical performance, please consult our companion paper:

    SC Lowe, LP McGarry, J Douglas, J Newport, S Oore, C Whidden, DJ Hasselman (2022). Echofilter: A Deep Learning Segmention Model Improves the Automation, Standardization, and Timeliness for Post-Processing Echosounder Data in Tidal Energy Streams. *Front. Mar. Sci.*, **9**, 1–21.
    doi: |nbsp| `10.3389/fmars.2022.867857 <doi_>`_.

.. |nbsp| unicode:: 0xA0
   :trim:
.. _doi: https://www.doi.org/10.3389/fmars.2022.867857

An overview for of notable model checkpoints available in echofilter are
provided below.

echofilter-v1_bifacing_700ep
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-   Trained on both :term:`upfacing` :term:`stationary` and
    :term:`downfacing` :term:`mobile` data.

-   Overall IoU performance of
    **99.15%** on :term:`downfacing` :term:`mobile` and
    93.0%--94.9% on :term:`upfacing` :term:`stationary`
    :term:`test<Test set>` data.

-   Default model checkpoint.

echofilter-v1_bifacing_300ep
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-   Trained on both :term:`upfacing` :term:`stationary` and
    :term:`downfacing` :term:`mobile` data.

-   Overall IoU performance of
    99.02% on :term:`downfacing` :term:`mobile` and
    93.2%--95.0% on :term:`upfacing` :term:`stationary`
    :term:`test<Test set>` data.

echofilter-v1_bifacing_100ep
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-   Trained on both :term:`upfacing` :term:`stationary` and
    :term:`downfacing` :term:`mobile` data.

-   Overall IoU performance of
    98.93% on :term:`downfacing` :term:`mobile` and
    **93.5%**--94.9% on :term:`upfacing` :term:`stationary`
    :term:`test<Test set>` data.

-   :term:`Sample<Sample (model input)>` outputs on :term:`upfacing`
    :term:`stationary` data were thoroughly verified via manual inspection
    by trained analysts.

echofilter-v1_upfacing_600ep
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-   Trained on :term:`upfacing` :term:`stationary` data only.

-   Overall IoU performance of
    92.1%--**95.1%** on :term:`upfacing` :term:`stationary`
    :term:`test<Test set>` data.

echofilter-v1_upfacing_200ep
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-   Trained on :term:`upfacing` :term:`stationary` data only.

-   Overall IoU performance of
    93.3%--95.1% on :term:`upfacing` :term:`stationary`
    :term:`test<Test set>` data.

-   :term:`Sample<Sample (model input)>` outputs thoroughly were thoroughly
    verified via manual inspection by trained analysts.

echofilter-v0.5_downfacing_300ep
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-   Trained on :term:`downfacing` :term:`mobile` data only.


Training Datasets
~~~~~~~~~~~~~~~~~

The machine learning model was trained on upfacing stationary and downfacing
mobile data provided by Fundy Ocean Research Centre for Energy (FORCE).
The training and evaluation data is
`available for download <https://data.fundyforce.ca/forceCloud/index.php/s/BzC87LpbGtnFsjT>`__.
Queries regarding dataset access should be directed to FORCE, info@fundyforce.ca.

Stationary
^^^^^^^^^^

:data collection:
    bottom-mounted :term:`stationary`, autonomous

:orientation:
    uplooking

:echosounder:
    120 kHz Simrad WBAT

:locations:

    - FORCE tidal power demonstration site, Minas Passage

        - 45°21'47.34"N  64°25'38.94"W
        - December 2017 through November 2018

    - SMEC, Grand Passage

        - 44°15'49.80"N  66°20'12.60"W
        - December 2019 through January 2020

:organization:
    FORCE

Mobile
^^^^^^

:data collection:
    vessel-based 24-hour transect surveys

:orientation:
    downlooking

:echosounder:
    120 kHz Simrad EK80

:locations:

    -  FORCE tidal power demonstration site, Minas Passage

        - 45°21'57.58"N  64°25'50.97"W
        - May 2016 through October 2018

:organization:
    FORCE

.. raw:: latex

    \clearpage
