Pre-trained models
------------------

The currently available model checkpoints can be seen by running the
command::

    echofilter --list-checkpoints

All current checkpoints were trained on data acquired by
`FORCE <http://fundyforce.ca>`__.

Training Datasets
~~~~~~~~~~~~~~~~~

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

Details for notable model checkpoints are provided below.

:conditional_mobile-stationary2_effunet6x2-1_lc32_v2.2:

   -  Trained on both :term:`upfacing` :term:`stationary` and
      :term:`downfacing` :term:`mobile` data.

   -  Jaccard Index of **96.84%** on :term:`downfacing` :term:`mobile` and
      **94.51%** on :term:`upfacing` :term:`stationary`
      :term:`validation<Validation set>` data.

   -  Default model checkpoint.

:conditional_mobile-stationary2_effunet6x2-1_lc32_v2.1:

   -  Trained on both :term:`upfacing` :term:`stationary` and
      :term:`downfacing` :term:`mobile` data.

   -  Jaccard Index of 96.8% on :term:`downfacing` :term:`mobile` and
      94.4% on :term:`upfacing` :term:`stationary`
      :term:`validation<Validation set>` data.

:conditional_mobile-stationary2_effunet6x2-1_lc32_v2.0:

   -  Trained on both :term:`upfacing` :term:`stationary` and
      :term:`downfacing` :term:`mobile` data.

   -  Jaccard Index of 96.62% on :term:`downfacing` :term:`mobile` and
      94.29% on :term:`upfacing` :term:`stationary`
      :term:`validation<Validation set>` data.

   -  :term:`Sample<Sample (model input)>` outputs on :term:`upfacing`
      :term:`stationary` data were thoroughly verified via manual inspection
      by trained analysts.

:stationary2_effunet6x2-1_lc32_v2.1:

   -  Trained on :term:`upfacing` :term:`stationary` data only.

   -  Jaccard Index of 94.4% on :term:`upfacing` :term:`stationary`
      :term:`validation<Validation set>` data.

:stationary2_effunet6x2-1_lc32_v2.0:

   -  Trained on :term:`upfacing` :term:`stationary` data only.

   -  Jaccard Index of 94.41% on :term:`upfacing` :term:`stationary`
      :term:`validation<Validation set>` data.

   -  :term:`Sample<Sample (model input)>` outputs thoroughly were thoroughly
      verified via manual inspection by trained analysts.

:mobile_effunet6x2-1_lc32_v1.0:

   -  Trained on :term:`downfacing` :term:`mobile` data only.

.. raw:: latex

    \clearpage
