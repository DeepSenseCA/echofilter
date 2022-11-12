Usage Guide
###########

Authors
    Scott C. Lowe, Louise McGarry

.. highlight:: powershell

.. raw:: latex

    \clearpage

Introduction
------------

:term:`Echofilter` is an application for segmenting an echogram. It takes as
its input an :term:`Echoview` .EV file, and produces as its output several
lines and regions:

-  :term:`turbulence<turbulence line>` (:term:`entrained air`) line

-  :term:`bottom (seafloor) line<bottom line>`

-  :term:`surface line`

-  :term:`nearfield line`

-  :term:`passive data` regions

-  \*bad data regions for entirely removed periods of time, in the form
   of boxes covering the entire vertical depth

-  \*bad data regions for localised anomalies, in the form of polygonal
   contour patches

:term:`Echofilter` uses a :term:`machine learning<Machine learning (ML)>`
:term:`model` to complete this task. The machine learning model was trained on
:term:`upfacing` :term:`stationary` and :term:`downfacing` :term:`mobile` data
provided by Fundy Ocean Research Centre for Energy
(`FORCE <http://fundyforce.ca>`__.).

Disclaimers
~~~~~~~~~~~

-  The :term:`model` is only confirmed to work reliably with :term:`upfacing`
   data recorded at the same location and with the same instrumentation as
   the data it was trained on. It is expected to work well on a wider
   range of data, but this has not been confirmed. Even on data similar
   to the :term:`training data`, the :term:`model` is not perfect and it is
   recommended that a human analyst manually inspects the results it generates
   to confirm they are correct.

-  \* :term:`Bad data regions` are particularly challenging for the
   :term:`model` to generate. Consequently, the bad data region outputs are
   not reliable and should be considered experimental. By default, these
   outputs are disabled.

-  Integration with :term:`Echoview` was tested for Echoview 10 and 11.

.. raw:: latex

    \clearpage


Glossary
--------

.. glossary::

    Active data
        Data collected while the :term:`echosounder` is emitting sonar pulses
        (":term:`pings<ping>`") at regular intervals. This is the normal
        operating mode for data in this project.

    Algorithm
        A finite sequence of well-defined, unambiguous,
        computer-implementable operations.

    Bad data regions
        Regions of data which must be excluded from analysis in their entirety.
        Bad data regions identified by :ref:`echofilter<echofilter CLI>` come
        in two forms: rectangular regions covering the full depth-extend of the
        echogram for a period of time, and polygonal or contour regions
        encompassing a localised area.

    Bottom line
        A line separating the seafloor from the :term:`water column`.

    Checkpoint
        A checkpoint file defines the weights for a particular
        :term:`neural network` :term:`model`.

    Conditional model
        A :term:`model` which outputs conditional probabilities. In the context
        of an :term:`echofilter` model, the conditional probabilities are
        :math:`p(x|\text{upfacing})` and :math:`p(x|\text{downfacing})`,
        where :math:`x` is any of the :term:`model` output
        types; conditional models are necessarily hybrid models.

    CSV
        A comma-separated values file. The :term:`Sv` data can be exported
        into this format by :term:`Echoview`.

    Dataset
        A collection of data :term:`samples<Sample (model input)>`. In this
        project, the datasets are :term:`Sv` recordings from multiple surveys.

    Downfacing
        The orientation of an :term:`echosounder` when it is located at the
        surface and records from the :term:`water column` below it.

    Echofilter
        A software package for defining the placement of the boundary lines
        and regions required to post-process :term:`echosounder` data.
        The topic of this usage guide.

    echofilter.exe
        The compiled :ref:`echofilter<echofilter CLI>` program which can be
        run on a Windows machine.

    Echogram
        The two-dimensional representation of a temporal series of
        :term:`echosounder`-collected data. Time is along the x-axis, and depth
        along the y-axis. A common way of plotting :term:`echosounder`
        recordings.

    Echosounder
        An electronic system that includes a computer, transceiver, and
        :term:`transducer`. The system emits sonar :term:`pings<ping>` and
        records the intensity of the reflected echos at some fixed sampling
        rate.

    Echoview
        A Windows software application (`Echoview <https://www.echoview.com/>`__
        Software Pty Ltd, Tasmania, Australia) for hydroacoustic data
        post-processing.

    Entrained air
        Bubbles of air which have been submerged into the ocean by waves or
        by the strong :term:`turbulence` commonly found in tidal energy
        channels.

    EV file
        An :term:`Echoview` file bundling :term:`Sv` data together with
        associated lines and regions produced by processing.

    EVL
        The :term:`Echoview` line file format.

    EVR
        The :term:`Echoview` region file format.

    Inference
        The procedure of using a :term:`model` to generate output predictions
        based on a particular input.

    Hybrid model
        A :term:`model` which has been trained on both :term:`downfacing` and
        :term:`upfacing` data.

    Machine learning (ML)
        The process by which an :term:`algorithm` builds a mathematical model
        based on :term:`sample<Sample (model input)>` data
        (":term:`training data`"), in order to make predictions or decisions
        without being explicitly programmed to do so. A subset of the field of
        Artificial Intelligence.

    Mobile
        A mobile :term:`echosounder` is one which is moving (relative to the
        ocean floor) during its period of operation.

    Model
        A mathematical model of a particular type of data. In our context,
        the model understands an echogram-like input
        :term:`sample<Sample (model input)>` of :term:`Sv` data
        (which is its input) and outputs a probability distribution for
        where it predicts the :term:`turbulence` (:term:`entrained air`)
        boundary, :term:`bottom boundary<Bottom line>`, and
        :term:`surface boundary<Surface line>` to be located, and the
        probability of :term:`passive<Passive data>` periods and
        :term:`bad data<Bad data regions>`.

    Nearfield
        The region of space too close to the :term:`echosounder` to collect
        viable data.

    Nearfield distance
        The maximum distance which is too close to the :term:`echosounder` to
        be viable for data collection.

    Nearfield line
        A line placed at the :term:`nearfield distance`.

    Neural network
        An artificial neural network contains layers of interconnected
        neurons with weights between them. The weights are learned through a
        :term:`machine learning<Machine learning (ML)>` process. After
        :term:`training`, the network is a :term:`model` mapping inputs to
        outputs.

    Passive data
        Data collected while the :term:`echosounder` is silent. Since the sonar
        pulses are not being generated, only ambient sounds are collected.
        This package is designed for analysing :term:`active data`, and hence
        :term:`passive data` is marked for removal.

    Ping
        An :term:`echosounder` sonar pulse event.

    Sample (model input)
        A single echogram-like matrix of :term:`Sv` values.

    Sample (ping)
        A single datapoint recorded at a certain temporal latency in response
        to a particular :term:`ping`.

    Stationary
        A stationary :term:`echosounder` is at a fixed location (relative to
        the ocean floor) during its period of operation.

    Surface line
        Separates atmosphere and water at the ocean surface.

    Sv
        The volume backscattering strength.

    Test set
        Data which was used to evaluate the ability of the :term:`model` to
        generalise to novel, unseen data.

    Training
        The process by which a :term:`model` is iteratively improved.

    Training data
        Data which was used to train the :term:`model(s)<model>`.

    Training set
        A subset (partition) of the :term:`dataset` which was used to train
        the :term:`model`.

    Transducer
        An underwater electronic device that converts electrical energy to
        sound pressure energy. The emitted sound pulse is called a
        ":term:`ping`". The device converts the returning sound pressure
        energy to electrical energy, which is then recorded.

    Turbulence
        In contrast to laminar flow, fluid motion in turbulent regions are
        characterized by chaotic fluctuations in flow speed and direction.
        Air is often entrained into the :term:`water column` in regions of
        strong turbulence.

    Turbulence line
        A line demarcating the depth of the end-boundary of air entrained
        into the :term:`water column` by :term:`turbulence` at the sea
        surface.

    Upfacing
        The orientation of an :term:`echosounder` when it is located at the
        seabed and records from the :term:`water column` above it.

    Validation set
        Data which was used during the :term:`training` process to evaluate the
        ability of the :term:`model` to generalise to novel, unseen data.

    Water column
        The body of water between seafloor and ocean surface.

.. raw:: latex

    \clearpage


Inference operations
--------------------

In this section, we describe the :term:`inference` process, its outputs and
inputs. Inference is the process of generating predictions from the
:term:`model`, and is the principal functionality of
:ref:`echofilter<echofilter CLI>`.

Processing overview
~~~~~~~~~~~~~~~~~~~

This is an overview of how files are processed in the :term:`inference`
pipeline.

First, the setup:

-  If a directory input was given, determine list of files to process.

-  Download the model :term:`checkpoint`, if necessary.

-  Load the :term:`model` from the :term:`checkpoint` into memory.

-  If any file to process is an :term:`EV file`, open :term:`Echoview`.

-  If it was not already open, hide the Echoview window.

After the :term:`model` is loaded from its checkpoint, each file is processed
in turn. The processing time for an individual file scales linearly with
the number of :term:`pings<ping>` in the file (twice as many pings = twice as
long to process).

Each file is processed in the following steps:

-  If the input is an :term:`EV file`, export the :term:`Sv` data to
   :term:`CSV` format.

   -  By default, the :term:`Sv` data is taken from ``"Fileset1: Sv pings T1"``.

   -  Unless ``--cache-csv`` is provided, the :term:`CSV file<CSV>` is
      output to a temporary file, which is deleted after the
      :term:`CSV file<CSV>` is imported.

-  Import the :term:`Sv` data from the :term:`CSV file<CSV>`. (If the
   input was a :term:`CSV file<CSV>`, this is the input; if the input
   was an :term:`EV file` this is the :term:`CSV file<CSV>` generated
   from the :term:`EV file` in the preceding step.)

-  Rescale the height of the :term:`Sv` input to have the number of pixels
   expected by the :term:`model`.

-  Automatically determine whether the :term:`echosounder` recording is
   :term:`upfacing` or :term:`downfacing`, based on the order of the Depths
   data in the :term:`CSV file<CSV>`.

   -  If the orientation was manually specified, issue a warning if it
      does not match the detected orientation.

   -  Reflect the data in the Depth dimension if it is :term:`upfacing`, so
      that the shallowest :term:`samples<Sample (ping)>` always occur first,
      and deepest last.

-  Normalise the distribution of the :term:`Sv` intensities to match that
   expected by the :term:`model`.

-  Split the input data into segments

   -  Detect temporal discontinuities between :term:`pings<ping>`.

   -  Split the input :term:`Sv` data into segments such that each segment
      contains contiguous :term:`pings<ping>`.

-  Pass the each segment of the input through the :term:`model` to generate
   output probabilities.

-  Crop the depth dimension down to zoom in on the most salient data.

   -  If :term:`upfacing`, crop the top off the echogram to show only 2m above
      the shallowest estimated :term:`surface line` depth.

   -  If :term:`downfacing`, crop the bottom off the echogram only 2m below the
      deepest estimated :term:`bottom line` depth.

   -  If more than 35% of the echogram's height (threshold value set
      with ``--autocrop-threshold``) was cropped away, pass the cropped
      :term:`Sv` data through the :term:`model` to get better predictions
      based on the zoomed in data.

-  Line boundary probabilities are converted into output depths.

   -  The boundary probabilities at each pixel are integrated to make a
      cumulative probability distribution across depth,
      :math:`p(\text{depth} > \text{boundary location})`.

   -  The output boundary depth is estimated as the depth at which the
      cumulative probability distribution first exceeds 50%.

-  Bottom, surface, and turbulence lines are output to :term:`EVL` files.

   -  Note: there is no EVL file for the :term:`nearfield line` since it
      is at a constant depth as provided by the user and not generated by
      the :term:`model`.

-  Regions are generated:

   -  Regions are collated if there is a small gap between consecutive
      :term:`passive data` or :term:`bad data regions`.

   -  Regions which are too small (fewer than 10 pings for rectangles)
      are dropped.

   -  All regions are written to a single :term:`EVR` file.

-  If the input was an :term:`EV file`, the lines and regions are imported into
   the :term:`EV file`, and a :term:`nearfield line` is added.

Simulating processing
~~~~~~~~~~~~~~~~~~~~~

To see which files will be processed by a command and what the output
will be, run :ref:`echofilter<echofilter CLI>` with the ``--dry-run`` argument.

Input
~~~~~

:ref:`Echofilter<echofilter CLI>` can process two types of file as its
input: .EV files and .CSV files. The :term:`EV file` input is more
user-friendly, but requires the Windows operating system, and a fully
operational :term:`Echoview` application (i.e. with an Echoview dongle).
The :term:`CSV file<CSV>` format can be processed without Echoview, but
must be generated in advance from the .EV file on a system with Echoview.
The :term:`CSV files<CSV>` must contain raw :term:`Sv` data (without
thresholding or masking) and in the format produced by exporting
:term:`Sv` data from Echoview. These raw :term:`CSV files<CSV>` can be
exported using the utility :ref:`ev2csv<ev2csv CLI>`, which is provided
as a separate executable in the :ref:`echofilter<echofilter CLI>` package.

If the input path is a directory, all files in the directory are
processed. By default, all subdirectories are recursively processed;
this behaviour can be disabled with the ``--no-recursive-dir-search``
argument. All files in the directory (and subdirectories) with an
appropriate file extension will be processed. By default, files with a
.CSV or .EV file extension (case insensitive) which will be processed.
The file extensions to include can be set with the ``--extension`` argument.

Multiple input files or directories can also be specified (each
separated by a space).

By default, when processing an :term:`EV file`, the :term:`Sv` data is taken
from the ``"Fileset1: Sv pings T1"`` variable. This can be changed with the
``--variable-name`` argument.

Loading model
~~~~~~~~~~~~~

The :term:`model` used to process the data is loaded from a :term:`checkpoint`
file. The executable :term:`echofilter.exe` comes with its default model
checkpoint bundled as part of the release. Aside from this, the first time a
particular model is used, the checkpoint file will be downloaded over the
internet. The checkpoint file will be cached on your system and will not need
to be downloaded again unless you clear your cache.

Multiple models are available to select from. These can be shown by
running the command ``echofilter --list-checkpoints``. The default model
will be highlighted in the output. In general, it is recommended to use
the default checkpoint. See :ref:`Model checkpoints` below for more details.

When running :ref:`echofilter<echofilter CLI>` for :term:`inference`, the
checkpoint can be specified with the ``--checkpoint`` argument.

If you wish to use a custom model which is not built in to :term:`echofilter`,
specify a path to the checkpoint file using the ``--checkpoint`` argument.

Output
~~~~~~

Output files
^^^^^^^^^^^^

For each input file, :ref:`echofilter<echofilter CLI>` produces the
following output files:

<input>.bottom.evl
    An Echoview line file containing the depth of the
    :term:`bottom line`.

<input>.regions.evr
    An Echoview region file containing
    spatiotemporal definitions of :term:`passive<passive data>` recording
    rectangle regions, :term:`bad data<Bad data regions>` full-vertical depth
    rectangle regions, and :term:`bad data<Bad data regions>` anomaly
    polygonal (contour) regions.

<input>.surface.evl
    An Echoview line file containing the depth of
    the :term:`surface line`.

<input>.turbulence.evl
    An Echoview line file containing the depth of
    the :term:`turbulence line`.

where <input> is the path to an input file, stripped of its file
extension. There is no :term:`EVL` file for the :term:`nearfield line`, since
it is a virtual line of fixed depth added to the :term:`EV file` during the
:ref:`Importing outputs into EV file` step.

By default, the output files are located in the same directory as the
file being processed. The output directory can be changed with the
``--output-dir`` argument, and a user-defined suffix can be added to the
output file names using the ``--suffix`` argument.

If the output files already exist, by default :ref:`echofilter<echofilter CLI>`
will stop running and raise an error. If you want to overwrite output files
which already exist, supply the ``--overwrite-files`` argument. If you want to
skip inputs whose output files all already exist, supply the ``--skip``
argument. Note: if both ``--skip`` and ``--overwrite-files`` are supplied,
inputs whose outputs all exist will be skipped and those inputs for
which only some of the outputs exist will have existing outputs
overwritten.

Specific outputs can be dropped by supplying the corresponding argument
``--no-bottom-line``, ``--no-surface-line``, or ``--no-turbulence-line``
respectively. To drop particular types of region entirely from the :term:`EVR`
output, use ``--minimum-passive-length -1``, ``--minimum-removed-length -1``,
or ``--minimum-patch-area -1`` respectively. By default,
:term:`bad data<Bad data regions>` regions (rectangles and contours) are not
included in the :term:`EVR` file. To include these, set
``--minimum-removed-length`` and ``--minimum-patch-area`` to non-negative
values.

The lines written to the :term:`EVL` files are the raw output from the model
and do not include any offset.

.. _Importing outputs into EV file:

Importing outputs into EV file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If the input file is an Echoview :term:`EV file`, by default
:ref:`echofilter<echofilter CLI>` will import the output files into the
:term:`EV file` and save the :term:`EV file` (overwriting the original
:term:`EV file`). The behaviour can be disabled by supplying the
``--no-ev-import`` argument.

All lines will be imported twice: once at the original depth and a
second time with an offset included. This offset ensures the exclusion
of data biased by the acoustic deadzone, and provides a margin of safety
at the bottom depth of the :term:`entrained air`. The offset moves the
:term:`surface<surface line>` and :term:`turbulence<turbulence line>` lines
downwards (deeper), and the :term:`bottom line` upwards (shallower).
The default offset is 1m for all three lines, and can be
set using the ``--offset`` argument. A different offset can be used for each
line by providing the ``--offset-bottom``, ``--offset-surface``, and
``--offset-turbulence`` arguments.

The names of the objects imported into the :term:`EV file` have the suffix
``"_echofilter"`` appended to them, to indicate the source of the
line/region. However, if the ``--suffix`` argument was provided, that suffix
is used instead. A custom suffix for the variable names within the EV
file can be specified using the ``--suffix-var`` argument.

If the variable name to be used for a line is already in use, the
default behaviour is to append the current datetime to the new variable
name. To instead overwrite existing line variables, supply the
``--overwrite-ev-lines`` argument. Note that existing regions will not be
overwritten (only lines).

By default, a :term:`nearfield line` is also added to the :term:`EV file`
at a fixed range of 1.7m from the :term:`transducer` position.
The :term:`nearfield distance` can be changed as appropriate for the
:term:`echosounder` in use by setting the ``--nearfield`` parameter.

The colour and thickness of the lines can be customised using the
``--color-surface``, ``--thickness-surface`` (etc) arguments.
See ``echofilter --list-colors`` to see the list of supported colour names.


Installation
------------

Installing as an executable file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:ref:`Echofilter<echofilter CLI>` is distributed as an
:term:`executable binary file<echofilter.exe>` for Windows. All
dependencies are packaged as part of the distribution.

1. Download the zip file containing the echofilter executable as follows:

   a. Go to the `releases tab <https://github.com/DeepSenseCA/echofilter/releases>`__ of the echofilter repository.

   b. Select the release to download. It is recommended to use the latest
      version, with the highest release number.

   c. Click on the file named echofilter-executable-M.N.P.zip, where M.N.P is
      replaced with the version number, to download it.
      For example:
      `echofilter-executable-1.0.2.zip <https://github.com/DeepSenseCA/echofilter/releases/download/1.0.2/echofilter-executable-1.0.2.zip>`__

      Alternatively, the zipped executables can be downloaded from a mirror on
      `GDrive <https://drive.google.com/open?id=1Vq_fVNGzFGwyqHxigX-5maW9UmXfwdOk>`__.

2. Unzip the zip file, and put the directory contained within it
   wherever you like on your Windows machine. It is recommended to put
   it as an "echofilter" directory within your Programs folder, or
   similar. (You may need the
   `WinZip <https://www.winzip.com/win/en/>`__ or
   `7z <https://www.7-zip.org/download.html>`__ application to unzip
   the .zip file.)

3. In File Explorer,

   a. navigate to the echofilter directory you unzipped. This directory
      contains a file named :term:`echofilter.exe`.

   b. left click on the echofilter directory containing the
      :term:`echofilter.exe` file

   c. Shift+Right click on the echofilter directory

   d. select "Copy as path"

   e. paste the path into a text editor of your choice (e.g. Notepad)

4. Find and open the Command Prompt application (your Windows machine
   comes with this pre-installed). That application is also called
   cmd.exe. It will open a window containing a terminal within which
   there is a command prompt where you can type to enter commands.

5. Within the Command Prompt window (the terminal window):

   a. type: ``"cd "`` (without quote marks, with a trailing space) and
      then right click and select paste in order to paste the full path
      to the echofilter directory, which you copied to the clipboard
      in step 3d.

   b. press enter to run this command, which will change the current
      working directory of the terminal to the echofilter directory.

   c. type: ``echofilter --version``

   d. press enter to run this command

   e. you will see the version number of echofilter printed in the
      terminal window

   f. type: ``echofilter --help``

   g. press enter to run this command

   h. you will see the help for echofilter printed in the terminal
      window

6. (Optional) So that you can just run :ref:`echofilter<echofilter CLI>`
   without having to change directory (using the ``cd`` command) to the
   directory containing :term:`echofilter.exe`, or use the full path to
   :term:`echofilter.exe`, every time you want to use it, it is useful to
   add echofilter to the PATH environment variable. This step is entirely
   optional and for your convenience only. The PATH environment variable
   tells the terminal where it should look for executable commands.

   a. Instructions for how to do this depend on your version of Windows
      and can be found here:
      `https://www.computerhope.com/issues/ch000549.htm <https://www.computerhope.com/issues/ch000549.htm>`__.

   b. An environment variable named PATH (case-insensitive) should
      already exist.

   c. If this is a string, you need to edit the string and prepend the
      path from 3e, plus a semicolon. For example, change the
      current value of
      ``C:\Program Files;C:\Winnt;C:\Winnt\System32``
      into
      ``C:\Program Files\echofilter;C:\Program Files;C:\Winnt;C:\Winnt\System32``

   d. If this is a list of strings (without semicolons), add your path
      from 3e (e.g. ``C:\Program Files\echofilter``) to the list

7. You can now run :ref:`echofilter<echofilter CLI>` on some files, by using
   the echofilter command in the terminal. :ref:`Example commands` are shown
   below.

.. raw:: latex

    \clearpage


Quick Start
-----------

Note that it is recommended to close :term:`Echoview` before running
:ref:`echofilter<echofilter CLI>` so that :ref:`echofilter<echofilter CLI>`
can run its own Echoview instance in the background.
After :ref:`echofilter<echofilter CLI>` has started processing the files,
you can open Echoview again for your own use without interrupting
:ref:`echofilter<echofilter CLI>`.

Recommended first time usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first time you use :ref:`echofilter<echofilter CLI>`, you should run
it in simulation mode (by supplying the ``--dry-run`` argument)
before-hand so you can see what it will do::

    echofilter some/path/to/directory_or_file --dry-run

The path you supply to :ref:`echofilter<echofilter CLI>` can be an
absolute path, or a relative path. If it is a relative path, it should be
relative to the current working directory of the command prompt.

.. _Example commands:

Example commands
~~~~~~~~~~~~~~~~

Review echofilter's documentation help within the terminal::

    echofilter --help

Specifying a single file to process, using an absolute path::

    echofilter "C:\Users\Bob\Desktop\MinasPassage\2020\20200801_SiteA.EV"

Specifying a single file to process, using a path relative to the
current directory of the command prompt::

    echofilter "MinasPassage\2020\20200801_SiteA.EV"

Simulating processing of a single file, using a relative path::

    echofilter "MinasPassage\2020\20200801_SiteA.EV" --dry-run

Specifying a directory of :term:`upfacing` :term:`stationary` data to process,
and excluding the bottom line from the output::

    echofilter "C:\Users\Bob\OneDrive\Desktop\MinasPassage\2020" --no-bottom-line

Specifying a directory of :term:`downfacing` :term:`mobile` data to process,
and excluding the surface line from the output::

    echofilter "C:\Users\Bob\Documents\MobileSurveyData\Survey11" --no-surface-line

Processing the same directory after some files were added to it,
skipping files already processed::

    echofilter "C:\Users\Bob\Documents\MobileSurveyData\Survey11" --no-surface --skip

Processing the same directory after some files were added to it,
overwriting files already processed::

    echofilter "C:\Users\Bob\Documents\MobileSurveyData\Survey11" --no-surface --force

Ignoring all :term:`bad data regions` (default),
using ``^`` to break up the long command into multiple lines::

    echofilter "path/to/file_or_directory" ^
        --minimum-removed-length -1 ^
        --minimum-patch-area -1

Including :term:`bad data regions` in the :term:`EVR` output::

    echofilter "path/to/file_or_directory" ^
        --minimum-removed-length 10 ^
        --minimum-patch-area 25

Keep line predictions during :term:`passive<passive data>` periods (default
is to linearly interpolate lines during passive data collection)::

    echofilter "path/to/file_or_directory" --lines-during-passive predict

Specifying file and variable suffix, and line colours and thickness::

    echofilter "path/to/file_or_directory" ^
        --suffix "_echofilter_stationary-model" ^
        --color-surface "green" --thickness-surface 4 ^
        --color-nearfield "red" --thickness-nearfield 3

Processing a file with more output messages displayed in the terminal::

    echofilter "path/to/file_or_directory" --verbose

Processing a file and sending the output to a log file instead of the
terminal::

    echofilter "path/to/file_or_directory" -v > path/to/log_file.txt 2>&1


Argument documentation
~~~~~~~~~~~~~~~~~~~~~~

:ref:`Echofilter<echofilter CLI>` has a large number of customisation options.
The complete list of argument options available to the user can be seen in the
:ref:`CLI Reference<echofilter CLI>`, or by consulting the help for
:ref:`echofilter<echofilter CLI>`. The help documentation is output to the
terminal when you run the command ``echofilter --help``.


Actions
~~~~~~~

The main :ref:`echofilter<echofilter CLI>` action is to perform
:term:`inference` on a file or collection of files. However, certain
arguments trigger different actions.

help
^^^^

Show :ref:`echofilter<echofilter CLI>` documentation and all possible
arguments.

.. code-block:: powershell

    echofilter --help

version
^^^^^^^

Show program's version number.

.. code-block:: powershell

    echofilter --version


list checkpoints
^^^^^^^^^^^^^^^^

Show the available model checkpoints and exit.

.. code-block:: powershell

    echofilter --list-checkpoints

list colours
^^^^^^^^^^^^

List the available (main) colour options for lines. The palette can be
viewed at https://matplotlib.org/gallery/color/named_colors.html

.. code-block:: powershell

    echofilter --list-colors

List all available colour options (very long list) including the XKCD
colour palette of 954 colours, which can be viewed at
https://xkcd.com/color/rgb/

.. code-block:: powershell

    echofilter --list-colors full


Command line interface primer
-----------------------------

In this section, we provide some pointers for users new to using the
command prompt.

Spaces in file names
~~~~~~~~~~~~~~~~~~~~

Running commands on files with spaces in their file names is
problematic. This is because spaces are used to separate arguments from
each other, so for instance::

    command-name some path with spaces

is actually running the command ``command-name`` with four arguments:
``some``, ``path``, ``with``, and ``spaces``.

You can run commands on paths containing spaces by encapsulating the path
in quotes (either single, ``'``, or double ``"`` quotes), so it becomes
a single string. For instance::

    command-name "some path with spaces"

In the long run, you may find it easier to change your directory
structure to not include any spaces in any of the names of directories
used for the data.

Trailing backslash
~~~~~~~~~~~~~~~~~~

The backslash (``\``) character is an
`escape character <https://en.wikipedia.org/wiki/Escape_character>`__,
used to give alternative meanings to symbols with special meanings.
For example, the quote characters ``"`` and ``'`` indicate the start or end
of a string but can be escaped to obtain a literal quote character.

On Windows, ``\`` is also used to denote directories. This overloads
the ``\`` symbol with multiple meanings. For this reason, you should not
include a trailing ``\`` when specifying directory inputs. Otherwise, if you
provide the path in quotes, an input of ``"some\path\"`` will not be
registered correctly, and will include a literal ``"`` character, with
the end of the string implicitly indicated by the end of the input.
Instead, you should use ``"some\path"``.

Alternatively, you could escape the backslash character to ensure
it is a literal backslash with ``"some\path\\"``, or use a forward
slash with ``"some/path/"`` since :ref:`echofilter<echofilter CLI>`
also understands forward slashes as a directory separator.

Argument types
~~~~~~~~~~~~~~

Commands at the command prompt can take arguments. There are a couple of
types of arguments:

-  mandatory, positional arguments

-  optional arguments

   -  shorthand arguments which start with a single hyphen (``-v``)

   -  longhand arguments which start with two hyphens (``--verbose``)

For :ref:`echofilter<echofilter CLI>`, the only positional argument is
the path to the file(s) or directory(ies) to process.

Arguments take differing numbers of parameters.
For :ref:`echofilter<echofilter CLI>` the positional argument (files to
process) must have at least one entry and can contain as many as you like.

Arguments which take zero parameters are sometimes called flags, such as
the flag ``--skip-existing``

Shorthand arguments can be given together, such as ``-vvfsn``, which is the
same as all of ``--verbose --verbose --force --skip --dry-run``.

In the help documentation, arguments which require at least one value to
be supplied have text in capitals after the argument, such as
``--suffix-var SUFFIX_VAR``. Arguments which have synonyms are listed
together in one entry, such as ``--skip-existing``, ``--skip``, ``-s``; and
``--output-dir OUTPUT_DIR``, ``-o OUTPUT_DIR``. Arguments where a variable is
optional have it shown in square brackets, such as
``--cache-csv [CSV_DIR]``. Arguments which accept a variable number of values
are shown such as ``--extension SEARCH_EXTENSION [SEARCH_EXTENSION ...]``.
Arguments whose value can only take one of a set number of options are
shown in curly brackets, such as ``--facing {downward,upward,auto}``.

Long lines for commands at the command prompt can be broken up into
multiple lines by using a continuation character. On Windows, the line
continuation character is ``^``, the caret symbol. When specifying optional
arguments requires that the command be continued on the next line,
finish the current line with ``^`` and begin the subsequent line at the
start of the next line.

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


Issues
------

Known issues
~~~~~~~~~~~~

There is a memory leak somewhere in :ref:`echofilter<echofilter CLI>`.
Consequently, its memory usage will slowly rise while it is in use.
When processing a very large number of files, you may eventually run out
of memory. In this case, you must close the Command Window (to release
the memory). You can then restart :ref:`echofilter<echofilter CLI>`
from where it was up to, or run the same command with the ``--skip``
argument, to process the rest of the files.

Troubleshooting
~~~~~~~~~~~~~~~

-  If you run out of memory after processing a single file, consider
   closing other programs to free up some memory. If this does not help,
   report the issue.

-  If you run out of memory when part way through processing a large
   number of files, restart the process by running the same command with
   the ``--skip`` argument. See the known issues section above.

-  If you have a problem using a :term:`checkpoint` for the first time:

   -  check your internet connection

   -  check that you have at least 100MB of hard-drive space available
      to download the new checkpoint

   -  if you have an error saying the checkpoint was not recognised,
      check the spelling of the checkpoint name.

-  If you receive error messages about writing or loading
   :term:`CSV files<CSV>` automatically generated from
   :term:`EV files<EV file>`, check that sufficient hard-drive space is
   available.

-  If you experience problems with operations which occur inside
   :term:`Echoview`, please re-run the code but manually open Echoview
   before running :ref:`echofilter<echofilter CLI>`. This will leave the
   Echoview window open and you will be able to read the error message
   within Echoview.

Reporting an issue
~~~~~~~~~~~~~~~~~~

If you experience a problem with :term:`echofilter`, please report it by
`creating a new issue on our repository <https://github.com/DeepSenseCA/echofilter/issues/new>`__
if possible, or otherwise by emailing scottclowe@gmail.com.

Please include:

-  Which version of echofilter which you are using. This is found by running
   the command ``echofilter --version``.

-  The operating system you are using.
   On Windows 10, system information information can be found by going to
   ``Start > Settings > System > About``.
   Instructions for other Windows versions can be
   `found here <https://support.microsoft.com/help/13443/windows-which-version-am-i-running>`__.

-  If you are using Echoview integration, your Echoview version number
   (which can be found by going to ``Help > About`` in Echoview), and
   whether you have and are using an Echoview HASP USB dongle.

-  What you expected to happen.

-  What actually happened.

-  All steps/details necessary to reproduce the issue.

-  Any error messages which were produced.

.. highlight:: python
