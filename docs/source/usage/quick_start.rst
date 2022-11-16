Quick Start
-----------

.. highlight:: winbatch

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
        --suffix "_echofilter-model" ^
        --color-surface "green" --thickness-surface 4 ^
        --color-nearfield "red" --thickness-nearfield 3

Processing a file with more output messages displayed in the terminal::

    echofilter "path/to/file_or_directory" --verbose

Processing a file and sending the output to a log file instead of the
terminal::

    echofilter "path/to/file_or_directory" -v > path/to/log_file.txt 2>&1


Config file
~~~~~~~~~~~

You may find that you are setting some parameters every time you call
echofilter, to consistently tweak the input or output processing settings in the
same way.
If this is the case, you can save these arguments to a configuration file,
and pass the configuration file to echofilter instead.

For example, if you have a file named ``"echofilter_params.cfg"`` with the following contents::

    --suffix "_echofilter-model"
    --color-surface "green"
    --thickness-surface 4
    --color-nearfield "red"
    --thickness-nearfield 3

then you can call echofilter with this configuration file as follows::

    echofilter "file_or_dir" --config "path/to/echofilter_params.cfg"

and it will use the parameters specified in your config file.
The format of the parameters is the same as they would be on the command prompt,
except in the config file each parameter must be on its own line.

The parameters in the config file also can be added to, or even overridden, at
the command prompt.
For example::

    echofilter "file_or_dir" --config "path/to/echofilter_params.cfg" --suffix "_test"

will use the ``--suffix "_test"`` argument from the command prompt instead of
the value set in the file ``"echofilter_params.cfg"``, but will still use the
other parameters as per the config file.

If you have several different workflows or protocols which you need to use,
you can create multiple config files corresponding to each of these workflows
and choose which one to use with the ``--config`` argument.

Common configuration options which you want to always be enabled can be set in
a special default config file in your home directory named ``".echofilter"``.
The path to your homedirectory, and hence to the default config file,
depends on your operating system.
On Windows it is typically ``"C:\Users\USERNAME\.echofilter"``, whilst on Linux
it is typically ``"/home/USERNAME/.echofilter"``, where USERNAME is replaced
with your username.
If it exists, the the default config file is always loaded everytime you run
echofilter.

If a config file is manually provided with the ``--config`` argument, any
parameters set in the manually provided config file override those in the
default config file ("~/.echofilter).

With the default verbosity settings, at the start of the inference routine
echofilter outputs the set of parameters it is using, and the source for each
of these parameters (command line, manual config file, default config file, or
program defaults).

You can read more about the `syntax for the configuration files here <https://goo.gl/R74nmi>`__.


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

.. code-block:: winbatch

    echofilter --help

version
^^^^^^^

Show program's version number.

.. code-block:: winbatch

    echofilter --version


list checkpoints
^^^^^^^^^^^^^^^^

Show the available model checkpoints and exit.

.. code-block:: winbatch

    echofilter --list-checkpoints

list colours
^^^^^^^^^^^^

List the available (main) colour options for lines. The palette can be
viewed at https://matplotlib.org/gallery/color/named_colors.html

.. code-block:: winbatch

    echofilter --list-colors

List all available colour options (very long list) including the XKCD
colour palette of 954 colours, which can be viewed at
https://xkcd.com/color/rgb/

.. code-block:: winbatch

    echofilter --list-colors full

.. highlight:: python

.. raw:: latex

    \clearpage
