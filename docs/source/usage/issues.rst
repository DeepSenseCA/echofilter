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
