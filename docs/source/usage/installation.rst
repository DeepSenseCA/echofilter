Installation
------------

.. highlight:: winbatch

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
      `echofilter-executable-1.2.0.zip <https://github.com/DeepSenseCA/echofilter/releases/download/1.2.0/echofilter-executable-1.2.0.zip>`__

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

.. highlight:: python

.. raw:: latex

    \clearpage
