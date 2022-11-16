Command line interface primer
-----------------------------

.. highlight:: winbatch

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

.. highlight:: python

.. raw:: latex

    \clearpage
