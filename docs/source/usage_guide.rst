Usage Guide
###########

Authors
    Scott C. Lowe, Louise McGarry

.. highlight:: winbatch

.. raw:: latex

    \clearpage


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

**Disclaimer**

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


.. toctree::
    :maxdepth: 3
    :caption: Contents:

    guide/installation
    guide/command_line_primer
    guide/quick_start
    guide/inference_steps
    guide/models
    guide/issues
    guide/glossary


.. raw:: latex

    \clearpage

.. highlight:: python
