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
