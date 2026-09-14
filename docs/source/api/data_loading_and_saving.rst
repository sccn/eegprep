.. _api_data_loading_and_saving:

Data Loading and Saving
=======================

Readers and writers for EEG datasets, channel locations, and MNE interchange.

.. autosummary::
   :toctree: generated/

   eegprep.biosig2eeglabevent
   eegprep.cart2topo
   eegprep.chancenter
   eegprep.convertlocs
   eegprep.coregister
   eegprep.decodechan
   eegprep.eeg2mne
   eegprep.eeg_chaninds
   eegprep.eeg_decodechan
   eegprep.eeg_eeg2mne
   eegprep.eeg_mne2eeg
   eegprep.eeg_mne2eeg_epochs
   eegprep.eeg_matchchans
   eegprep.eeg_mergechan
   eegprep.eeg_mergelocs
   eegprep.floatread
   eegprep.floatwrite
   eegprep.loadcnt
   eegprep.loadeeg
   eegprep.loadtxt
   eegprep.loadset
   eegprep.MemmapData
   eegprep.mmo
   eegprep.getchanlist
   eegprep.mne2eeg
   eegprep.mne2eeg_epochs
   eegprep.openbdf
   eegprep.parsetxt
   eegprep.pop_loadbv
   eegprep.pop_loadcnt
   eegprep.readbdf
   eegprep.read_erpss
   eegprep.readeetraklocs
   eegprep.readegilocs
   eegprep.readelp
   eegprep.readlocs
   eegprep.readneurodat
   eegprep.readtxtfile
   eegprep.saveset
   eegprep.snapread
   eegprep.shortread
   eegprep.writelocs
   eegprep.writegdf

``readegilocs`` includes packaged EGI montages for 32/33, 64/65, 128/129,
and 256/257-channel nets.

``loadcnt`` and ``pop_loadcnt`` read Neuroscan CNT data without an EEGLAB
checkout. They support 16- and 32-bit recordings, channel-blocked storage,
microvolt calibration, partial reads, event tables, and ``.fdt``-backed data.
ANT Neuro CNT is a separate format and is not accepted by this reader.

BrainVision recordings
----------------------

Load a BrainVision Data Exchange recording through its header. The companion
data and marker files are resolved from the header and must remain beside it.

.. code-block:: python

   from eegprep import pop_loadbv

   EEG = pop_loadbv("subject01.vhdr")
   EEG = pop_loadbv("/data/session", "subject01.vhdr", [1001, 5000], [1, 2, 8])

``srange`` and ``chans`` use EEGLAB-compatible 1-based indexing; a two-value
sample range is inclusive. Binary and ASCII, multiplexed and vectorized data
are supported. Voltage data is normalized to microvolts while the original
unit and resolution remain in each channel's ``bvunit`` and ``bvresolution``
fields. Use ``metadata=True`` to inspect dimensions, channels, and events
without loading samples. Uniform marker-based or fixed-time segments are
returned as trials rather than a flattened continuous array.

``read_erpss`` reads uncompressed and delta-compressed ERPSS ``.RAW`` and
``.RDF`` recordings without a compiled MEX extension. It preserves channel
labels and 1-based event sample offsets, and supports both little- and
big-endian files.
