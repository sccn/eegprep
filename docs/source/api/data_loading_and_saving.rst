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
   eegprep.loadeeg
   eegprep.loadtxt
   eegprep.loadset
   eegprep.MemmapData
   eegprep.mmo
   eegprep.getchanlist
   eegprep.mne2eeg
   eegprep.mne2eeg_epochs
   eegprep.parsetxt
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

``readegilocs`` includes packaged EGI montages for 32/33, 64/65, 128/129,
and 256/257-channel nets.
