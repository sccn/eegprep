.. _pop_functions_api:

============================
EEGLAB-Style pop_ Wrappers
============================

This page lists the public ``pop_*`` wrappers exported by ``eegprep``. Use
``return_com=True`` on history-relevant wrappers when you need the replayable
EEGLAB-style command string. In normal Python, assign returned values
explicitly; in ``eegprep-console``, registered wrappers can update the shared
session automatically.

.. autosummary::
   :nosignatures:

   eegprep.pop_addindepvar
   eegprep.pop_adjustevents
   eegprep.pop_autorej
   eegprep.pop_averef
   eegprep.pop_biosig
   eegprep.pop_chancenter
   eegprep.pop_chancoresp
   eegprep.pop_chanedit
   eegprep.pop_chanevent
   eegprep.pop_chanplot
   eegprep.pop_chansel
   eegprep.pop_clean_rawdata
   eegprep.pop_clust
   eegprep.pop_clustedit
   eegprep.pop_comments
   eegprep.pop_compareerps
   eegprep.pop_comperp
   eegprep.pop_copyset
   eegprep.pop_crossf
   eegprep.pop_delset
   eegprep.pop_dipfit_gridsearch
   eegprep.pop_dipfit_headmodel
   eegprep.pop_dipfit_loreta
   eegprep.pop_dipfit_nonlinear
   eegprep.pop_dipfit_settings
   eegprep.pop_dipplot
   eegprep.pop_editeventfield
   eegprep.pop_editeventvals
   eegprep.pop_editoptions
   eegprep.pop_editset
   eegprep.pop_eegfilt
   eegprep.pop_eegfiltnew
   eegprep.pop_eegplot
   eegprep.pop_eegthresh
   eegprep.pop_envtopo
   eegprep.pop_epoch
   eegprep.pop_erpimage
   eegprep.pop_eventstat
   eegprep.pop_expevents
   eegprep.pop_expica
   eegprep.pop_export
   eegprep.pop_exportbids
   eegprep.pop_fileio
   eegprep.pop_fileio_brainvision_mat
   eegprep.pop_findmatchingcomps
   eegprep.pop_firma
   eegprep.pop_firpm
   eegprep.pop_firpmord
   eegprep.pop_firws
   eegprep.pop_firwsord
   eegprep.pop_fusechanrej
   eegprep.pop_headplot
   eegprep.pop_icathresh
   eegprep.pop_icflag
   eegprep.pop_iclabel
   eegprep.pop_importbids
   eegprep.pop_importdata
   eegprep.pop_importepoch
   eegprep.pop_importerplab
   eegprep.pop_importevent
   eegprep.pop_importgroupvar
   eegprep.pop_importpres
   eegprep.pop_interp
   eegprep.pop_jointprob
   eegprep.pop_kaiserbeta
   eegprep.pop_leadfield
   eegprep.pop_limo
   eegprep.pop_limoresults
   eegprep.pop_listfactors
   eegprep.pop_load_frombids
   eegprep.pop_loadbci
   eegprep.pop_loadset
   eegprep.pop_loadset_h5
   eegprep.pop_loadstudy
   eegprep.pop_mergeset
   eegprep.pop_multifit
   eegprep.pop_newcrossf
   eegprep.pop_newset
   eegprep.pop_newtimef
   eegprep.pop_plotdata
   eegprep.pop_plottopo
   eegprep.pop_preclust
   eegprep.pop_precomp
   eegprep.pop_prop
   eegprep.pop_prop_extended
   eegprep.pop_readlocs
   eegprep.pop_rejchan
   eegprep.pop_rejchanspec
   eegprep.pop_rejcont
   eegprep.pop_rejepoch
   eegprep.pop_rejkurt
   eegprep.pop_rejmenu
   eegprep.pop_rejspec
   eegprep.pop_rejtrend
   eegprep.pop_reref
   eegprep.pop_resample
   eegprep.pop_rmbase
   eegprep.pop_rmdat
   eegprep.pop_runica
   eegprep.pop_runscript
   eegprep.pop_saveh
   eegprep.pop_saveset
   eegprep.pop_savestudy
   eegprep.pop_select
   eegprep.pop_selectcomps
   eegprep.pop_selectevent
   eegprep.pop_signalstat
   eegprep.pop_snapread
   eegprep.pop_spectopo
   eegprep.pop_study
   eegprep.pop_studydesign
   eegprep.pop_studyerp
   eegprep.pop_studywizard
   eegprep.pop_subcomp
   eegprep.pop_timef
   eegprep.pop_timtopo
   eegprep.pop_topochansel
   eegprep.pop_topoplot
   eegprep.pop_viewprops
   eegprep.pop_writeeeg
   eegprep.pop_writelocs
   eegprep.pop_xfirws

Coverage Notes
==============

Major user-facing workflows are described in the user guide. Thin compatibility
wrappers, legacy import/export helpers, and plotting helpers are discoverable
here and through their function docstrings. Packaged GUI Help topics live under
``src/eegprep/resources/help`` and are opened by ``pophelp``.
