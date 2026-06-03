# EEGPrep Remaining EEGLAB Migration Gap Audit

Audit date: 2026-06-03  
Branch audited: `fix/eeglab-session-history-progress` / PR #128  
Reference: vendored EEGLAB source under `src/eegprep/eeglab/`

## Scope

This note compares the current EEGPrep stacked branch against the vendored
EEGLAB source tree. The goal is to capture what remains missing in EEGPrep for
the porting project.

This is not a strict one-file-to-one-file checklist. Some EEGLAB MATLAB helpers
are intentionally consolidated into one Python module, and some MATLAB-only
GUI/runtime helpers do not map cleanly to standalone Python. Where that matters,
the note calls it out explicitly.

## Current State

The main EEGPrep menu surface is no longer the primary migration gap on this
branch:

- `src/eegprep/functions/guifunc/menu_placeholders.py` has an empty placeholder
  inventory.
- Tests assert that every main-window action is implemented, classified, or
  intentionally handled.
- EEGBrowser/eegplot-style scrolling workflows are no longer marked as excluded.
- Bundled in-repo plugin surfaces are represented: `clean_rawdata`, `ICLabel`,
  `firfilt`, `EEG_BIDS`, and `dipfit`.

The remaining gap is mostly deeper EEGLAB parity: long-tail function coverage,
option-level behavior inside implemented functions, MATLAB helper ecosystems,
statistics/time-frequency/STUDY depth, file-format edge cases, and external
plugin ecosystem coverage.

## High-Confidence Remaining Gaps

### 1. Long-Tail EEGLAB Helper Coverage

EEGLAB still has many helper functions that have no same-name EEGPrep runtime
module. Some are MATLAB compatibility utilities or development helpers, but many
support real EEGLAB user workflows.

Approximate current source counts:

| Area | EEGLAB MATLAB files | EEGPrep Python files | Missing by same-name check |
| --- | ---: | ---: | ---: |
| `adminfunc` | 68 | 14 | 60 |
| `guifunc` | 9 | 15 | 6 |
| `miscfunc` | 111 | 6 | 111 |
| `popfunc` | 131 | 90 | 59 |
| `sigprocfunc` | 115 | 17 | 99 |
| `statistics` | 15 | 0 | 15 |
| `studyfunc` | 133 | 32 | 105 |
| `timefreqfunc` | 25 | 2 | 23 |

False-positive examples from the same-name check:

- `eegplot2event`, `eegplot2trial`, and `trial2eegplot` are implemented inside
  `src/eegprep/functions/sigprocfunc/eegplot.py` rather than separate files.
- EEGLAB's `adminfunc/pop_rejmenu.m` maps to EEGPrep's
  `functions/popfunc/pop_rejmenu.py`.
- Some EEGLAB GUI wrappers such as `questdlg2`/`warndlg2` are not needed as
  one-for-one Python files because EEGPrep uses Qt-native helpers.

### 2. Missing Or Legacy `pop_*` Entry Points

EEGPrep has the modern/main menu `pop_*` workflows, but not every EEGLAB
same-name `pop_*` wrapper or legacy alias. Missing by same-name check:

- `pop_averef`
- `pop_biosig16`
- `pop_biosig16ying`
- `pop_chancenter`
- `pop_chancoresp`
- `pop_compareerps`
- `pop_crossf`
- `pop_fileiodir`
- `pop_findmatchingcomps`
- `pop_fusechanrej`
- `pop_icathresh`
- `pop_importegimat`
- `pop_loadbci`
- `pop_readegi`
- `pop_readlocs`
- `pop_readsegegi`
- `pop_rejchanspec`
- `pop_snapread`
- `pop_timef`
- `pop_topochansel`
- `pop_writelocs`

Some may be intentionally superseded by newer EEGPrep APIs, but EEGLAB users may
still expect these names and workflows.

### 3. Implemented Functions With Unsupported EEGLAB Options

Several user-facing functions exist but still reject some EEGLAB options with a
clear `NotImplementedError`.

Known examples:

- `pop_newtimef`
  - unsupported: `timewarp`, `timewarpms`, `timewarpidx`, `rboot`, `pboot`,
    `erspboot`, `itcboot`
  - bootstrap significance is not implemented
  - baseline normalization modes are not implemented
  - curve plots are not implemented
- `pop_newcrossf`
  - unsupported: `rboot`, `boottype`, `baseboot`, `condboot`, `shuffle`,
    `subitc`, `amplag`
  - bootstrap significance is not implemented
- `pop_erpimage`
  - event-field sorting, event-type sorting, event-window sorting, event
    alignment, phase sorting, ITC/coherence overlays, spectrum inset, baseline
    amplitude limits, amplitude image mode, and free-form ERP-image options are
    not fully supported
- `pop_comperp`
  - significance highlighting, standard-deviation displays, all-ERP display
    variants, and some average/difference display variants are not fully
    supported
- `pop_editset`
  - file/workspace expressions for `data`, ICA matrices, and some channel
    location paths are not fully supported
  - history serialization for mapping values and channel-location structures is
    incomplete
- `pop_export`
  - `expr` export filtering is not supported
- `pop_epoch`
  - loading EEG data from a filename in `EEG["data"]` is not implemented
- `pop_eegfilt`
  - legacy FFT filtering path is not implemented; users are directed to
    `pop_eegfiltnew`
- `pop_runica`
  - supports `runica`, `picard`, and AMICA-style routes; EEGLAB algorithms such
    as JADER, SOBI, and FastICA are not ported
- `headplot`
  - `plotmeshonly` preview and setup with original unprojected locations are not
    implemented
- `coregister`
  - only standalone `traditional` and `globalrescale` alignment methods are
    supported
- `clean_asr`
  - Riemannian ASR mode is not implemented

### 4. STUDY And Group-Level Depth

EEGPrep has STUDY support, but EEGLAB's `studyfunc` ecosystem is still much
larger. Missing same-name examples include:

- LIMO and statistics integration: `pop_limo`, `pop_limoresults`, `std_limo`,
  `std_limodesign`, `std_limoresults`, `std_readfilelimo`
- design and variable helpers: `pop_addindepvar`, `pop_importgroupvar`,
  `pop_listfactors`, `std_addvarlevel`, `std_builddesignmat`,
  `std_rebuilddesign`, `std_findgroupvars`, `std_saveindvar`
- measure readers/writers: `std_readerp`, `std_readersp`, `std_readitc`,
  `std_readspec`, `std_readtopo`, `std_readpac`, `std_savedat`,
  `std_readdatafield`-style behavior
- plotting helpers: `std_plot`, `std_plotcurve`, `std_plottf`, `std_topoplot`,
  `std_chantopo`, `std_erpimageplot`, `std_propplot`
- clustering details: `std_apcluster`, `std_centroid`, `std_dipoleclusters`,
  `std_findoutlierclust`, `optimal_kmeans`, `robust_kmeans`
- dataset consistency and file checks: `std_checkfiles`,
  `std_checkdatasession`, `std_uniformfiles`, `std_uniformsetinds`

### 5. Statistics Package

EEGLAB has a dedicated `functions/statistics` folder. EEGPrep does not yet have
a same-name statistics package.

Missing same-name functions:

- `anova1_cell`
- `anova1rm_cell`
- `anova2_cell`
- `anova2rm_cell`
- `concatdata`
- `corrcoef_cell`
- `fdr`
- `stat_surrogate_ci`
- `stat_surrogate_pvals`
- `statcond`
- `statcondfieldtrip`
- `surrogdistrib`
- `teststat`
- `ttest2_cell`
- `ttest_cell`

### 6. Time-Frequency Internals

EEGPrep has `newtimef` and `newcrossf`, but the full EEGLAB time-frequency
stack is not migrated.

Missing same-name functions:

- `angtimewarp`
- `bootstat`
- `correct_mc`
- `correctfit`
- `crossf`
- `dftfilt`
- `dftfilt2`
- `dftfilt3`
- `newtimefbaseln`
- `newtimefitc`
- `newtimefpowerunit`
- `newtimeftrialbaseln`
- `pac`
- `pac_cont`
- `rsadjust`
- `rsfit`
- `rsget`
- `rspdfsolv`
- `rspfunc`
- `tf_cycle_calc`
- `timef`
- `timefreq`
- `timewarp`

### 7. File Format And Channel-Location Long Tail

Core load/save/import/export paths exist, but EEGLAB has many old or
format-specific readers/writers and channel-location helpers that are not fully
mirrored.

Missing or incomplete areas include:

- EGI-specific direct import helpers: `pop_readegi`, `pop_readsegegi`,
  `readegi`, `readegihdr`, `readegilocs`
- BCI/snapread helpers: `pop_loadbci`, `pop_snapread`, `snapread`
- location file helpers: `pop_readlocs`, `pop_writelocs`, `readlocs`,
  `writelocs`, `readelp`, `readeetraklocs`, `readneurolocs`
- channel geometry workflows: `pop_chancenter`, `pop_chancoresp`,
  `chancenter`, `convertlocs`, `forcelocs`, `lookupchantemplate`,
  `plotchans3d`
- legacy BIOSIG path helpers: `pop_biosig16`, `biosigpathfirst`,
  `biosigpathlast`

### 8. ICA Algorithm And Rejection Long Tail

EEGPrep has the main ICA path and common rejection paths, but not all EEGLAB
algorithm choices and helper routes.

Missing or partial areas include:

- ICA algorithms/helpers: `binica`, `jader`, `sobi`, `acsobiro`, `fastif`,
  `runica_ml`, `runica_ml2`, `runica_mlb`, `runicalowmem`, `runicatest`
- rejection helpers: `eegthresh`, `entropy_rej`, `jointprob`, `kurt`,
  `rejkurt`, `rejtrend`, `rejstatepoch`, `realproba`
- ICA/component helper calculations: `compvar`, `eeg_getica`, `eeg_oldica`,
  `eeg_pv`, `eeg_pvaf`, `icaact`, `icaproj`, `icavar`
- user-facing wrappers: `pop_icathresh`, `pop_rejchanspec`,
  `pop_findmatchingcomps`, `pop_fusechanrej`

### 9. MATLAB Memory-Mapped And Object Infrastructure

EEGLAB has MATLAB class-style folders that are not fully ported:

- `@eegobj`: only minimal EEGPrep object support exists
- `@memmapdata`: no one-for-one EEGPrep implementation
- `@mmo`: no one-for-one EEGPrep implementation

Missing same-name memory/object helpers include:

- `@eegobj/display`, `fieldnames`, `horzcat2`, `isfield`, `isstruct`,
  `length`, `orderfields`, `rmfield`, `simpletest`, `subsasgn`, `subsref`
- `@memmapdata/display`, `double`, `end`, `isnumeric`, `length`,
  `memmapdata`, `ndims`, `reshape`, `size`, `subsasgn`, `subsref`, `sum`
- `@mmo/binaryopp`, `bsxfun`, `changefile`, `checkcopies_local`,
  `checkworkspace`, `ctranspose`, `display`, `double`, `end`, `fft`,
  `isnumeric`, `length`, `mmo`, `ndims`, `permute`, `reshape`, `size`,
  `subsasgn`, `subsasgn_old`, `subsref`, `sum`, `transpose`, `unitaryopp`,
  `var`

This matters for full storedisk/memory-mapped parity. EEGPrep currently favors
explicit Python session/dataset semantics.

### 10. Bundled Plugin Depth

The bundled plugin folders are represented, but not one-for-one.

#### clean_rawdata

EEGPrep implements the core clean_rawdata/ASR path:

- `asr_calibrate`
- `asr_process`
- `clean_artifacts`
- `clean_asr`
- `clean_channels`
- `clean_channels_nolocs`
- `clean_drifts`
- `clean_flatlines`
- `clean_windows`
- `pop_clean_rawdata`

Not fully represented:

- Riemannian ASR MATLAB paths: `asr_calibrate_r`, `asr_process_r`
- `vis_artifacts` depth and exact EEGLAB visualization behavior
- several private MATLAB helper equivalents
- the bundled Manopt dependency tree is not ported into EEGPrep as Python

#### firfilt

Implemented core EEGPrep files include:

- `firws`
- `firwsord`
- `pop_eegfiltnew`
- `pop_firma`
- `pop_firpm`
- `pop_firws`

Missing same-name firfilt helpers include:

- `findboundaries`
- `fir_filterdcpadded`
- `firfilt`
- `firfiltreport`
- `firfiltsplit`
- `invfirwsord`
- `invkaiserbeta`
- `kaiserbeta`
- `minphaserceps`
- `plotfresp`
- `pop_firpmord`
- `pop_firwsord`
- `pop_kaiserbeta`
- `pop_xfirws`
- `windows`

#### ICLabel / Viewprops

Implemented core EEGPrep files include:

- `ICL_feature_extractor`
- `eeg_autocorr`
- `eeg_autocorr_fftw`
- `eeg_autocorr_welch`
- `eeg_icflag`
- `eeg_rpsd`
- `iclabel`
- `iclabel_net`
- `pop_icflag`
- `pop_iclabel`
- `pop_prop_extended`
- `pop_viewprops`

Remaining gaps include:

- `eeg_icalabelstat`
- full MatConvNet MATLAB runtime/tooling parity
- alternate ICLabel network artifacts such as beta/lite variants unless
  explicitly packaged
- exact MATLAB viewprops helper parity where EEGPrep uses native Python/Qt
  replacements

#### DIPFIT

Implemented main pop-level EEGPrep files include:

- `pop_dipfit_gridsearch`
- `pop_dipfit_headmodel`
- `pop_dipfit_loreta`
- `pop_dipfit_nonlinear`
- `pop_dipfit_settings`
- `pop_dipplot`
- `pop_leadfield`
- `pop_multifit`

Missing same-name DIPFIT helpers include:

- `dipfit_gridsearch`
- `dipfit_nonlinear`
- `dipfit_reject`
- `dipplot`
- `electroderealign`
- `headcoordinates`
- `load_afni_atlas`
- `mni2tal`
- `plot3dmeshalign`
- `pop_dipfit_batch`
- `pop_dipfit_manual`
- private transform helpers: `rigidbody`, `rotate`, `scale`, `translate`,
  `warp_apply`, `warp_error`, `warp_optim`
- several FieldTrip/atlas conversion helpers

### 11. External Plugin Ecosystem

EEGPrep now has extension infrastructure, but the actual EEGLAB external plugin
ecosystem is not ported. This is expected and should stay separate from core
EEGPrep migration.

Examples of external-plugin families that should not be assumed present unless
implemented as EEGPrep extensions:

- ERPLAB
- LIMO
- SIFT
- NFT
- MFF/importer plugin ecosystems
- lab-specific processing plugins

### 12. Docs, Tutorials, And Help Corpus

EEGPrep has EEGPrep-owned help resources, but EEGLAB's full help/tutorial corpus
is not fully mirrored. Remaining work includes:

- full pophelp-level option documentation for every implemented `pop_*`
  function
- tutorial workflows comparable to EEGLAB's docs
- full examples for STUDY, EEGBrowser, extension authoring, file I/O, and
  time-frequency/statistics workflows
- parity evidence links or notes for every major GUI workflow

## Full Missing-By-Name Inventory

The following lists are generated from same-name comparisons between vendored
EEGLAB MATLAB files and EEGPrep Python modules. They are useful for audit
coverage, but they should not be treated as direct implementation instructions
without checking whether the behavior is already consolidated elsewhere in
Python.

### `adminfunc`

`abouteeglab`, `biosigpathfirst`, `biosigpathlast`, `eeg_cache`,
`eeg_checkchanlocs`, `eeg_eval`, `eeg_getdatact`, `eeg_getversion`,
`eeg_global`, `eeg_helpadmin`, `eeg_helpgui`, `eeg_helphelp`, `eeg_helpmenu`,
`eeg_helpmisc`, `eeg_helppop`, `eeg_helpsigproc`, `eeg_helpstatistics`,
`eeg_helpstudy`, `eeg_helptimefreq`, `eeg_hist`, `eeg_optionsbackup`,
`eeg_readoptions`, `eeglab_error`, `eeglab_execmenu`, `eeglab_new`,
`eeglab_options`, `eeglab_update`, `eeglab_warning`, `error_bc`, `gethelpvar`,
`getkeyval`, `gettext`, `hlp_argstruct2linearcell`, `intersect_bc`, `is_sccn`,
`iseeglabdeployed`, `ismatlab`, `ismember_bc`, `plugin_askinstall`,
`plugin_extract`, `plugin_getweb`, `plugin_install`, `plugin_movepath`,
`plugin_remove`, `plugin_search`, `plugin_status`, `plugin_uifilter`,
`plugin_uiupdate`, `plugin_urlread`, `plugin_urlreadwrite`, `plugin_urlsize`,
`plugin_urlwrite`, `pop_rejmenu`, `pop_stdwarn`, `removepath`, `setdiff_bc`,
`troubleshooting_data_formats`, `union_bc`, `unique_bc`, `vararg2str`

### `guifunc`

`errordlg2`, `finputcheck`, `inputdlg2`, `questdlg2`, `supergui`, `warndlg2`

### `miscfunc`

`abspeak`, `arrow`, `averef`, `brainstorm2eeglab`, `caliper`, `chanproj`,
`cleanvarname`, `compareeeglabdistrib`, `compdsp`, `compheads`,
`compile_eeglab`, `compmap`, `compplot`, `compsort`, `convolve`, `corrimage`,
`covary`, `crossfold`, `crossfreq`, `datlim`, `del2map`, `dendhier`,
`dendplot`, `detectmalware`, `difftopo`, `eeg_ms2f`, `eeg_time2prev`,
`eegdraw`, `eegdrawg`, `eegmovie`, `eegplotgold`, `eegplotold`, `eegplotsold`,
`envproj`, `erpregout`, `erpregoutfunc`, `eucl`, `fastregress`,
`fieldtrip2eeglab`, `fillcurves`, `findduplicatefunctions`, `gabor2d`,
`gauss`, `gauss2d`, `gauss3d`, `getallmenus`, `getallmenuseeglab`,
`getipsph`, `getmfilelist`, `gradmap`, `gradplot`, `headmovie`, `help2html2`,
`helpforexe`, `hist2`, `hungarian`, `icademo`, `imagescloglog`,
`imagesclogy`, `kmeans_st`, `laplac2d`, `lapplot`, `loadelec`, `loc_subsets`,
`logimagesc`, `loglike`, `logspec`, `make_timewarp`, `makeelec`,
`makehelpfiles`, `makehtml`, `mapcorr`, `matcorr`, `matperm`, `means`,
`nan_std`, `numdim`, `pcexpand`, `pcsquash`, `perminv`, `plotproj`, `promax`,
`qrtimax`, `read_rdf`, `readlocsold`, `replace_in_all_files`, `rmart`,
`rmsave`, `rotatematlab`, `runicalowmem`, `runicatest`, `runpca`, `runpca2`,
`scanfold`, `seemovie`, `setfont`, `shortread`, `show_events`, `testica`,
`textgui`, `tftopo`, `timefrq`, `topoimage`, `tutorial`,
`unique_cell_string`, `uniquef`, `upgma`, `varimax`, `varsort`, `vectdata`,
`zica`

### `popfunc`

`eeg_addnewevents`, `eeg_amplitudearea`, `eeg_boundarytype`, `eeg_chaninds`,
`eeg_context`, `eeg_countepochs`, `eeg_dipselect`, `eeg_epoch2continuous`,
`eeg_epochformat`, `eeg_eventformat`, `eeg_eventhist`, `eeg_eventtable`,
`eeg_eventtypes`, `eeg_getepochevent`, `eeg_getica`, `eeg_hedremoveunicode`,
`eeg_import`, `eeg_insertbound`, `eeg_insertboundold`, `eeg_isboundary`,
`eeg_laplac`, `eeg_latencyur`, `eeg_matchchans`, `eeg_mergechan`,
`eeg_mergelocs`, `eeg_mergelocs_diffstruct`, `eeg_oldica`, `eeg_pv`,
`eeg_pvaf`, `eeg_regepochs`, `eeg_rejmacro`, `eeg_rereject`,
`eeg_timeinterp`, `eeg_topoplot`, `eeg_uniformepochinfo`, `eeg_urlatency`,
`getchanlist`, `importevent`, `pop_averef`, `pop_biosig16`,
`pop_biosig16ying`, `pop_chancenter`, `pop_chancoresp`, `pop_compareerps`,
`pop_crossf`, `pop_fileiodir`, `pop_findmatchingcomps`, `pop_fusechanrej`,
`pop_icathresh`, `pop_importegimat`, `pop_loadbci`, `pop_readegi`,
`pop_readlocs`, `pop_readsegegi`, `pop_rejchanspec`, `pop_snapread`,
`pop_timef`, `pop_topochansel`, `pop_writelocs`

### `sigprocfunc`

`acsobiro`, `axcopy`, `binica`, `biosig2eeglab`, `biosig2eeglabevent`,
`blockave`, `cbar`, `celltomat`, `chancenter`, `changeunits`, `compvar`,
`condstat`, `convertlocs`, `copyaxis`, `dipoledensity`, `eegfilt`,
`eegfiltfft`, `eegplot2event`, `eegplot2trial`, `eegplot_readkey`,
`eegplotlegacy`, `eegthresh`, `entropy_rej`, `env`, `eventalign`,
`eventlock`, `eyelike`, `fastif`, `floatread`, `floatwrite`, `forcelocs`,
`gettempfolder`, `icaact`, `icadefs`, `icaproj`, `icavar`, `imagesctc`,
`isscript`, `jader`, `jointprob`, `kmeanscluster`, `kurt`, `loadtxt`,
`lookupchantemplate`, `matsel`, `mattocell`, `metaplottopo`, `movav`,
`moveaxes`, `mri3dplot`, `nan_mean`, `parsetxt`, `phasecoher`, `plotchans3d`,
`plotcurve`, `plotdata`, `ploterp`, `plotmesh`, `plotsphere`, `posact`,
`projtopo`, `qqdiagram`, `rand_permutation`, `readedf`, `readeetraklocs`,
`readegi`, `readegihdr`, `readegilocs`, `readelp`, `readlocs`,
`readneurodat`, `readneurolocs`, `readtxtfile`, `realproba`, `rejkurt`,
`rejstatepoch`, `rejtrend`, `runica_ml`, `runica_ml2`, `runica_mlb`,
`sbplot`, `shuffle`, `slider`, `snapread`, `sobi`, `spec`, `sph2topo`,
`spher`, `spherror`, `strmultiline`, `textsc`, `timefdetails`,
`topo2sph`, `transformcoords`, `trial2eegplot`, `uigetfile2`, `uiputfile2`,
`writeeeg`, `writelocs`

### `statistics`

`anova1_cell`, `anova1rm_cell`, `anova2_cell`, `anova2rm_cell`, `concatdata`,
`corrcoef_cell`, `fdr`, `stat_surrogate_ci`, `stat_surrogate_pvals`,
`statcond`, `statcondfieldtrip`, `surrogdistrib`, `teststat`, `ttest2_cell`,
`ttest_cell`

### `studyfunc`

`compute_ersp_times`, `eeglabciplot`, `gethashcode`, `neural_net`,
`optimal_kmeans`, `pop_addindepvar`, `pop_dipparams`, `pop_erpimparams`,
`pop_erpparams`, `pop_erspparams`, `pop_importgroupvar`, `pop_limo`,
`pop_limoresults`, `pop_listfactors`, `pop_specparams`, `pop_statparams`,
`robust_kmeans`, `std_addvarlevel`, `std_apcluster`, `std_builddesignmat`,
`std_cell2setcomps`, `std_cell2table`, `std_centroid`, `std_changroup`,
`std_chaninds`, `std_chantopo`, `std_checkconsist`, `std_checkdatasession`,
`std_checkdesign`, `std_checkfiles`, `std_clustmaxelec`, `std_combtrialinfo`,
`std_comppol`, `std_custom`, `std_detachplots`, `std_dipoleclusters`,
`std_dipplot`, `std_erp`, `std_erpimage`, `std_erpimageplot`, `std_ersp`,
`std_figtitle`, `std_filecheck`, `std_fileinfo`, `std_findgroupvars`,
`std_findoutlierclust`, `std_findsameica`, `std_getdataset`, `std_getindvar`,
`std_gettrialsind`, `std_indvarmatch`, `std_interp`, `std_limo`,
`std_limodesign`, `std_limoerase`, `std_limoresults`, `std_lm_getvars`,
`std_lm_seteegfields`, `std_loadalleeg`, `std_maketrialinfo`,
`std_mergeruns`, `std_movie`, `std_pac`, `std_pacplot`, `std_plot`,
`std_plotcurve`, `std_plotdmat`, `std_plottf`, `std_precomp_worker`,
`std_prepare_neighbors`, `std_propplot`, `std_pvaf`, `std_readcustom`,
`std_readeegfield`, `std_readerp`, `std_readerpimage`, `std_readersp`,
`std_readfile`, `std_readfilelimo`, `std_readitc`, `std_readpac`,
`std_readspec`, `std_readspecgram`, `std_readtopo`, `std_readtopoclust`,
`std_rebuilddesign`, `std_renamestudyfiles`, `std_reset`,
`std_rmalldatafields`, `std_rmdat`, `std_savedat`, `std_saveindvar`,
`std_selcomp`, `std_selectdataset`, `std_selsubject`, `std_serialize`,
`std_spec`, `std_specgram`, `std_stat`, `std_substudy`, `std_topo`,
`std_topoplot`, `std_uniformfiles`, `std_uniformsetinds`, `toporeplot`

### `timefreqfunc`

`angtimewarp`, `bootstat`, `correct_mc`, `correctfit`, `crossf`, `dftfilt`,
`dftfilt2`, `dftfilt3`, `newtimefbaseln`, `newtimefitc`,
`newtimefpowerunit`, `newtimeftrialbaseln`, `pac`, `pac_cont`, `rsadjust`,
`rsfit`, `rsget`, `rspdfsolv`, `rspfunc`, `tf_cycle_calc`, `timef`,
`timefreq`, `timewarp`

## Recommended Next Migration Planning Step

The next useful planning artifact should be a generated parity matrix with one
row per EEGLAB public or semi-public function:

1. EEGLAB function path
2. EEGPrep equivalent path, if any
3. status: implemented / consolidated / partial / intentionally replaced /
   not ported
4. user-facing surface: menu / console / Python API / internal helper
5. unsupported options or edge cases
6. test coverage
7. MATLAB parity coverage
8. GUI visual parity coverage, when relevant

That matrix would avoid double-counting consolidated Python ports while making
the remaining porting work concrete enough to split into issues.
