# PAC - Phase-amplitude coupling

`pac` computes epoched phase-amplitude coupling using EEGPrep's standalone
time-frequency backend.

Inputs are time-by-trial arrays for the amplitude and phase signals plus the
sampling rate. The result contains a complex PAC grid, output times, amplitude
frequencies, phase frequencies, and the single-trial time-frequency
decompositions used for the calculation.

Bootstrap significance and EEGLAB's `latphase` histogram mode are explicit
future backend boundaries and raise `NotImplementedError`.

See also: PAC_CONT, STD_PAC
