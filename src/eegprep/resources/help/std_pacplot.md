# STD_PACPLOT - Plot STUDY PAC caches

`std_pacplot` reads EEGPrep-owned PAC caches through `std_readpac` and plots
PAC magnitude over time and amplitude frequency.

Use `std_pac` first to compute channel PAC caches. Range options such as
`timerange` and `freqrange` are applied by the cache reader, so scripted and
console calls see the same data that the plot displays.

See also: STD_PAC, STD_READPAC
