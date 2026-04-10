"""Minimal EEGPrep pipeline: resample + highpass filter only.

Processes a BIDS EEG dataset and writes a BIDS-compliant derivative.
Designed to run standalone or inside a Docker container.

Usage:
    python bids_minimal_preproc.py --input /path/to/bids_dataset
    python bids_minimal_preproc.py --input /path/to/bids_dataset --output /path/to/output
    python bids_minimal_preproc.py --input /path/to/bids_dataset --srate 100 --highpass 0.5
"""

import argparse
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
)


def main():
    parser = argparse.ArgumentParser(
        description='Minimal EEGPrep preprocessing: resample and highpass filter.')
    parser.add_argument('--input', required=True,
                        help='Path to the BIDS dataset root directory')
    parser.add_argument('--output', default=None,
                        help='Output directory (default: {input}/derivatives/eegprep)')
    parser.add_argument('--srate', type=float, default=100.0,
                        help='Target sampling rate in Hz (default: 100)')
    parser.add_argument('--highpass', type=float, default=0.5,
                        help='Highpass filter cutoff in Hz (default: 0.5)')
    parser.add_argument('--subjects', nargs='*', default=None,
                        help='Subject IDs or 0-based indices to process (default: all)')
    parser.add_argument('--jobs', default='1CPU',
                        help='Parallelism spec, e.g. "1CPU" or "4GB" (default: 1CPU)')
    parser.add_argument('--desc', default='',
                        help='desc- label for output files (default: none)')
    parser.add_argument('--report-dir', default='code/reports',
                        help='Directory for report files relative to output (default: code/reports)')
    args = parser.parse_args()

    from eegprep import bids_preproc

    # Transition band: [cutoff, cutoff*2] Hz for the FIR highpass
    hp_cutoff = args.highpass
    hp_transition = (hp_cutoff, hp_cutoff * 2)

    output_dir = args.output or '{root}/derivatives/eegprep'

    subjects = ()
    if args.subjects:
        # Try to parse as integers (0-based indices), otherwise use as string IDs
        parsed = []
        for s in args.subjects:
            try:
                parsed.append(int(s))
            except ValueError:
                parsed.append(s)
        subjects = parsed

    bids_preproc(
        args.input,
        OutputDir=output_dir,
        SamplingRate=args.srate,
        Highpass=hp_transition,
        # Disable all cleaning stages
        ChannelCriterion='off',
        LineNoiseCriterion='off',
        BurstCriterion='off',
        WindowCriterion='off',
        FlatlineCriterion='off',
        # Disable all optional processing
        OnlyChannelsWithPosition=False,
        WithICA=False,
        WithICLabel=False,
        WithInterp=False,
        CommonAverageReference=False,
        # Output naming
        FinalDesc=args.desc,
        ReportDir=args.report_dir,
        # Run config
        Subjects=subjects,
        ReservePerJob=args.jobs,
        SkipIfPresent=True,
    )


if __name__ == '__main__':
    main()
