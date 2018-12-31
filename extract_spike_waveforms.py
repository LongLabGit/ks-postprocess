"""Extract waveforms from recording file at detected spike times."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np
import os, os.path
from recordingio import load_recording

# -----------------------------------------------------------------------------
# Extract waveforms
# -----------------------------------------------------------------------------


# need: all spike times, spike clusters and cluster channels -> need templates!
# templates shape: (template, sample, channel)
# e.g. max channel where max(abs(samples)) is maximal (i.e. peak rectified amplitude)
# for each spike time, look up cluster and corresponding channels,
# then extract samples around spike time (+- 0.5 nsamples) only on channels
# on same shank and store in sequential file
# edge cases: pad with zeros
# also: implement hp-filtering of waveforms (padding required? don't think so)