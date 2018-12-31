"""Generate small test data set."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np
import os, os.path
from recordingio import copy_recording_chunk

# -----------------------------------------------------------------------------
# Generate test data
# -----------------------------------------------------------------------------

# some global variables
spiketimes_name = 'spike_times.npy'
spiketimes_chunk_name = 'spike_times_chunk.npy'
spikeclusters_name = 'spike_clusters.npy'
spikeclusters_chunk_name = 'spike_clusters_chunk.npy'
recording_name = 'amplifier.dat'
recording_chunk_name = 'amplifier_chunk.dat'


def generate_test_data(folder, nchannels, start_time, stop_time, fs):
    """generate subset of whole dataset for developemt
    cut binary file between start and stop time
    cut spike times between start and stop time
    """
    fname = os.path.join(folder, recording_name)
    outname = os.path.join(folder, recording_chunk_name)
    copy_recording_chunk(fname, outname, nchannels, start_time, stop_time, fs)

    start_sample = int(start_time*fs)
    stop_sample = int(stop_time*fs)
    spiketimes = np.load(os.path.join(folder, spiketimes_name))
    spikeclusters = np.load(os.path.join(folder, spikeclusters_name))
    chunk_spikes = np.where((spiketimes >= start_sample) * (spiketimes < stop_sample))
    spiketimes_chunk = spiketimes[chunk_spikes] - start_sample
    spikeclusters_chunk = spikeclusters[chunk_spikes]

    outstr = 'Saving %d of %d spike times in chunk from %.1f s to %.1f s' % \
             (len(spiketimes_chunk), len(spiketimes), start_time, stop_time)
    print(outstr)
    np.save(os.path.join(folder, spiketimes_chunk_name), spiketimes_chunk)
    np.save(os.path.join(folder, spikeclusters_chunk_name), spikeclusters_chunk)


folder = r'/Volumes/Time Machine Backups/UCLAProbeTest'
nchannels = 128
start_time = 0.0
stop_time = 60.0
fs = 3e4
generate_test_data(folder, nchannels, start_time, stop_time, fs)