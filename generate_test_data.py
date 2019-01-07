"""Generate small test data set."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np
import os.path
from recordingio import copy_recording_chunk

# -----------------------------------------------------------------------------
# Generate test data
# -----------------------------------------------------------------------------

# some global variables
spiketimes_name = 'spike_times.npy'
spiketimes_chunk_name = 'spike_times_chunk.npy'
spikeclusters_name = 'spike_templates.npy'
spikeclusters_chunk_name = 'spike_templates_chunk.npy'
amplitudes_name = 'amplitudes.npy'
amplitudes_chunk_name = 'amplitudes_chunk.npy'
recording_name = 'cut_amplifier.dat'
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
    spike_times = np.load(os.path.join(folder, spiketimes_name))
    spike_clusters = np.load(os.path.join(folder, spikeclusters_name))
    spike_amplitudes = np.load(os.path.join(folder, amplitudes_name))
    chunk_spikes = np.where((spike_times >= start_sample) * (spike_times < stop_sample))
    spike_times_chunk = spike_times[chunk_spikes] - start_sample
    spike_clusters_chunk = spike_clusters[chunk_spikes]
    spike_amplitudes_chunk = spike_amplitudes[chunk_spikes]

    outstr = 'Saving %d of %d spike times in chunk from %.1f s to %.1f s' % \
             (len(spike_times_chunk), len(spike_times), start_time, stop_time)
    print(outstr)
    np.save(os.path.join(folder, spiketimes_chunk_name), spike_times_chunk)
    np.save(os.path.join(folder, spikeclusters_chunk_name), spike_clusters_chunk)
    np.save(os.path.join(folder, amplitudes_chunk_name), spike_amplitudes_chunk)


folder = r'C:\Users\User\Desktop\MargotClustersCut'
nchannels = 64
start_time = 0.0
stop_time = 60.0
fs = 3e4
generate_test_data(folder, nchannels, start_time, stop_time, fs)