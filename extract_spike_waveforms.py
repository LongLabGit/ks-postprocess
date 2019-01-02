"""Extract waveforms from recording file at detected spike times."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np
import os.path
from recordingio import load_recording
from scipy import signal
from tqdm import tqdm
import argparse

# -----------------------------------------------------------------------------
# Extract waveforms
# -----------------------------------------------------------------------------


# some default variables
recording_name = 'amplifier.dat'
extract_wf_name = 'extracted_wf.dat'
spiketimes_name = 'spike_times.npy'
spikeclusters_name = 'spike_clusters.npy'
templates_name = 'templates.npy'
channel_map_name = 'channel_map.npy'
channel_shank_map_name = 'channel_shank_map.npy'

filter_order = 3
high_pass = 500.0 # KiloSort default
low_pass = 0.475 # just below Nyquist


def _get_cluster_channels(templates, channel_map, channel_shank_map):
    # templates shape: (template, sample, channel)
    # max channel where max(abs(samples)) is maximal (i.e. peak rectified amplitude)
    # first marginalize across samples, then channels (one dimension less after first marginalize)
    max_channels = np.argmax(np.max(np.abs(templates), axis=1), axis=1)

    shank_channels = []
    for channel in max_channels:
        original_channel = np.where(channel_map == channel)
        shank = channel_shank_map[original_channel]
        shank_channels_ = np.where(channel_shank_map == shank)
        shank_orig_channels = []
        for shank_ch in shank_channels_[0]:
            shank_orig_channels.append(np.where(channel_map == shank_ch)[0])
        shank_channels.append(shank_orig_channels)

    return np.array(shank_channels)


def _get_n_template_samples(templates):
    return templates.shape[1]


def _set_up_filter(order, highpass, lowpass, fs):
    return signal.butter(order, (highpass / (fs / 2.), lowpass / (fs / 2.)), 'bandpass')


def extract_cluster_waveforms(folder, nchannels, fs, wf_samples=61, dtype=np.dtype('int16')):
    """
    for each spike time, look up cluster and corresponding channels,
    then extract samples around spike time (+- 0.5 nsamples) only on channels
    on same shank and store in sequential file
    edge cases: pad with zeros
    also: implement hp-filtering of waveforms (padding required? don't think so)
    :param folder: absolute path to location of recording and KiloSort output files
    :param nchannels: number of channels in recording file
    :param fs: sampling rate in Hz
    :param wf_samples: number of samples (odd) used in template waveforms (used to extract waveform; KS default: 61)
    :param dtype: data type of recording file (default: int16)
    :return: None
    """
    # load stuff
    spike_times = np.load(os.path.join(folder, spiketimes_name)).flatten()
    spike_clusters = np.load(os.path.join(folder, spikeclusters_name)).flatten()
    templates = np.load(os.path.join(folder, templates_name))
    channel_map = np.load(os.path.join(folder, channel_map_name))
    channel_shank_map = np.load(os.path.join(folder, channel_shank_map_name))
    template_samples = _get_n_template_samples(templates)
    if wf_samples != template_samples:
        e = 'Number of waveform samples (%d) does not match template samples (%d)' \
        % (wf_samples, template_samples)
        raise ValueError(e)

    # figure out on which channels which template can be found
    cluster_channels = _get_cluster_channels(templates, channel_map, channel_shank_map)

    # create binary file where we can paste all spike waveforms
    new_samples = wf_samples*len(spike_times)
    min_shank_id = np.min(channel_shank_map)
    channels_per_shank = np.sum(channel_shank_map == min_shank_id)
    extract_wf_file = np.memmap(os.path.join(folder, extract_wf_name), dtype=dtype.name, mode='w+',
                                shape=(channels_per_shank, new_samples), order='F')

    # load recording data file
    rec_file = load_recording(os.path.join(folder, recording_name), nchannels)
    rec_file_info = os.stat(os.path.join(folder, recording_name))
    rec_file_size = rec_file_info.st_size
    rec_samples = int(rec_file_size / (dtype.itemsize * nchannels))

    # set up high-pass filter
    b, a = _set_up_filter(filter_order, high_pass, low_pass*fs, fs)
    bp_filter = lambda x: signal.filtfilt(b, a, x, axis=0)

    # main loop
    wf_offset = wf_samples//2
    for i, spike_sample in enumerate(tqdm(spike_times)):
        spike_cluster = spike_clusters[i]
        wf = np.zeros((channels_per_shank, wf_samples), dtype=dtype.name, order='F')
        # careful with the edge cases - zero-padding
        # uint64 converted silently to float64 when adding an int - cast to int64
        start_index_ = np.int64(spike_sample) - wf_offset
        start_index, start_diff = (start_index_, 0) if start_index_ >= 0 \
                                else (0, -start_index_)
        # uint64 converted silently to float64 when adding an int - cast to int64
        stop_index_ = np.int64(spike_sample) + wf_offset + 1
        stop_index, stop_diff = (stop_index_, 0) if stop_index_ < rec_samples \
                            else (rec_samples, rec_samples - stop_index_)
        # now copy the appropriately sized snippet from channels on same clusters
        wf[:, start_diff:wf_samples-stop_diff] += rec_file[cluster_channels[spike_cluster].flatten(),
                                                  start_index:stop_index]
        # wf = bp_filter(wf) # takes about twice as long to filter every spike
        extract_wf_file[:, i*wf_samples:(i+1)*wf_samples] = wf

    extract_wf_file.flush()


# command line usage
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True, nargs='+')
    parser.add_argument('--nchan', required=True, type=int)
    parser.add_argument('--fs', type=float, default=3e4)
    parser.add_argument('--wfsamp', type=int, default=61)
    parser.add_argument('--dtype', default='int16')
    parser.add_argument('--recname', nargs='+')
    parser.add_argument('--outname', nargs='+')
    parser.add_argument('--stname')
    parser.add_argument('--scname')
    parser.add_argument('--tname')
    parser.add_argument('--chanmap')
    parser.add_argument('--chanshmap')

    args = parser.parse_args()

    if args.recname:
        recording_name = ' '.join(args.recname)
    if args.outname:
        extract_wf_name = ' '.join(args.outname)
    if args.stname:
        spiketimes_name = args.stname
    if args.scname:
        spikeclusters_name = args.scname
    if args.tname:
        templates_name = args.tname
    if args.chanmap:
        channel_map_name = args.chanmap
    if args.chanshmap:
        channel_shank_map_name = args.chanshmap

    dir_ = ' '.join(args.dir)
    extract_cluster_waveforms(dir_, args.nchan, args.fs, args.wfsamp, np.dtype(args.dtype))
