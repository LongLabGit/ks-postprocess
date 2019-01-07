"""Waveform view."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np
import os

# -----------------------------------------------------------------------------
# Recording
# -----------------------------------------------------------------------------


def load_recording(fname, nchannels, dtype=np.dtype('int16')):
    """returns pointer to binary file
    rows: channel numbers
    columns: samples
    """
    file_info = os.stat(fname)
    file_size = file_info.st_size
    nsamples = int(file_size / (dtype.itemsize * nchannels))
    return np.memmap(fname, dtype=dtype.name, mode='r', shape=(int(nchannels), int(nsamples)), order='F')


def copy_recording_chunk(fname, outname, nchannels, chunk_start, chunk_stop, fs, dtype=np.dtype('int16')):
    """copy binary file from time chunk_start to chunk_stop"""
    if fname == outname:
        e = 'Chunk output name cannot be same as original file'
        raise ValueError(e)
    
    sample_start = int(chunk_start*fs)
    sample_stop = int(chunk_stop*fs)
    if sample_stop <= sample_start:
        raise ValueError
    nsamples = sample_stop - sample_start

    outstr = 'Copying %d channels from file %s to file %s from time %.1f to %.1f s; fs = %.1f' % \
             (nchannels, fname, outname, chunk_start, chunk_stop, fs)
    print(outstr)
    recording = load_recording(fname, nchannels, dtype)
    recording_chunk = np.memmap(outname, dtype=dtype.name, mode='w+', shape=(nchannels, nsamples), order='F')
    recording_chunk[:,:] = recording[:, sample_start:sample_stop]
    recording_chunk.flush()
