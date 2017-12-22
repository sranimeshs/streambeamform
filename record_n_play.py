import os
import sys
import pdb
import wave
import time
import click
import numpy as np
import logging
import subprocess
from scipy.io import wavfile

logging.basicConfig(filename="record_n_play.log", level=logging.INFO)
logger = logging.getLogger(__name__)

class BeamFormer(object):
    def __init__(self, datatype, delay_list):
        self.delays = delay_list

    def get_beamformed_data(self, left_wav, right_wav):
        # Prepare data
        left_wav = np.append(self.left_history, left_wav)
        right_wav = np.append(self.right_history, right_wav)

        # Compute full correlation. Now we have results for all positive and
        # negative delays.
        xcorr = np.correlate(left_wav, right_wav, 'full')

        opt_delay = 0
        max_corr = -sys.maxsize

        # Find which delay in the list of delays gives us the max correlation
        for delay in self.delays:
            if xcorr[len(xcorr)/2 + delay] > max_corr:
                max_corr = xcorr[len(xcorr)/2 + delay]
                opt_delay = delay

        if opt_delay < 0:
            r = right_wav[-opt_delay: min(-opt_delay + left_wav.size, right_wav.size)]
            l = left_wav[:r.size]

        elif opt_delay >= 0:
            l = left_wav[opt_delay: min(left_wav.size, opt_delay + right_wav.size)]
            r = right_wav[:l.size]

        # Avoid overflowing by dividing first and then adding
        assert l.size == r.size, "left and right buffer sizes are not matching"
        logger.info("%d, %d, %d, %d", opt_delay, l.size, self.left_history.size, self.right_history.size)
        return (l/2 + r/2)


@click.command()
@click.option('--inp_wav_file', type=str, default="Wav file", help="Fullpath of the input wav file")
@click.option('--out_wav_file', type=str, default="Wav file", help="Fullpath of the output wav file")
@click.option('--buff_size', type=str, default="buffer size", help="Amount of data to be read from wav file at a time")
@click.option('--stats_file', type=str, default="Text file", help="Fullpath of the file containing stats")
def replay_and_record(inp_wav_file, out_wav_file, buff_size, stats_file):

    global BUFF_SIZE
    BUFF_SIZE = int(buff_size)

    # Get the data and load it into the memory so that we don't read it from the file
    # all the time.
    rate, wavdata = wavfile.read(inp_wav_file)
    leftwav = wavdata[:,0]
    rightwav = wavdata[:,1]

    outwav = np.zeros((len(leftwav),), dtype=leftwav.dtype)

    # Delay list that we want to investigate
    delay_list = range(-3, 4)
    # Get the beamformer instance
    beamformer = BeamFormer(leftwav.dtype, delay_list)

    frames_read = 0
    frames_written = 0
    time_taken_per_buffer = []
    while len(leftwav) - frames_read > BUFF_SIZE:
        frame_end_ind = frames_read + min(BUFF_SIZE, len(leftwav) - frames_read)
        left_frames = leftwav[frames_read : frame_end_ind]
        right_frames = rightwav[frames_read : frame_end_ind]

        start_beamforming = time.time()
        mono_out = beamformer.get_beamformed_data(left_frames, right_frames)
        end_beamforming = time.time()

        outwav[frames_written:frames_written + mono_out.size] = mono_out;
        frames_written += mono_out.size

        frames_read = frame_end_ind
        time_taken_per_buffer.append([(end_beamforming - start_beamforming), mono_out.size])

    wavfile.write(out_wav_file, rate, outwav)

    with open(stats_file, 'w') as fw:
        for stat in time_taken_per_buffer:
            fw.write(str(stat[0]) + ',' + str(stat[1]) + '\n')


if __name__ == '__main__':
    replay_and_record()
