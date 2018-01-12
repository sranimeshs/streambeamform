import os
import sys
import pdb
import wave
import time
import click
import numpy as np
import logging
import subprocess
import audioop
from scipy.io import wavfile

logging.basicConfig(filename="record_n_play.log", level=logging.INFO)
logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
SAMPLE_RATE_SS = 44100

class BeamFormer(object):
    def __init__(self, datatype, delay_list):
        # The list of delays we are interested in.
        self.delays = delay_list

    def get_beamformed_data(self, left_wav, right_wav):
        # Compute full correlation. Now we have results for all positive and
        # negative delays.
        max_corr = -sys.maxsize
        opt_delay = 0
        opt_l = left_wav
        opt_r = right_wav

        for delay in self.delays:
            if delay < 0:
                r = right_wav[-delay:]
                l = left_wav[: r.size]
                corrcoef = np.corrcoef(l, r)[0,1]
            else:
                l = left_wav[delay:]
                r = right_wav[: l.size]
                corrcoef = np.corrcoef(l, r)[0,1]

            if max_corr < corrcoef:
                max_corr = corrcoef
                opt_delay = delay
                opt_l = l
                opt_r = r

        left_pad = np.zeros((0,), dtype=left_wav.dtype)
        right_pad = np.zeros((0,), dtype=left_wav.dtype)

        if opt_delay < 0:
            right_pad = left_wav[opt_l.size:]

        else:
            left_pad = left_wav[:opt_delay]

        # Avoid overflowing by dividing first and then adding
        assert opt_l.size == opt_r.size, "left and right buffer sizes are not matching"
        outdata = np.append(np.append(left_pad, (opt_l/2 + opt_r/2)), right_pad)
        logger.info("%d, %d", opt_delay, outdata.size)
        return audioop.ratecv(outdata.tobytes(), 2, 1, SAMPLE_RATE_SS, SAMPLE_RATE, None)[0]


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

    #outwav = np.zeros((len(leftwav),), dtype=leftwav.dtype)
    outwav = []

    # Delay list that we want to investigate
    delay_list = range(-3, 4)
    # Get the beamformer instance
    beamformer = BeamFormer(leftwav.dtype, delay_list)

    frames_read = 0
    frames_written = 0
    time_taken_per_buffer = []

    faudio = open('audio.raw', 'w+')
    while len(leftwav) - frames_read > BUFF_SIZE:
        frame_end_ind = frames_read + min(BUFF_SIZE, len(leftwav) - frames_read)
        left_frames = leftwav[frames_read : frame_end_ind]
        right_frames = rightwav[frames_read : frame_end_ind]

        start_beamforming = time.time()
        mono_out = beamformer.get_beamformed_data(left_frames, right_frames)
        end_beamforming = time.time()

        faudio.write(mono_out)
        frames_written += len(mono_out)/2

        frames_read = frame_end_ind
        time_taken_per_buffer.append([(end_beamforming - start_beamforming), len(mono_out)/2])

    #wavfile.write(out_wav_file, SAMPLE_RATE, outwav)
    faudio.close()

    with open(stats_file, 'w') as fw:
        for stat in time_taken_per_buffer:
            fw.write(str(stat[0]) + ',' + str(stat[1]) + '\n')


if __name__ == '__main__':
    replay_and_record()
