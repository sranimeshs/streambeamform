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


# Sample rate for recording the audio
SAMPLE_RATE = 44100
# Number of channels
NUM_CHANNELS = 2
# Number of buffers for computing correlation
NUM_BUFF = 3
# Size of buffer
BUFF_SIZE = 1024

class WavBuffer(object):
    def __init__(self, datatype):
        self.buff = np.zeros((NUM_BUFF * BUFF_SIZE,), dtype=datatype)
        self.size = 0

    def is_buffer_full(self):
        if self.size == NUM_BUFF * BUFF_SIZE:
            return True
        return False

    def insert(self, data):
        assert len(data) <= BUFF_SIZE, "The size of the data is more than %d" % BUFF_SIZE

        if self.is_buffer_full():
            # Make space for the new buffer
            self.buff[:(NUM_BUFF-1)*BUFF_SIZE] = self.buff[BUFF_SIZE:NUM_BUFF*BUFF_SIZE]
            self.buff[-BUFF_SIZE:] = data
        else:
            self.buff[self.size: self.size + len(data)] = data
            self.size += len(data)


class BeamFormer(object):
    def __init__(self, datatype, delay_list):
        self.delays = delay_list

        self.left_buff = WavBuffer(datatype)
        self.right_buff = WavBuffer(datatype)

        #self.shifted = {delay: np.zeros((BUFF_SIZE,), dtype=datatype) for delay in self.delays}

    def get_beamformed_data(self, left_wav, right_wav):
        self.left_buff.insert(left_wav)
        self.right_buff.insert(right_wav)

        if self.left_buff.is_buffer_full():
            # Compute full correlation. Now we have results for all positive and
            # negative delays.
            xcorr = np.correlate(self.left_buff.buff, self.right_buff.buff, 'full')

            opt_delay = 0
            max_corr = -sys.maxsize

            # Find which delay in the list of delays gives us the max correlation
            for delay in self.delays:
                if xcorr[len(xcorr)/2 + delay] > max_corr:
                    max_corr = xcorr[len(xcorr)/2 + delay]
                    opt_delay = delay

            # Get a temporary right buffer and shift it accordingly
            temp_right = np.zeros((NUM_BUFF * BUFF_SIZE,), dtype=right_wav.dtype)
            # Right buffer lags behind left buffer
            if opt_delay < 0:
                temp_right[-opt_delay:] = self.right_buff.buff[:self.right_buff.size + opt_delay]
            # Right buffer leads the left buffer
            elif opt_delay > 0:
                temp_right[:len(temp_right) - opt_delay] = self.right_buff.buff[opt_delay:]

            # Return the last BUFF_SIZE of beamformed data
            return (self.left_buff.buff[(NUM_BUFF-1)*BUFF_SIZE:NUM_BUFF*BUFF_SIZE] +
                    temp_right[(NUM_BUFF-1)*BUFF_SIZE:NUM_BUFF*BUFF_SIZE])/2
        else:
            # We are bootstrapping. Send the average of the mono channels.
            return (left_wav + right_wav)/2


@click.command()
@click.option('--inp_wav_file', type=str, default="Wav file", help="Fullpath of the input wav file")
@click.option('--out_wav_file', type=str, default="Wav file", help="Fullpath of the output wav file")
def replay_and_record(inp_wav_file, out_wav_file):

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
    time_taken_per_buffer = []
    while len(leftwav) - frames_read > BUFF_SIZE:
        #print('frames read %d out of %d frames' % (frames_read, len(leftwav)))
        frame_end_ind = frames_read + min(BUFF_SIZE, len(leftwav) - frames_read)
        left_frames = leftwav[frames_read : frame_end_ind]
        right_frames = rightwav[frames_read : frame_end_ind]

        start_beamforming = time.time()
        mono_out = beamformer.get_beamformed_data(left_frames, right_frames)
        end_beamforming = time.time()

        outwav[frames_read:frame_end_ind] = mono_out;
        frames_read = frame_end_ind

        time_taken_per_buffer.append((end_beamforming - start_beamforming))

    wavfile.write(out_wav_file, rate, outwav)

    with open('stats.txt', 'w') as fw:
        for stat in time_taken_per_buffer:
            fw.write(str(stat) + '\n')


if __name__ == '__main__':
    replay_and_record()
