import os, glob, re ,sys
import pandas as pd
import numpy as np

import soundfile as sf
import librosa
import noisereduce as nr

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

 
def cal_ad_rms(clean_rms, snr):
    a = float(snr) /20
    noise_rm =clean_rms/(10**a)
    return noise_rm

def cal_amp(wf):
    buffer1=wf.readframs(wf.getframes())
    amptitude = (np.formbuffer(buffer1, dtype='int16')).astype(np.float64)
    return amptitude

def cal_rms(amp):
    return np.sqrt(np.mean(np.square(amp), axis=-1))

def audio_amp_change(st_file, in_file):
    sr = 22050
    st_amp = librosa.load(st_file, sr=sr)
    in_amp = librosa.load(in_file,sr=sr)
    st_rms = cal_rms(st_amp[0])
    in_rms = cal_rms(in_amp[0])
    snr = -1
    ad_in_rms = cal_ad_rms(st_rms, snr)
    ad_in_amp = in_amp[0] * (ad_in_rms /in_rms)
    return ad_in_amp, sr

def amp_syn_nr(stand_audio, input_audio):
    ch_x, ch_sr =audio_amp_change(stand_audio, input_audio)
    noise_part = ch_x[1000:round(ch_sr/2)]
    noise_can = nr.reduce_noise(audio_clip =ch_x, noise_clip=noise_part, verbose=False)
    return noise_can, ch_sr    


def convert_wav(audio_path, out_path):
    os.makedirs(out_path, exist_ok=True)
    name = os.path.basename(audio_path)
    name = name.split('.')[0]
    os.system('ffmpeg -i %s -ac 1 -ar 22050 %s/%s.wav'%(audio_path, out_path, name))


class Flu_score():

    def compare_audio_mfcc(self, st_au, st_sr, in_au, in_sr): 
        st_stft = librosa.stft(st_au, n_fft=2048, win_length=2048, hop_length=512)
        in_stft = librosa.stft(in_au, n_fft=2048, win_length=2048, hop_length=512)
        st_mfcc = librosa.feature.mfcc(S = librosa.power_to_db(st_stft), sr = st_sr, n_mfcc=20)
        in_mfcc = librosa.feature.mfcc(S = librosa.power_to_db(in_stft), sr = in_sr, n_mfcc=20)
        count = 0
        for i in range(20):
            dis, path = fastdtw(st_mfcc[i,:], in_mfcc[i,:], dist=euclidean)
            count += dis
        return count/20

    def fl_sc(self, stand_file, in_au, in_sr, input_file):
        self.st_au ,self.st_sr = librosa.load(stand_file)
        self.in_au = in_au
        self.in_sr = in_sr

        sec_range = 1981.746
        low_rate = 0.933
        high_rate = 1.12
        
        st_name = os.path.basename(stand_file)
        st_name = st_name.split('.')[0]
        in_name1 = os.path.basename(input_file)
        in_name = in_name1.split('_')[0]

        if st_name == in_name:
            score_med = (len(self.st_au)/self.st_sr)*sec_range
            score_low_b = score_med*low_rate
            score_high_b = score_med*high_rate
            check_score = self.compare_audio_mfcc(self.st_au, self.st_sr, self.in_au, self.in_sr)
        return check_score, len(self.st_au)/self.st_sr



stand_audio = sys.argv[1]
compare_audio = sys.argv[2]

in_au , in_sr = amp_syn_nr(stand_audio, compare_audio)

score = Flu_score()
grade, st_time = score.fl_sc(stand_audio, in_au, in_sr, compare_audio)
print("%s|%s"%(str(grade),str(st_time)))






















