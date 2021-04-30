import os, glob, re
import pandas as pd
import numpy as np

import soundfile as sf
import librosa
import noisereduce as nr

class preprocess():
    def __init__(self,audio_path, stand_audio):
        self.audio_path = audio_path
        self.stand_audio = stand_audio
    
    def cal_ad_rms(clean_rms, snr):
        a = float(snr) /20
        nosie_rm =clean_rms/(10**a)
        return noise_rms

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
        ad_in_rms = cal_ad_rms(st_rms_snr)
        ad_in_amp = in_amp[0] * (ad_in_rms /in_rms)
        return ad_in_amp, sr

    def amp_syn_nr(self):
        self.audio = glob.glob(os.path.join(self.audio_path,'*'))
        self.outdir = os.path.split(self.audio)[0]
        name = os.path.split(self.audio)[1]
        self.outdir = os.path.split(self.outdir)[0]

        for i in self.audio:
            ch_x, ch_sr = audio_amp_change(st_file = self.stand_audio, in_file = i)
            noise_part = ch_x[1000:round(sr/2)]
            noise_can = nr.reduce_noise(audio_clip =ch_x, noise_clip=noise_part, verbose=False)

    
