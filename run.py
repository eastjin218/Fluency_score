import os, glob ,re

from utils import convert_wav
from utils import Preprocess, Flu_score

au_path = glob.glob('../Fluency_Sample/rawdata/*.m4a')
out_path = '../Fluency_Sample/con_wav'

#for i in au_path:
#    print(i)
#    convert_wav(audio_path=i, out_path=out_path)

audio_path = '../Fluency_Sample/split.audio'  
stand_audio = '../Fluency_Sample/standard/0000.mp3'

#prep = Preprocess(
#audio_path,stand_audio,
#)
#prep.amp_syn_nr() 


score =Flu_score()
compare_files = glob.glob('../Fluency_Sample/audio.amp.nr/*')
for i in compare_files:
    grade = score.fl_sc(stand_audio, i)
