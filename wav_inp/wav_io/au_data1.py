import os
import shutil
import scipy
import scipy.io.wavfile
import scipy.io.wavfile as wav
import numpy as np
from python_speech_features import mfcc
from sklearn.externals import joblib
from .models import WavIO
import json
import eyed3

class CreateDataAu:
    def __init__(self, file):
        self.file = file
        self.res_dir = 'wav_io/file/tmp/' + self.file.name + ''
        self.wav_list = []
        self.is_save = False
        file_name, file_extension = os.path.splitext(self.file.name)
        if file_extension != '.wav' and file_extension != '.mp3':
            print(file_extension + '!')
            self.is_save = False
        else:
            with open(self.res_dir, 'wb+') as destination:
                for chunk in self.file.chunks():
                    destination.write(chunk)
            if file_extension == '.mp3':
                self.parse_metadata()
                self.convert_to_wav(file_name)

            self.is_save = True
            self.create_data()

    def parse_metadata(self):
        au = eyed3.load(self.res_dir)
        self.artist = au.tag.artist
        self.albom  = au.tag.album
        self.title   = au.tag.title


    def convert_to_wav(self, file_name):
        com = 'sox ' +'wav_io/file/tmp/"' + file_name + '.mp3" -c 1 -t wav -r 8k ' + 'wav_io/file/tmp/"' + file_name + '.wav" remix -'
        # com = 'mpg123 -w ' + '"wav_io/file/tmp/' + file_name + '.wav" "' + 'wav_io/file/tmp/30_' + file_name + '.mp3"'
        print(com)
        print(os.system(com))
        os.remove('wav_io/file/tmp/' + file_name + '.mp3')
        self.res_dir = 'wav_io/file/tmp/' + file_name + '.wav'

    clust_dir = 'wav_io/file/models/kmeans_clust.pkl'
    def create_data(self):
        if self.is_save:
            (rate, sig) = wav.read(self.res_dir)
            mfcc_features_20 = (mfcc(sig[:len(sig)//3], rate, numcep=20, nfft=1200))

            res_mean = mfcc_features_20.mean(axis=0).tolist()
            res_std = mfcc_features_20.std(axis=0).tolist()
            res_median = np.median(mfcc_features_20, axis=0).tolist()

            res_list = res_mean + res_std + res_median
            clust = joblib.load(self.clust_dir)
            self.tag1 = clust.predict([res_list])

            obj_list = WavIO.objects.filter(tag=self.tag1)
            for obj in obj_list:
                obj.dist = abs(np.linalg.norm(np.array(res_list)-np.array(json.loads(obj.data))))
                obj.save()

            self.wav_list = WavIO.objects.filter(tag=self.tag1).order_by('dist')

            if len(self.wav_list) == 0  or self.wav_list[0].dist > 10:
                WavIO(name = self.file.name,
                    tag = clust.predict([res_list]),
                    data = json.dumps([res_list]),
                    dist = 0).save()
            os.remove(self.res_dir)
            self.is_save = False