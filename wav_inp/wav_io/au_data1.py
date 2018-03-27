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

class CreateDataAu:
    def __init__(self, file):
        self.file = file
        self.res_dir = 'wav_io/file/tmp/' + self.file.name
        _, file_extension = os.path.splitext(self.file.name)
        self.wav_list = []

        if (file_extension != '.wav' and file_extension != '.mp3'):
            print(file_extension + '!')
        else:
            with open(self.res_dir, 'wb+') as destination:
                for chunk in self.file.chunks():
                    destination.write(chunk)
            self.is_save = True

    clust_dir = 'wav_io/file/clust/kmeans_clust.pkl'
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