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
from scipy.stats import skew, kurtosis
from django.shortcuts import get_object_or_404

class CreateDataAu:

    clust_dir = 'wav_io/file/classifiers/'
    def __init__(self):

        self.ftag = {'pop': 0, 'jazz': 0, 'synth': 0, 'rock': 0, 'metal': 0, 'blues':0, 'hiphop': 0, 'neoclassic': 0}

        self.data = 'non'
        self.clust = 'non'
        self.artist = 'non'
        self.albom = 'non'
        self.title = 'non'

    def create_model(self, file):
        self.res_dir = 'wav_io/file/tmp/' + file.name + ''
        file_name, file_extension = os.path.splitext(file.name)

        if file_extension != '.wav' and file_extension != '.mp3':
            print(file_extension + '!!!')
        else:
            with open(self.res_dir, 'wb+') as destination:
                for chunk in file.chunks():
                    destination.write(chunk)
            if file_extension == '.mp3':
                self.parse_metadata()
                self.convert_to_wav(file_name)

            rate, sig = wav.read(self.res_dir)
            mfcc_features_40 = mfcc(sig, rate, numcep=40, nfilt=80, nfft=1200)

            l = (len(mfcc_features_40) // 3)
            for i in range(3):
                self.create_tag(mfcc_features_40[i*l: i*l+l])
            for k in self.ftag.keys():
                self.ftag[k] = int(self.ftag[k]/3*100)
            # self.create_recomendations()

            w_io =  WavIO(name=self.title, albom = self.albom, artist = self.artist,
                  data=json.dumps(self.data.tolist()), ftag=json.dumps(self.ftag),
                  clust=self.clust, dist=0)
            if len(WavIO.objects.filter(name=w_io.name, albom = w_io.albom, artist = w_io.artist)) == 0:
                w_io.save()
                self.id = w_io.pk
            else:
                self.id = get_object_or_404(WavIO, name=w_io.name, albom = w_io.albom, artist = w_io.artist)
            os.remove(self.res_dir)

    def parse_metadata(self):
        au = eyed3.load(self.res_dir)
        self.artist = au.tag.artist
        self.albom  = au.tag.album
        self.title  = au.tag.title


    def convert_to_wav(self, file_name):
        com = 'sox wav_io/file/tmp/"' + file_name + '.mp3" -c 1 -t wav -r 8k wav_io/file/tmp/"' + file_name + '.wav" remix -'
        print(com)
        print(os.system(com))
        os.remove('wav_io/file/tmp/' + file_name + '.mp3')
        self.res_dir = 'wav_io/file/tmp/' + file_name + '.wav'

    def create_tag(self, mfcc_features_40):
        data_mean = mfcc_features_40.mean(axis=0).tolist()
        data_std = mfcc_features_40.std(axis=0).tolist()
        data_median = np.median(mfcc_features_40, axis=0).tolist()
        data_skev = skew(mfcc_features_40, axis=0).tolist()
        data_kurt = kurtosis(mfcc_features_40, axis=0).tolist()
        data = data_mean + data_std + data_median + data_skev + data_kurt

        coef = scipy.load(self.clust_dir+'med_coef.npy')
        f_importances = scipy.load(self.clust_dir + 'feature_importances.npy')
        np_data = np.array(data)
        np_data = np_data/np.linalg.norm(np_data)

        datafor_lregr = np_data[f_importances>coef[1]]
        regr = joblib.load(self.clust_dir+'lrc.pkl')
        data_regr = regr.predict_proba([datafor_lregr]).tolist()

        self.data = np_data[f_importances > coef[2]]
        data_km = []
        kmean1 = joblib.load(self.clust_dir+'kmeans1.pkl')
        data_km += kmean1.predict([self.data]).tolist()
        kmean2 = joblib.load(self.clust_dir + 'kmeans2.pkl')
        data_km += kmean2.predict([self.data]).tolist()
        kmean3 = joblib.load(self.clust_dir + 'kmeans3.pkl')
        data_km += kmean3.predict([self.data]).tolist()
        kmean4 = joblib.load(self.clust_dir + 'kmeans4.pkl')
        data_km += kmean4.predict([self.data]).tolist()

        datafor_rfc = np_data[f_importances > coef[0]]
        rfc = joblib.load(self.clust_dir + 'rfc.pkl')
        self.ftag[rfc.predict(np.array([datafor_rfc.tolist() + data_regr[0] + data_km]))[0]] += 1
        self.clust = data_km[2]

    def create_recomendations(self, au_id):
        au_obj = get_object_or_404(WavIO, id = au_id)
        obj_list = WavIO.objects.filter(clust=au_obj.clust)
        for obj in obj_list:
            obj.dist = abs(np.linalg.norm(json.loads(au_obj.data)-np.array(json.loads(obj.data))))
            obj.save()
        self.wav_list = WavIO.objects.filter(clust=au_obj.clust).order_by('dist')[:10]






