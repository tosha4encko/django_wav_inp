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

class CreateDataAu:

    clust_dir = 'wav_io/file/classifiers/'
    def __init__(self, file):

        self.artist = 'non'
        self.albom = 'non'
        self.title = 'non'
        self.ftag = 'non'
        self.file = file
        self.res_dir = 'wav_io/file/tmp/' + self.file.name + ''
        file_name, file_extension = os.path.splitext(self.file.name)

        if file_extension != '.wav' and file_extension != '.mp3':
            print(file_extension + '!!!')
        else:
            with open(self.res_dir, 'wb+') as destination:
                for chunk in self.file.chunks():
                    destination.write(chunk)
            if file_extension == '.mp3':
                self.parse_metadata()
                self.convert_to_wav(file_name)
            (rate, sig) = wav.read(self.res_dir)
            self.create_tag(rate, sig)
            os.remove(self.res_dir)

    def parse_metadata(self):
        au = eyed3.load(self.res_dir)
        self.artist = au.tag.artist
        self.albom  = au.tag.album
        self.title  = au.tag.title


    def convert_to_wav(self, file_name):
        com = 'sox wav_io/file/tmp/"' + file_name + '.mp3" -c 1 -t wav -r 8k wav_io/file/tmp/"' + file_name + '.wav" remix -'
        # com = 'mpg123 -w ' + '"wav_io/file/tmp/' + file_name + '.wav" "' + 'wav_io/file/tmp/30_' + file_name + '.mp3"'
        print(com)
        print(os.system(com))
        os.remove('wav_io/file/tmp/' + file_name + '.mp3')
        self.res_dir = 'wav_io/file/tmp/' + file_name + '.wav'

    def create_tag(self, rate, sig):
        mfcc_features_40 = (mfcc(sig, rate, numcep=40, nfilt=80, nfft=1200))

        data_mean = mfcc_features_40.mean(axis=0).tolist()
        data_std = mfcc_features_40.std(axis=0).tolist()
        data_median = np.median(mfcc_features_40, axis=0).tolist()
        data_skev = skew(mfcc_features_40, axis=0).tolist()
        data_kurt = kurtosis(mfcc_features_40, axis=0).tolist()
        data = data_mean + data_std + data_median + data_skev + data_kurt

        coef = scipy.load(self.clust_dir+'med_coef.npy')
        f_importances = scipy.load(self.clust_dir + 'feature_importances.npy')
        np_data = np.array(data)

        datafor_lregr = np_data[f_importances>coef[1]]
        regr = joblib.load(self.clust_dir+'lrc.pkl')
        data_regr = regr.predict_proba([datafor_lregr]).tolist()

        datafor_kmeans = np_data[f_importances > coef[2]]
        data_km = []
        kmean1 = joblib.load(self.clust_dir+'kmeans1.pkl')
        data_km += kmean1.predict([datafor_kmeans]).tolist()
        kmean2 = joblib.load(self.clust_dir + 'kmeans2.pkl')
        data_km += kmean2.predict([datafor_kmeans]).tolist()
        kmean3 = joblib.load(self.clust_dir + 'kmeans3.pkl')
        data_km += kmean3.predict([datafor_kmeans]).tolist()
        kmean4 = joblib.load(self.clust_dir + 'kmeans4.pkl')
        data_km += kmean4.predict([datafor_kmeans]).tolist()

        datafor_rfc = np_data[f_importances > coef[0]]
        rfc = joblib.load(self.clust_dir + 'rfc.pkl')
        self.ftag = rfc.predict(np.array([datafor_rfc.tolist() + data_regr[0] + data_km]))
            # obj_list = WavIO.objects.filter(tag=self.tag1)
            # for obj in obj_list:
            #     obj.dist = abs(np.linalg.norm(np.array(res_list)-np.array(json.loads(obj.data))))
            #     obj.save()
            #
            # self.wav_list = WavIO.objects.filter(tag=self.tag1).order_by('dist')
            #
            # if len(self.wav_list) == 0  or self.wav_list[0].dist > 10:
            #     WavIO(name = self.file.name,
            #         tag = clust.predict([res_list]),
            #         data = json.dumps([res_list]),
            #         dist = 0).save()
            # os.remove(self.res_dir)
            # self.is_save = False

