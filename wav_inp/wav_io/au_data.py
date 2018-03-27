import os
import shutil
import scipy
import scipy.io.wavfile
import scipy.io.wavfile as wav
import numpy as np
from python_speech_features import mfcc
import csv

class CreateDataAu:
    def __init__(self):
        self.res_dir = '/home/tosha/work/au_data/'
        self.set_tag = {'synth/' , 'metal/', 'neoclassic/'}
        self.name_frame = 'wav_frame.csv'
        self.csv_title = ['tag',
                          'm1',  'm2',  'm3',  'm4',  'm5',  'm6',  'm7',  'm8',  'm9',  'm10',
                          'm11', 'm12', 'm13', 'm14', 'm15', 'm16', 'm17', 'm18', 'm19', 'm20',
                          's1',  's2',  's3',  's4',  's5',  's6',  's7',  's8',  's9',  's10',
                          's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20',
                          'med1',  'med2',  'med3',  'med4',  'med5',  'med6',  'med7',  'med8',  'med9',  'med10',
                          'med11', 'med12', 'med13', 'med14', 'med15', 'med16', 'med17', 'med18', 'med19', 'med20']
        with open(self.name_frame , 'w', newline='') as out_file:
            csv.writer(out_file, delimiter=',').writerow(self.csv_title)

    def create_7z_data(self, file_dir, tag):
        _7z_list = os.listdir(file_dir)
        for _7z in _7z_list:
            print(os.system('7z x ' + file_dir + _7z + ' -o' + self.res_dir + tag))

    def create_file_data(self, file_dir, tag):
        obj_list = os.listdir(self.res_dir + tag)
        for obj in obj_list:
            if os.path.isdir(self.res_dir + tag + obj):
                file_list = os.listdir(self.res_dir + tag + obj)
                for file in file_list:
                    print(self.res_dir + tag + obj + '/' + file, self.res_dir + tag)
                    try:
                        shutil.copy(self.res_dir + tag + obj + '/' + file, self.res_dir + tag)
                    except:
                        continue
                shutil.rmtree(self.res_dir + tag + obj)

    def create_wav_data(self, tag, ind):
        dir = self.res_dir + tag
        list_file = os.listdir(dir)
        for file in list_file:
            try:
                com = 'mpg123 -w ' + dir + str(ind) + '.wav ' + dir+ '"' + file + '"'
                print(os.system(com))
                ind += 1

            except:
                continue

            os.remove(dir+file)
            if (ind == 300):
                exit()

    def create_data(self, file_dir, tag):
        self.create_7z_data(file_dir, tag)
        self.create_file_data(file_dir, tag)

    def save_mfcc(self, tag):
        # list_tag = os.listdir(self.res_dir)
        ind = 1
        os.mkdir(tag)
        list_file = os.listdir(os.path.join(self.res_dir, tag))
        for file in list_file:
            file_dir = os.path.join(self.res_dir, tag, file)
            try:
                (rate, sig) = wav.read(file_dir)
                # mfcc_features_13     = mfcc(sig, rate)[:3000]
                # scipy.save(tag + '/' + str(ind) + '_13.spec', mfcc_features_13)
                mfcc_features_20     = (mfcc(sig, rate, numcep=20))[:3000]
                scipy.save(tag + '/' + str(ind) + '_20.spec', mfcc_features_20)
                # mfcc_features_30     = (mfcc(sig, rate,numcep=30))[:3000]
                # scipy.save(tag + '/' + str(ind) + '_30.spec', mfcc_features_30)
                # mfcc_features_50     = (mfcc(sig, rate,numcep=50))[:3000]
                # scipy.save(tag + '/' + str(ind) + '_50.spec', mfcc_features_50)
                print(ind)
                ind += 1
            except:
                print('err', ind)
                continue

    def create_frame(self):
        for tag in self.set_tag:
            obj_list = os.listdir(tag)
            for obj in obj_list[:145]:
                np_arr = scipy.load(os.path.join(tag, obj))
                res_mean = np_arr.mean(axis=0).tolist()
                res_std = np_arr.std(axis=0).tolist()
                res_median = np.median(np_arr, axis=0).tolist()
                res_list = [tag] + res_mean + res_std + res_median
                with open(self.name_frame, 'a', newline='') as out_file:
                    csv.writer(out_file, delimiter=',').writerow(res_list)
            #     if len(np_arr) == 3000:
            #         c += 1
            #     else:
            #         print(tag, len(np_arr))
            # print(tag, ' : ', c)
            # c = 0
