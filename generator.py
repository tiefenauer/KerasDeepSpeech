import sys
from abc import abstractmethod
from genericpath import isfile
from os.path import join

import numpy as np
import pandas as pd
import scipy.io.wavfile as wav
from keras.preprocessing.sequence import pad_sequences
from python_speech_features import mfcc
from sklearn.utils import shuffle

from utils import text_to_int_sequence


class BatchGenerator(object):
    def __init__(self, n, shuffle, batch_size):
        self.n = n
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.cur_index = 0

    def __len__(self):
        return self.n // self.batch_size  # number of batches

    def __iter__(self):
        return self

    def __next__(self):
        """
        Returns a generator for the dataset. Because for training the dataset is iterated over once per epoch, this
        function is an endless loop over set.
        :return:
        """
        while True:
            if (self.cur_index + 1) * self.batch_size > self.n:
                self.cur_index = 0

            ret = self.get_batch(self.cur_index)
            self.cur_index += 1
            return ret

    def get_batch(self, idx):
        index_array = range(idx * self.batch_size, (idx + 1) * self.batch_size)
        features = self.extract_features(index_array)
        X = pad_sequences(features, dtype='float32', padding='post')
        X_lengths = np.array([feature.shape[0] for feature in features])

        labels = self.extract_labels(index_array)
        Y = pad_sequences([text_to_int_sequence(label) for label in labels], padding='post', value=28)
        Y_lengths = np.array([len(label) for label in labels])

        inputs = {
            'the_input': X,
            'the_labels': Y,
            'input_length': X_lengths,
            'label_length': Y_lengths,
            'source_str': labels
        }

        outputs = {'ctc': np.zeros([self.batch_size])}

        return inputs, outputs

    @abstractmethod
    def shuffle_entries(self):
        raise NotImplementedError

    @abstractmethod
    def extract_features(self, index_array):
        """
        Extract unpadded features for a batch of elements with specified indices
        :param index_array: array with indices of elements in current batch
        :return: list of unpadded features (batch_size x num_timesteps x num_features)
        """
        raise NotImplementedError

    @abstractmethod
    def extract_labels(self, index_array):
        """
        Extract unpadded, unencoded labels for a batch of elements with specified indices
        :param index_array: array with indices of elements in current batch
        :return: list of textual labels
        """
        """"""
        raise NotImplementedError


class CSVBatchGenerator(BatchGenerator):

    def __init__(self, csv_path, shuffle=False, n_batches=None, batch_size=16):
        df = read_data_from_csv(csv_path=csv_path, sort=True)
        if n_batches:
            df = df.head(n_batches * batch_size)

        self.wav_files = df['wav_filename'].tolist()
        self.wav_sizes = df['wav_filesize'].tolist()
        self.transcripts = df['transcript'].tolist()
        self.wav_features = [None] * len(self.wav_files)

        super().__init__(n=len(df.index), batch_size=batch_size, shuffle=shuffle)
        del df

    def shuffle_entries(self):
        self.wav_files, self.transcripts, self.wav_sizes, self.wav_features = shuffle(self.wav_files, self.transcripts,
                                                                                      self.wav_sizes, self.wav_features)

    def extract_features(self, index_array):
        features = []
        for i in index_array:
            if i < len(self.wav_features) and self.wav_features[i] is not None:
                print('cache hit!')
                features.append(self.wav_features[i])
            else:
                print('cache miss!')
                feature = extract_mfcc(self.wav_files[i])
                self.wav_features[i] = feature
                features.append(feature)
        return features
        # return [extract_mfcc(wav_file) for wav_file in (self.wav_files[i] for i in index_array)]

    def extract_labels(self, index_array):
        return [self.transcripts[i] for i in index_array]


def read_data_from_csv(csv_path, sort=True, create_word_list=False):
    if not isfile(csv_path):
        print(f'ERROR: CSV file {csv_path} does not exist!', file=sys.stderr)
        exit(0)

    print(f'Reading samples from {csv_path}...')
    df = pd.read_csv(csv_path, sep=',', encoding='utf-8')
    print(f'...done! Read {len(df.index)} samples.')

    if create_word_list:
        df['transcript'].to_csv(join('lm', 'df_all_word_list.csv'), header=False, index=False)

    if sort:
        df = df.sort_values(by='wav_filesize', ascending=True)

    return df.reset_index(drop=True)


def extract_mfcc(wav_file_path):
    fs, audio = wav.read(wav_file_path)
    return mfcc(audio, samplerate=fs, numcep=26)  # (num_timesteps x num_features)
