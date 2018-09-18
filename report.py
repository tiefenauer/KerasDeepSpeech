import itertools
import os
import sys
from os import makedirs
from os.path import join, isdir

import keras.backend as K
import numpy as np
from keras import callbacks
from tqdm import tqdm

from text import *
from utils import save_model, int_to_text_sequence


class ReportCallback(callbacks.Callback):
    def __init__(self, data_valid, model, runtimestr, save, force_output=False):
        super().__init__()
        self.data_valid = data_valid

        y_pred = model.get_layer('ctc').input[0]
        input_data = model.get_layer('the_input').input
        test_func = K.function([input_data, K.learning_phase()], [y_pred])
        self.test_func = test_func

        self.validdata_next_val = self.data_valid.next_batch()
        self.batch_size = data_valid.batch_size
        self.save = save

        # useful if you want to decrease amount in validation
        self.valid_test_devide = 1  # 1=no reduce, 10 = 1/10th
        # if socket.gethostname().lower() in 'rs-e5550'.lower(): self.valid_test_devide = 50

        self.val_best_mean_ed = 0
        self.val_best_norm_mean_ed = 0

        self.lm = get_model()

        self.model = model
        self.runtimestr = runtimestr

        self.mean_wer_log = []
        self.mean_ler_log = []
        self.norm_mean_ler_log = []

        self.earlystopping = True
        self.shuffle_epoch_end = True
        self.force_output = force_output

    def validate_epoch(self, epoch):
        print(f'validating epoch {epoch}')
        K.set_learning_phase(0)

        if self.shuffle_epoch_end:
            print("shuffling validation data")
            self.data_valid.genshuffle()

        originals, results = [], []
        self.data_valid.cur_index = 0  # reset index

        n_val_batches = len(self.data_valid.wav_files) // self.data_valid.batch_size
        if self.valid_test_devide:
            n_val_batches = n_val_batches // self.valid_test_devide

        # make a pass through all the validation data and assess score
        for _ in tqdm(range(0, n_val_batches)):

            word_batch = next(self.validdata_next_val)[0]
            decoded_res = decode_batch(self.test_func,
                                       word_batch['the_input'][0:self.batch_size],
                                       self.batch_size)

            for j in range(0, self.batch_size):
                label_actual = word_batch['source_str'][j]
                label_decoded = decoded_res[j]
                label_corrected = correction(label_decoded)

                wer_decoded = wer(label_actual, label_decoded)
                wer_corrected = wer(label_actual, label_corrected)

                if self.force_output or wer_decoded < 0.4 or wer_corrected < 0.4:
                    print(f'{j} GroundTruth:{label_actual}')
                    print(f'{j} Transcribed:{label_decoded}')
                    print(f'{j} LMCorrected:{label_corrected}')

                originals.append(label_actual)
                results.append(label_corrected)

        rates, mean = wers(originals, results)
        lrates, lmean, norm_lrates, norm_lmean = lers(originals, results)
        print("########################################################")
        print("Validation results: WER & LER (using LM)")
        print("WER average is   :", mean)
        print("LER average is   :", lmean)
        print("normalised LER is:", norm_lmean)
        print("########################################################")

        self.mean_wer_log.append(mean)
        self.mean_ler_log.append(lmean)
        self.norm_mean_ler_log.append(norm_lmean)

        K.set_learning_phase(1)

    def on_epoch_end(self, epoch, logs=None):
        self.validate_epoch(epoch)

        # early stopping if VAL WER worse 4 times in a row
        if self.earlystopping and is_early_stopping(self.mean_wer_log):
            print("EARLY STOPPING")
            print("Mean WER   :", self.mean_wer_log)
            print("Mean LER   :", self.mean_ler_log)
            print("NormMeanLER:", self.norm_mean_ler_log)

            sys.exit()

        # save checkpoint if last LER or last WER was better than all previous values
        if self.save and (new_benchmark(self.mean_ler_log) or new_benchmark(self.mean_wer_log)):
            save_dir = join('checkpoints', 'epoch', f'LER-WER-best-{self.runtimestr}')
            print(f'New WER or LER benchmark! Saving model and weights at {save_dir}')
            if not isdir(save_dir):
                makedirs(save_dir)
            try:
                save_model(self.model, name=save_dir)
            except Exception as e:
                print("couldn't save error:", e)


def decode_batch(test_func, word_batch, batch_size):
    ret = []
    output = test_func([word_batch])[0]  # 16xTIMEx29 = batch x time x classes
    greedy = True
    merge_chars = True

    for j in range(batch_size):  # 0:batch_size

        if greedy:
            out = output[j]
            best = list(np.argmax(out, axis=1))

            if merge_chars:
                merge = [k for k, g in itertools.groupby(best)]

            else:
                raise NotImplementedError("not implemented no merge")

        else:
            pass
            raise NotImplementedError("not implemented beam")

        try:
            outStr = int_to_text_sequence(merge)

        except Exception as e:
            print("Unrecognised character on decode error:", e)
            raise ValueError("DECODE ERROR2")

        ret.append(''.join(outStr))

    return ret


def is_early_stopping(wer_logs):
    """
    stop early if last WER is bigger than all 4 previous WERs
    :param wer_logs: log-scaled WER values
    :return:
    """
    if len(wer_logs) <= 4:
        return False

    last = wer_logs[-1]
    rest = wer_logs[-5:-1]
    print(last, " vs ", rest)

    return all(i <= last for i in rest)


def new_benchmark(values):
    """
    We have a new benchmark if the last value in a sequence of values is the smallest
    :param values: sequence of values
    :return:
    """
    return len(values) > 2 and values[-1] < np.min(values[:-1])
