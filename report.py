import itertools
import sys
from os import makedirs
from os.path import join, isdir

import keras.backend as K
from keras import callbacks
from tabulate import tabulate
from tqdm import tqdm

from text import *
from util.rnn_util import decode
from utils import save_model


def decode_batch_keras(y_pred, input_length, greedy=True):
    # https://www.dlology.com/blog/how-to-train-a-keras-model-to-recognize-variable-length-text/
    decoded_int = K.get_value(K.ctc_decode(y_pred=y_pred, input_length=input_length, greedy=greedy)[0][0])
    decoded_str = [decode(int_seq) for int_seq in decoded_int]
    return decoded_str


class ReportCallback(callbacks.Callback):
    def __init__(self, data_valid, model, run_id, save_progress=True, early_stopping=False, shuffle_data=True,
                 force_output=False):
        """
        Will calculate WER and LER at epoch end and print out infered transcriptions from validation set using the 
        current model and weights
        :param data_valid: validation data
        :param model: compiled model
        :param run_id: string that identifies the current run
        :param save_progress:
        :param early_stopping: 
        :param shuffle_data: 
        :param force_output: 
        """
        super().__init__()
        self.data_valid = data_valid
        self.model = model
        self.run_id = run_id
        self.save_progress = save_progress
        self.early_stopping = early_stopping
        self.shuffle_data = shuffle_data
        self.force_output = force_output

        y_pred = model.get_layer('ctc').input[0]
        input_data = model.get_layer('the_input').input
        self.test_func = K.function([input_data, K.learning_phase()], [y_pred])

        # WER/LER history
        self.mean_wer_log = []
        self.mean_ler_log = []
        self.norm_mean_ler_log = []

    def validate_epoch(self, epoch):
        K.set_learning_phase(0)

        if self.shuffle_data:
            print("shuffling validation data")
            self.data_valid.shuffle_entries()

        print(f'validating epoch {epoch+1}')
        originals, results = [], []
        self.data_valid.cur_index = 0  # reset index

        validation_results = []

        for _ in tqdm(range(len(self.data_valid))):
            batch_inputs, _ = next(self.data_valid)
            # decoded_res = decode_batch(self.test_func, batch_inputs['the_input'])
            # print(' '.join(decoded_res))
            # y_pred_0 = batch_inputs['the_input']
            # input_length_0 = batch_inputs['input_length']
            # decoded_res_0 = decode_batch_keras(y_pred_0, input_length_0)

            y_pred = self.test_func([batch_inputs['the_input']])[0]
            input_length = batch_inputs['input_length']
            # decoded_res = decode_batch_keras(y_pred, input_length, greedy=True)
            # print(' '.join(decoded_res))
            decoded_res = decode_batch_keras(y_pred, input_length, greedy=False)
            print(' '.join(decoded_res))

            # y_pred_3 = self.test_func([batch_inputs['the_input']])[0]
            # input_length_3 = batch_inputs['input_length']
            # decoded_res_3 = decode_batch_keras(y_pred_3, input_length_3)

            for ground_truth, prediction in zip(batch_inputs['source_str'], decoded_res):
                pred_lm = correction(prediction)

                ler_pred = ler(ground_truth, prediction)
                ler_lm = ler(ground_truth, pred_lm)

                wer_pred = wer(ground_truth, prediction)
                wer_lm = wer(ground_truth, pred_lm)

                if self.force_output or wer_pred < 0.4 or wer_lm < 0.4:
                    validation_results.append((ground_truth, prediction, ler_pred, wer_pred, pred_lm, ler_lm, wer_lm))

                originals.append(ground_truth)
                results.append(pred_lm)

        if validation_results:
            headers = ['Ground Truth', 'Prediction', 'LER', 'WER', 'Prediction (LM-corrected)', 'LER', 'WER']
            print(tabulate(validation_results, headers=headers, floatfmt=".4f"))
        wer_values, wer_mean = wers(originals, results)
        ler_values, ler_mean, ler_values_norm, ler_values_norm_mean = lers(originals, results)
        print("########################################################")
        print("Validation results: WER & LER (using LM)")
        print("WER average is   :", wer_mean)
        print("LER average is   :", ler_mean)
        print("normalised LER is:", ler_values_norm_mean)
        print("########################################################")

        self.mean_wer_log.append(wer_mean)
        self.mean_ler_log.append(ler_mean)
        self.norm_mean_ler_log.append(ler_values_norm_mean)

        K.set_learning_phase(1)

    def on_epoch_end(self, epoch, logs=None):
        self.validate_epoch(epoch)

        # early stopping if VAL WER worse 4 times in a row
        if self.early_stopping and is_early_stopping(self.mean_wer_log):
            print("EARLY STOPPING")
            print("Mean WER   :", self.mean_wer_log)
            print("Mean LER   :", self.mean_ler_log)
            print("NormMeanLER:", self.norm_mean_ler_log)

            sys.exit()

        # save checkpoint if last LER or last WER was better than all previous values
        if self.save_progress and (new_benchmark(self.mean_ler_log) or new_benchmark(self.mean_wer_log)):
            save_dir = join('checkpoints', 'epoch', f'LER-WER-best-{self.run_id}')
            print(f'New WER or LER benchmark! Saving model and weights at {save_dir}')
            if not isdir(save_dir):
                makedirs(save_dir)
            try:
                save_model(self.model, name=save_dir)
            except Exception as e:
                print("couldn't save error:", e)


def decode_batch(test_func, word_batch):
    ret = []
    y_pred = test_func([word_batch])[0]  # 16xTIMEx29 = batch x time x classes

    for out in y_pred:
        best = list(np.argmax(out, axis=1))
        merge = [k for k, g in itertools.groupby(best)]
        # outStr = int_to_text_sequence(merge)
        ret.append(decode(merge))

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
    print(f'{last} vs {rest}')

    return all(i <= last for i in rest)


def new_benchmark(values):
    """
    We have a new benchmark if the last value in a sequence of values is the smallest
    :param values: sequence of values
    :return:
    """
    return len(values) > 2 and values[-1] < np.min(values[:-1])
