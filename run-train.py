#!/usr/bin/env python

"""
Keras implementation of simplified DeepSpeech model.
Forked and adjusted from: https://github.com/robmsmt/KerasDeepSpeech
"""

import argparse
import datetime
import os
import sys
from os import makedirs
from os.path import join, abspath, isdir

from keras.callbacks import TensorBoard
from keras.optimizers import SGD

from generator import CSVBatchGenerator
from model import *
from report import ReportCallback
from util.log_util import create_args_str
from utils import load_model_checkpoint, save_model, MemoryCallback

#######################################################
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Prevent pool_allocator message
#######################################################

parser = argparse.ArgumentParser()
parser.add_argument('--tensorboard', type=bool, default=True, help='True/False to use tensorboard')
parser.add_argument('--memcheck', type=bool, default=False, help='print out memory details for each epoch')
parser.add_argument('--name', type=str, default='',
                    help='name of run, used to set checkpoint save name. Default uses timestamp')
parser.add_argument('--train_files', type=str, default='',
                    help='list of all train files, seperated by a comma if multiple')
parser.add_argument('--valid_files', type=str, default='',
                    help='list of all validation files, seperate by a comma if multiple')
parser.add_argument('--train_batches', type=int, default=0,
                    help='number of batches to use for training in each epoch. Use 0 for automatic')
parser.add_argument('--valid_batches', type=int, default=0,
                    help='number of batches to use for validation in each epoch. Use 0 for automatic')
parser.add_argument('--fc_size', type=int, default=512, help='fully connected size for model')
parser.add_argument('--rnn_size', type=int, default=512, help='size of the RNN layer')
parser.add_argument('--model_path', type=str, default='',
                    help="""If value set, load the checkpoint in a folder minus name minus the extension (weights 
                       assumed as same name diff ext) e.g. --model_path ./checkpoints/ TRIMMED_ds_ctc_model/""")
parser.add_argument('--learning_rate', type=float, default=0.01, help='the learning rate used by the optimiser')
parser.add_argument('--sort_samples', type=bool, default=True,
                    help='sort utterances by their length in the first epoch')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train the model')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size used to train the model')
parser.add_argument('--gpu', type=str, nargs='?', default='2', help='(optional) GPU(s) to use for training. Default: 2')
args = parser.parse_args()


def main():
    output_dir = setup()
    print(create_args_str(args))

    model = create_model(output_dir)
    # opt = Adam(lr=args.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8, clipnorm=5)
    opt = SGD(lr=args.learning_rate, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    model.compile(optimizer=opt, loss=ctc)

    train_model(model)


def setup():
    if args.name == "":
        args.name = 'DS' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')

    output_dir = join('checkpoints', 'results', f'model_{args.name}')
    if not isdir(output_dir):
        makedirs(output_dir)

    log_file_path = join(output_dir, 'train.log')
    # redirect_to_file(log_file_path)

    # detect user here too
    if args.train_files == "" and args.valid_files == "":
        # if paths to file not specified, assume testing
        test_path = join('data', 'ldc93s1')
        args.train_files = abspath(join(test_path, "ldc93s1.csv"))
        args.valid_files = abspath(join(test_path, "ldc93s1.csv"))

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    return output_dir


def create_model(output_dir):
    if args.model_path:
        print(f'Loading model from {args.model_path}')
        if not isdir(args.model_path):
            print(f'ERROR: directory {args.model_path} does not exist!', file=sys.stderr)
            exit(0)
        model = load_model_checkpoint(args.model_path)
    else:
        print('Creating new model')
        # model = deep_speech_dropout(input_dim=26, fc_size=args.fc_size, rnn_size=args.rnn_size, output_dim=29)
        model = ds1(input_dim=26, fc_size=args.fc_size, rnn_size=args.rnn_size, output_dim=29)
        save_model(model, output_dir)
        print(f'model saved in {output_dir}')

    model.summary()
    return model


def train_model(model):
    print("Creating data batch generators")
    data_train = CSVBatchGenerator(args.train_files, shuffle=False, n_batches=args.train_batches,
                                   batch_size=args.batch_size)
    data_valid = CSVBatchGenerator(args.valid_files, shuffle=True, n_batches=args.valid_batches,
                                   batch_size=args.batch_size)

    cb_list = []
    if args.memcheck:
        cb_list.append(MemoryCallback())

    if args.tensorboard:
        tb_cb = TensorBoard(log_dir=join('tensorboard', args.name), write_graph=False, write_images=True)
        cb_list.append(tb_cb)

    report_cb = ReportCallback(data_valid, model, args.name, save=True)

    cb_list.append(report_cb)

    model.fit_generator(generator=data_train, validation_data=data_valid, epochs=args.epochs, callbacks=cb_list)

    print("Mean WER   :", report_cb.mean_wer_log)
    print("Mean LER   :", report_cb.mean_ler_log)
    print("NormMeanLER:", report_cb.norm_mean_ler_log)

    K.clear_session()


if __name__ == '__main__':
    main()
