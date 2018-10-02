#!/usr/bin/env bash
# set -xe
usage="$(basename "$0") [-h] [-r <string>] [-d <path>] [-t <path>] [-v <path>] [-m <int>]
where:
    -h                              show this help text
    -r <string>                     run-id to use (used to resume training)
    -d <path>                       destination directory to store results
    -x <'beamsearch'|'bestpath'>    decoder to use
    -t <path>                       path to CSV file containing the corpus files to use for training
    -v <path>                       path to CSV file containing the corpus files to use for validation
    -m <int>                        number of minutes of audio for training. If not set or set to 0, all training data will be used (default: 0)
    -g <int>                        GPU to use

Train a simplified model of the DeepSpeech RNN on a given corpus of training- and validation-data.
"

# Defaults
run_id=''
target_dir='/home/daniel_tiefenauer'
minutes=0
gpu='2'
decoder='beamsearch'
train_files='/media/all/D1/readylingua-en/readylingua-en-train.csv'
valid_files='/media/all/D1/readylingua-en/readylingua-en-dev.csv'

POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"
case $key in
    -h|--help)
    echo ${usage}
    shift # past argument
    ;;
    -r|--run_id)
    run_id="$2"
    shift # past argument
    shift # past value
    ;;
    -d|--destination)
    target_dir="$2"
    shift # past argument
    shift # past value
    ;;
    -x|--decoder)
    decoder="$2"
    shift # past argument
    shift # past value
    ;;
    -t|--train_files)
    train_files="$2"
    shift # past argument
    shift # past value
    ;;
    -v|--valid_files)
    valid_files="$2"
    shift # past argument
    shift # past value
    ;;
    -m|--minutes)
    minutes="$2"
    shift # past argument
    shift # past value
    ;;
    -g|--gpu)
    minutes="$2"
    shift # past argument
    shift # past value
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

echo run_id       = "${run_id}"
echo train_files  = "${train_files}"
echo valid_files  = "${valid_files}"
echo target_dir   = "${target_dir}"
echo minutes      = "${minutes}"
echo decoder      = "${decoder}"
echo gpu          = "${gpu}"

python3 run-train.py \
    --run_id ${run_id} \
    --target_dir ${target_dir} \
    --minutes ${minutes} \
    --decoder ${decoder} \
    --gpu ${gpu} \
    --train_files ${train_files} \
    --valid_files ${valid_files} \
    2>&1 | tee ${target_dir}/${run_id}.log # write to stdout and log file
