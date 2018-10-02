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

while getopts ':hs:' option; do
  case "$option" in
    h) echo "$usage"
       exit
       ;;
    r) run_id=$OPTARG
       ;;
    d) target_dir=$OPTARG
       ;;
    m) minutes=$OPTARG
       ;;
    x) decoder=$OPTARG
       ;;
    t) train_files=$OPTARG
       ;;
    v) valid_files=$OPTARG
       ;;
    g) gpu=$OPTARG
       ;;
#    :) printf "missing argument for -%s\n" "$OPTARG" >&2
#       echo "$usage" >&2
#       exit 1
#       ;;
   \?) printf "illegal option: -%s\n" "$OPTARG" >&2
       echo "$usage" >&2
       exit 1
       ;;
  esac
done
shift $((OPTIND - 1))

echo train_files  = "${train_files}"
echo valid_files  = "${valid_files}"
echo target_dir   = "${target_dir}"
echo minutes      = "${minutes}"
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
