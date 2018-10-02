#!/usr/bin/env bash
# set -xe
usage="$(basename "$0") [-h|--help] [-d|--destination <path>] [-t|--train_files <path>] [-v|--valid_files <path>] [-g|--gpu]
where:
    -h|--help                    show this help text
    -d|--destination <path>      destination directory to store results
    -l|--lm                      path to n-gram KenLM model (if possible binary)
    -a|--lm_vocab                path to file containing the vocabulary of the LM specified by -lm. The file must contain the words used for training delimited by space (no newlines)
    -t|--train_files <path>      one or more comma-separated paths to CSV files containing the corpus files to use for training
    -v|--valid_files <path>      one or more comma-separated paths to CSV files containing the corpus files to use for validation
    -g|--gpu <int>               GPU to use (default: 2)

Create data to plot a learning curve by running a simplified version of the DeepSpeech-BRNN. The purpose of this script is simply to call ./run-train.sh with varying parameters along the following dimensions:

- time dimension: use increasing amounts of training data (1 to 1000 minutes)
- decoder dimension: use different decoding methods  (Beam Search, Best-Path and Old)
- LM dimension: train with or without a Language model (LM)

For each element in the cartesian product of these dimensions a training run is started. A unique run-id is assigned to each training run from which the value of each dimension can be derived.
"

# Defaults
lm=''
lm_vocab=''
train_files='/media/all/D1/readylingua-en/readylingua-en-train.csv'
valid_files='/media/all/D1/readylingua-en/readylingua-en-dev.csv'
target_dir='/home/daniel_tiefenauer/learning_curve_0'
gpu='2'

POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"
case $key in
    -h|--help)
    echo ${usage}
    shift # past argument
    ;;
    -d|--destination)
    target_dir="$2"
    shift # past argument
    shift # past value
    ;;
    -l|--lm)
    lm="$2"
    shift
    shift
    ;;
    -a|--lm_vocab)
    lm_vocab="$2"
    shift
    shift
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

echo target_dir   = "${target_dir}"
echo lm           = "${lm}"
echo lm_vocab     = "${lm_vocab}"
echo train_files  = "${train_files}"
echo valid_files  = "${valid_files}"
echo gpu          = "${gpu}"

# time dimension
for minutes in 1 10 100 1000
do
    # LM dimension
    for use_lm in true false
    do
        # decoder dimension
        for decoder in 'beamsearch' 'bestpath' 'old'
        do
            [[$use_lm == true]] && lm="withLM" || lm="noLM"
            run_id="${minutes}min_${lm}_${decoder}"

            mkdir -p ${target_dir}

            echo "#################################################################################################"
            echo " Training on $minutes, use_lm=$use_lm, decoding=$decoder"
            echo " run id: $run_id"
            echo " target directory: $target_dir"
            echo "#################################################################################################"

            bash ./run-train.sh \
                --run_id ${run_id} \
                --target_dir ${target_dir} \
                --minutes ${minutes} \
                --decoder ${decoder} \
                --lm ${lm} \
                --lm_vocab ${lm_vocab} \
                --train_files ${train_files} \
                --valid_files ${valid_files} \
                --gpu ${gpu} \

            echo "#################################################################################################"
            echo " Finished $run_id"
            echo "#################################################################################################"
        done

    done

done
