#!/usr/bin/env bash
# set -xe
usage="$(basename "$0") [-h] [-t <path>] [-v <path>] [-d <path>]
where:
    -h           show this help text
    -t <path>    use CSV file at <path> containing the corpus files for training
    -v <path>    use CSV file at <path> containing the corpus files for evaluation
    -d <path>    store results at <path>
    -g <int>     GPU to use

Create data to plot a learning curve by running a simplified version of the DeepSpeech-BRNN along the following dimensions:

- time dimension: use varying amounts of training data (1 to 1000 minutes)
- decoder dimension: use different decoding methods  (Beam Search, Best-Path and Old)
- LM dimension: train with or without a Language model (LM)

For each element in the cartesian product of these dimensions a training run is started. Each training run is assigned
a unique run-id from which the value of each dimension can be derived.
"

# Defaults
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

echo train_files  = "${train_files}"
echo valid_files  = "${valid_files}"
echo target_dir   = "${target_dir}"
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

            echo "#################################################################################################"
            echo " Training on $minutes, use_lm=$use_lm, decoding=$decoder"
            echo " run id: $run_id"
            echo " target directory: $target_dir"
            echo "#################################################################################################"

            bash ./run-train.sh \
                --target_dir ${target_dir} \
                --run_id ${run_id} \
                --minutes ${minutes} \
                --decoder ${decoder} \
                --gpu ${gpu} \
                --train_files ${train_files} \
                --valid_files ${valid_files}

            echo "#################################################################################################"
            echo " Finished $run_id"
            echo "#################################################################################################"
        done

    done

done
