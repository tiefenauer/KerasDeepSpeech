#!/usr/bin/env bash

# create data to plot a learning curve by running a simplified version of the DeepSpeech-BRNN with varying amounts of
# training data (1 to 1000 minutes)
# script idea: https://stackoverflow.com/questions/192249/how-do-i-parse-command-line-arguments-in-bash

POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
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
    -d|--target_dir)
    target_dir="$2"
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

for minutes in 1 10 100 1000
do
    echo "training on $minutes minutes"
    bash ./run-train.sh \
            --train_files $train_files \
            --valid_files $valid_files \
            --target_dir $target_dir \
            --minutes $minutes
done

