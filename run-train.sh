#!/usr/bin/env bash

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
    -m|--minutes)
    minutes="$2"
    shift # past argument
    shift # past value
    ;;
    -r|--run_id)
    run_id="$2"
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
echo minutes      = "${minutes}"


python3 run-train.py \
    --run_id $run_id \
    --train_files $train_files \
    --valid_files $valid_files \
    --target_dir $target_dir \
    --minutes $minutes 2>&1 | tee $target_dir/$run_id/$run_id.log