#!/usr/bin/env bash
count=0
logdir=../log/dads_baseline

launch() {
    CUDA_VISIBLE_DEVICES=$count python unsupervised_skill_learning/dads_off.py \
	   --flagfile=configs/$1.txt \
	   --logdir=$logdir/$1 &
    export count=$(($count + 1))
}

if [ -e "$logdir" ]; then
    echo "Are you sure you want to remove $logdir?"
    select yn in "Yes" "No"; do
	case $yn in
	    Yes ) rm -rv "$logdir"; break;;
	    No ) exit 1;;
	esac
    done
fi
mkdir "$logdir"

launch ant
launch ant_reset_free
launch ant_xy
launch ant_reset_free_xy
