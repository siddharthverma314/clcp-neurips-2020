#!/usr/bin/env bash

# Env variables for docker-machine
export MACHINE_DRIVER="google"
export GOOGLE_APPLICATION_CREDENTIALS=~/secrets/vsiddharth.json
export GOOGLE_MACHINE_TYPE="n1-standard-4"
export GOOGLE_PROJECT="kelvinxu-research"
export GOOGLE_ZONE="us-west1-a"
export GOOGLE_DISK_SIZE=50

# instance name, script, args
launch_program() {
    if [[ -e $BASE_LOGDIR ]]; then
	rmdir $BASE_LOGDIR
    fi

    script="sudo mkdir /log; sudo docker run -v /log:/log -d siddharthverma/adversarial $2 --logdir=/log --device=cpu ${@:3}"
    echo LAUNCHING: $1
    echo SCRIPT: $script
    docker-machine create $1
    docker-machine ssh $1 $script
}


# instance name, script, checkpoint, args
launch_hrl() {
    script="sudo mkdir /log; sudo mv ~/* /log; sudo docker run -v /log:/log -d siddharthverma/adversarial $2 --logdir=/log --checkpoint=/log/checkpoint.pkl --device=cpu ${@:4}"
    echo LAUNCHING: $1
    echo SCRIPT: $script
    docker-machine create $1
    docker-machine scp "$3/snapshot_999.pkl" "$1:~/checkpoint.pkl"
    docker-machine scp "$3/hyperparams.json" "$1:~/hyperparams.json"
    docker-machine ssh $1 $script
}

local_start_program() {
    ./$2 --logdir="$BASE_LOGDIR/$1" --device=cpu "${@:3}"
}




for bob in 200 250 300 350; do
    for rw in 0.5 1 1.5 2; do
	rwmul=$(python -c "print(int($rw * 100))")
	name="bob-$bob-rw-$rwmul-xy-f"
	start_program $name adv \
	    --env-name=AntResetFree-v4 \
	    --alice-path-length=200 --alice-train-repeats=200 \
	    --bob-path-length=$bob --bob-train-repeats=$bob \
	    --num-pretrain=1 \
	    --num-epochs=1 \
	    --reward-weight=$rw &

	name="bob-$bob-rw-$rwmul-xy-t"
	start_program $name adv \
	    --env-name=AntResetFree-v4 \
	    --alice-path-length=200 --alice-train-repeats=200 \
	    --bob-path-length=$bob --bob-train-repeats=$bob \
	    --num-pretrain=1 \
	    --num-epochs=1 \
	    --reward-weight=$rw --xy-prior &

    done
done


# WAYPOINT

for diayn in "free" "reset" "free-xy" "reset-xy"; do
    start_program_hrl "diayn-$diayn-waypoint" "/mnt/nfs/users/vsiddharth/adversarial_results/diayn baseline/$diayn/" \
		      --backward-steps=0 \
		      --forward-steps=200 &
done


start_program_hrl "ours-backward-waypoint" /mnt/nfs/users/vsiddharth/adversarial_results/tune_job_2/bob-200-rw-300-xy-f/ \
      --backward-steps=20 \
      --forward-steps=200 &

start_program_hrl "ours-regular-waypoint" /mnt/nfs/users/vsiddharth/adversarial_results/tune_job_2/bob-200-rw-300-xy-f/ \
      --backward-steps=0 \
      --forward-steps=200 &


for diayn in "free" "reset" "free-xy" "reset-xy"; do
    start_program_hrl "diayn-$diayn" "/mnt/nfs/users/vsiddharth/adversarial_results/diayn baseline/$diayn/" \
		      --backward-steps=0 \
		      --forward-steps=200 &
done

# MAZE

start_program_hrl "ours-backward" /mnt/nfs/users/vsiddharth/adversarial_results/tune_job_2/bob-200-rw-300-xy-f/ \
      --backward-steps=20 \
      --forward-steps=200 &

start_program_hrl "ours-regular" /mnt/nfs/users/vsiddharth/adversarial_results/tune_job_2/bob-200-rw-300-xy-f/ \
      --backward-steps=0 \
      --forward-steps=200 &
