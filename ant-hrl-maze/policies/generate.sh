#!/usr/bin/fish

set -l logdir /store/vsiddharth/ray_results/gym/Ant/v4/rebuttal_num_skills
#set -l logdir /store/vsiddharth/ray_results/gym/Ant/v4/2020-08-11T17-17-36-2020-08-11T17-17-36
set -l policydir $HOME/research/ant-hrl-maze/policies/

# clear out everything
rm $policydir/ours_rebuttal_*/*

# link the stuff
ln -s $logdir/tune_num_skills_1*/checkpoint_680/checkpoint.pkl $policydir/ours_rebuttal_1/
ln -s $logdir/tune_num_skills_1*/checkpoint_680/checkpoint.pkl $policydir/ours_rebuttal_2/
ln -s $logdir/tune_num_skills_1*/checkpoint_680/checkpoint.pkl $policydir/ours_rebuttal_3/
ln -s $logdir/tune_num_skills_1*/checkpoint_680/checkpoint.pkl $policydir/ours_rebuttal_4/

for i in (seq 1 4)
    python softlearning_to_policy.py ours_rebuttal_$i
end
