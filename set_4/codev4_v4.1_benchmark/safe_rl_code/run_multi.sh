#!/bin/bash

# run_multi.sh
#input_args=(0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1)
#input_args=(0.01)

#for arg in ${input_args[@]} 
#do
#    python3 main_multi.py --train fixed_argument --other_arg $arg &
#done
#wait
#echo "data collection -completed"
#if [ $? -ne 0 ]; then
#    echo "Error in Job 1. Exiting..."
#  exit 1
#fi
sleep 60m
python3 merge_replay_buffer.py
#if [ $? -ne 0 ]; then
#    echo "Error in Job 1. Exiting..."
#    exit 1
#fi


    
#input_args=(1 2 3 4 5 6 7 8 9 10)
#input_args=(0.01 ) #0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1)

#for arg in ${input_args[@]}
#do
    #python main_multi.py $arg1 $arg &
pthon3 train_with_next_state_on_merged_replay_buffer.py 0.1
    #python3 main_multi.py --train fixed_argument --other_arg $arg &
#done


#wait

#echo "process-completed"

