sed -i "s/^heuristic_dict_id = .*/heuristic_dict_id = 0/" ./config.py;


# start=$(date +%s%N)
python3 main_multi.py --train fixed_argument --other_arg  42
if [ $? -ne 0 ]; then
   echo "Error in Job 1. Exiting..."
   exit 1
fi
# end=$(date +%s%N)
# echo "Execution time was $(expr $end - $start) nanoseconds."



#bash del_sim.sh 
#python3 main_multi.py --test 500 41


#python3 stream_heuristic_stream_test_lower_arr_compare_post_processing_non-average.py;
#python3 pickle_movie_images.py

#python3 merge_replay_buffer.py;
#if [ $? -ne 0 ]; then
#    echo "Error in Job 1. Exiting..."
#    exit 1
#fi



#python3 train_with_next_state_on_merged_replay_buffer.py;
#bash del_sim.sh 
######## python3 main.py --test 50000;
#python3 main_multi.py --test 100000  #> output.txt
