#!/bin/bash

# Define arr_rates_to_simulate
arr_rates_to_simulate=(0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1)

# Get the current working directory and go one level up
current_directory=$(cd "$(dirname "$0")" && pwd -P)

# Go one level up from the script's directory
base_directory=$(cd "$current_directory/.." && pwd -P)

# Assign train_iter the value of 100,000
train_iter=500 #100000

# Check if the base directory exists
if [ -d "$base_directory" ]; then
    # Update the current directory to be the base directory
    cd "$base_directory"

    # Go one level back and then forward to the "data" folder
    #cd "../data"

    # Change working directory to "data"
    cd "data_version/version_41"
    echo "Current directory: $(pwd)"

    

    # Loop over arr_rates_to_simulate
    for arr_rate in "${arr_rates_to_simulate[@]}"; do
        # Create train_sim list from 1 to 10
        train_sim_list=($(seq 1 10))

        # Create sim_num list from 1 to 10
        sim_num=($(seq 1 51))

        # Loop over train_sim_list
        for train_sim in "${train_sim_list[@]}"; do
            # Loop over sim_num
            for sim_num_iter in "${sim_num[@]}"; do
                # Delete files within subfolders
                files="arr_${arr_rate}/test_homo_stream/train_sim_${train_sim}/train_iter_${train_iter}/pickobj_sim_${sim_num_iter}/*"
                rm -f $files

                echo "Files deleted successfully for $files"
            done
        done
    done

    echo "Files in $base_directory/data deleted successfully."
else
    echo "Base directory $base_directory does not exist."
fi
