start=$(date +%s%N)

# Define a function to execute run_tr-te.sh, print current directory, and handle errors
execute_script() {
    current_dir=$(pwd)
    echo "Current directory: $current_dir"
    bash run_tr-te.sh # Assuming the script is in the current directory, change if necessary
    if [ $? -ne 0 ]; then
        echo "Error in $1. Exiting..."
        exit 1
    fi
}

# Execute the script four times in parallel
(
    # cd ../code_v1
    execute_script "Job 4"
) &
(
    cd ../codev3_v5.1
    execute_script "Job 5"
) &
(
    cd ../codev3_v6.1
    execute_script "Job 6"
)
 
wait
end=$(date +%s%N)
echo "Execution time was $(expr $end - $start) nanoseconds."


