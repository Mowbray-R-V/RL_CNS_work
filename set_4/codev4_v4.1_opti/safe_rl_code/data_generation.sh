#!/bin/bash
#set -e

start=`date +%s%N`
input_args=(0.01)  #(0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1)

failed_args=()

handle_error() {
    echo "Error in Job with argument: $current_arg. Data collection failed."
    failed_args+=("$current_arg")
}

trap 'handle_error' ERR

for arg in "${input_args[@]}"; do
    current_arg="$arg"
    python3 main_multi.py --train fixed_argument --other_arg "$arg" &
done

wait

if [ ${#failed_args[@]} -eq 0 ]; then
    echo "Data collection completed."
else
    echo "Data collection failed for the following arguments: ${failed_args[@]}"
    exit 1
fi
end=`date +%s%N`
echo Execution time was `expr $end - $start` nanoseconds.
