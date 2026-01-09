#!/bin/bash
echo "ngpus:n:time"
for ngpu in 1 2 4 8 ; do

    # Set HIP_VISIBLE_DEVICES based on number of GPUs
    case $ngpu in
        1)
            export HIP_VISIBLE_DEVICES=0
            ;;
        2)
            export HIP_VISIBLE_DEVICES=0,4
            ;;
        4)
            export HIP_VISIBLE_DEVICES=0,2,4,6
            ;;
        8)
            export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
            ;;
    esac

    for i in {13..24} ; do
        r=$(XKRT_NGPUS=$ngpu julia paper/benchmarks/axpy/axpy.jl $[2**i] $[2**i/ngpu] 2> /dev/null | grep seconds | tail -n +6)
        while IFS= read -r line; do
            time=$(echo "$line" | xargs | cut -d ' ' -f 1)
            echo "$ngpu:$[2**i]:$time"
        done <<< "$r"
    done
done
