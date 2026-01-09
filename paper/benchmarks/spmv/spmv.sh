#!/bin/bash
echo "ngpus:n:density:time"
# for ngpu in 1 2 3 4 ; do
for ngpu in 1 ; do
    #for density in 0.001 0.01 0.1 ; do
    for density in 0.1 ; do
        for i in {12..18} ; do
            n=$[2**i]
            r=$(XKRT_NGPUS=$ngpu julia --project=. paper/benchmarks/spmv/spmv.jl $n $density 2> /dev/null | grep seconds | tail -n +6)
            while IFS= read -r line; do
                time=$(echo "$line" | xargs | cut -d ' ' -f 1)
                echo "$ngpu:$[2**i]:$density:$time"
            done <<< "$r"
        done
    done
done
