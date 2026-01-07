#!/bin/bash
echo "ngpus:n:density:time"
for ngpu in 1 4 ; do
    for density in 0.001 0.01 0.1 ; do
    for i in {12..20} ; do
        r=$(XKRT_NGPUS=$ngpu julia --project=. paper/benchmarks/spmv/spmv.jl $[2**i] $density 2> /dev/null | grep seconds | tail -n +6)
        while IFS= read -r line; do
            time=$(echo "$line" | xargs | cut -d ' ' -f 1)
            echo "$ngpu:$[2**i]:$density:$time"
        done <<< "$r"
    done
    done
done
