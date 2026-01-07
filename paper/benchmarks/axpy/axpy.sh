#!/bin/bash
echo "ngpus:n:time"
for ngpu in 1 2 3 4 ; do
    for i in {13..24} ; do
        r=$(XKRT_NGPUS=$ngpu julia --project=. paper/benchmarks/axpy/axpy.jl $[2**i] $[2**i/ngpu] 2> /dev/null | grep seconds | tail -n +6)
        while IFS= read -r line; do
            time=$(echo "$line" | xargs | cut -d ' ' -f 1)
            echo "$ngpu:$[2**i]:$time"
        done <<< "$r"
    done
done
