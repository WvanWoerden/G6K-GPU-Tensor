#!/bin/sh
for b in 2048 4096 8192 16384 32768 65536
do
    for m in 1 2
    do
        ./bench_quality $b $m
    done
done
