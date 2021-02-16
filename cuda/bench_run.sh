#!/bin/sh
for m in 0 1
do
    for b in 1024 2048 4096 8192 16384 32768 65536
    do
        CUDA_VISIBLE_DEVICES=0 numactl -N 0 -m 0 ./bench_sieving $b $m
        sleep 20 # to let GPU calm down
    done
done
