#!/bin/sh
n=128
mh=4
repeats=64

echo "n bucketer bsize mh repeats coll cost"
for bucket_size in 1024 2048 4096 8192 16384 32768
do
    ./bench_quality rand $bucket_size $mh $repeats $n
    ./bench_quality bgj1 $bucket_size $mh $repeats $n
    ./bench_quality bdgl $bucket_size $mh $repeats $n 1
    ./bench_quality bdgl $bucket_size $mh $repeats $n 2
done
