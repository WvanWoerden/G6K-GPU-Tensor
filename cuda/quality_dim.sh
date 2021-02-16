#!/bin/sh
bucket_size=16384
mh=4
repeats=64

echo "n bucketer bsize mh repeats coll cost"
for n in 80 96 112 128 144
do
    ./bench_quality rand $bucket_size $mh $repeats $n
    ./bench_quality bgj1 $bucket_size $mh $repeats $n
    ./bench_quality bdgl $bucket_size $mh $repeats $n 1
    ./bench_quality bdgl $bucket_size $mh $repeats $n 2
done
