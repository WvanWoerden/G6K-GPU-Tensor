#!/bin/sh
n=128
bucket_size=16384

echo "n bucketer bsize mh repeats coll cost"
for mh in 1 2 4 8 16
do
    repeats=$((256 / mh))


    ./bench_quality rand $bucket_size $mh $repeats $n
    ./bench_quality bgj1 $bucket_size $mh $repeats $n
    ./bench_quality bdgl $bucket_size $mh $repeats $n 1
    ./bench_quality bdgl $bucket_size $mh $repeats $n 2
done
