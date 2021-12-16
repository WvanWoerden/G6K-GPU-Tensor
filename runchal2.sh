#!/bin/bash

gpus=4
threads=40
run=1
# run=seed

tag=rampup8_dh32_highprecd

dim=140

loadmatrix=""
#loadmatrix="--load_matrix <file> --workout/start_n <firstsievedim>"

for ((; dim<=180; dim+=2)); do
	
	######### AUTOMATIC CONFIGURATION

	fileprefix=svpchallenge_runs/svpchal_${dim}_${tag}
	mkdir svpchallenge_runs 2>/dev/null

	dhmin=106 #$(($dim-36))
	oversieve=4 # how far to sieve extra, but without increasing db size

	dfreemin=`echo "$dim/l($dim) - $oversieve" | bc -l | cut -d'.' -f1`
	dhdff=$dfreemin

	dblimit=`echo "2.77 * e(l(4/3)*(($dim - $dfreemin - $oversieve)/2))" | bc -l | cut -d'.' -f1`

	vecnum=`echo "sqrt( ${dblimit} * 2 ) * 2" | bc -l | cut -d'.' -f1`
	if [ $vecnum -ge 65536 ]; then
		vecnum=$(( (vecnum+1023)/1024 ))
		vecnum=$(( vecnum*1024 ))
	else
		vecnum=65536
	fi

	maxsievedim=$(($dim - $dfreemin))
	if [ $maxsievedim -gt 128 ]; then
		maxsievedim=$(( ((maxsievedim+15)/16)*16 ))
	else
		maxsievedim=128
	fi


	######### REBUILD G6K for this run
	echo "./rebuild.sh -f -y -j 10 -m $maxsievedim --gpuvecnum $vecnum"
	
	./rebuild.sh -f -y -j 10 -m $maxsievedim --gpuvecnum $vecnum
	

	######### EXECUTE G6K svp challenge run

	dhargs="--dh_min $dhmin --dh_vecs 32 --dh_dim 24 --dh_d4f $dhdff"
#	dhargs="--dh_vecs 0"

	woargs="--workout/dim4free_min ${dfreemin} --workout/dim4free_dec 2 --pump/down_sieve True --pump/prefer_left_insert 1.2 --verbose --db_size_factor 2.77 --saturation_ratio .375 --db_limit ${dblimit} --multi_bucket 2"

	opts="${dim} --seed ${run} --gpus ${gpus} --threads ${threads} ${dhargs} ${woargs} --max_nr_buckets 0 --trace False ${loadmatrix}"
	loadmatrix="" # reset loadmatrix after use


	echo "Running: svp_challenge.py $opts" | tee ${fileprefix}.cout${run}.log
	cat .last_build >> ${fileprefix}.cout${run}.log
	python3 ./svp_challenge.py ${opts} --workout/save_prefix ${fileprefix}.mat${run} 2> ${fileprefix}.cerr${run}.log | tee -a ${fileprefix}.cout${run}.log
	echo "Finished"

	exit
	sleep 10
done
