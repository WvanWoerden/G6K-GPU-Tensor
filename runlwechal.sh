#!/bin/bash

gpus=3
threads=30
run=1
# run=seed

tag=lwe_try4

n=45
a=0.025

lweopts=""
#lweopts="$lweopts --lwe/svp_bkz_time_factor 1.0"
#lweopts="$lweopts --bkz/fpylll_crossover 50"
#lweopts="$lweopts --bkz/blocksizes X"

for chal in 85_0.005 70_0.010 60_0.015 90_0.005 45_0.030 55_0.020 50_0.025 75_0.010; do

	n=`echo $chal | cut -d_ -f1`
	a=`echo $chal | cut -d_ -f2`
	
	######### AUTOMATIC CONFIGURATION

	fileprefix=lwechallenge_runs/lwechal_${n}_${a}_${tag}
	mkdir lwechallenge_runs 2>/dev/null
	if python3 ./lwe_challenge.py 80 --show-defaults &>/dev/null ; then
		echo -n ""
	else
		echo "Rebuilding G6K-GPU first..."
		./rebuild.sh -y -j 10 &> /dev/null
	fi
	
	echo "LWE challenge n=$n alpha=$a, computing prediction..."
	lweparam=`python3 ./lwe_challenge.py $n --lwe/alpha ${a} $lweopts --threads $threads --gpus $gpus 2>/dev/null | head -n3 | grep -o "bkz-[0-9]*.*svp-[0-9]*"`
	if [ "$lweparam" = "" ]; then
		echo "No satisfying parameters found!"
		exit 1
	fi
	echo "LWE prediction: $lweparam"

	bkzdim=`echo $lweparam | cut -d- -f2 | grep -o "[0-9]*"`
	svpdim=`echo $lweparam | cut -d- -f3 | grep -o "[0-9]*"`
	
	oversieve=4 # how far to sieve extra, but without increasing db size

	dfreemin=`echo "$svpdim/l($svpdim) - $oversieve" | bc -l | cut -d'.' -f1`
	dhdff=$dfreemin

	dblimit=`echo "2.77 * e(l(4/3)*(($svpdim + $oversieve)/2))" | bc -l | cut -d'.' -f1`
	multibucket=2

	vecnum=`echo "sqrt( ${dblimit} ) * 2" | bc -l | cut -d'.' -f1`
	if [ $vecnum -ge 65536 ]; then
		vecnum=$(( (vecnum+1023)/1024 ))
		vecnum=$(( vecnum*1024 ))
	else
		vecnum=65536
	fi

	maxsievedim=$(($svpdim + $oversieve)) # - $dfreemin))
	if [ $maxsievedim -gt 128 ]; then
		maxsievedim=$(( ((maxsievedim+15)/16)*16 ))
	else
		maxsievedim=128
	fi


	######### REBUILD G6K for this run
	echo "./rebuild.sh -g -f -y -j 10 -m $maxsievedim --gpuvecnum $vecnum --with-cuda"
	sleep 10
#	exit
	
	./rebuild.sh -g -f -y -j 10 -m $maxsievedim --gpuvecnum $vecnum --with-cuda || exit 1
#	./rebuild.sh -g -f -y -j 10 -m $maxsievedim --gpuvecnum $vecnum

	######### EXECUTE G6K svp challenge run

#	dhargs="--dh_min $dhmin --dh_vecs 48 --dh_dim 24 --dh_d4f $dhdff"
	dhargs="--dh_vecs 0"

#	woargs="--workout/dim4free_min ${dfreemin} --workout/dim4free_dec 2 --pump/down_sieve True --pump/prefer_left_insert 1.2 --verbose --db_size_factor 2.77 --saturation_ratio .375 --db_limit ${dblimit} --multi_bucket ${multibucket}"
	woargs="--pump/prefer_left_insert 1.2 --verbose --pump/down_sieve True --db_limit ${dblimit} --multi_bucket ${multibucket}" # --db_size_factor 2.77 --saturation_ratio .375"

	opts="${n} --lwe/alpha ${a} ${lweopts} --gpus ${gpus} --threads ${threads} ${dhargs} ${woargs}" # --max_nr_buckets 0 --trace False"

	echo "Running: lwe_challenge.py $opts" #| tee ${fileprefix}.cout${run}.log
	cat .last_build >> ${fileprefix}.cout${run}.log
	/usr/bin/time -v -o ${fileprefix}.time${run}.log python3 ./lwe_challenge.py ${opts} 2> ${fileprefix}.cerr${run}.log | tee -a ${fileprefix}.cout${run}.log
#	gdb --args python3 ./lwe_challenge.py ${opts} 
	echo "Finished"
	tail ${fileprefix}.cout${run}.log -n30 | mail -s "${fileprefix}.cout" stevens@cwi.nl

#	sleep 10
#	exit
done
