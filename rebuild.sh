#!/bin/bash

echo "Git status: " > .last_build
git log --pretty=short | head -n5 >> .last_build
git status | grep "modified" >> .last_build
echo "=================" >> .last_build
echo "rebuild.sh $*" >> .last_build
echo "=================" >> .last_build

enable_cpucounters=0
enable_stats=0
enable_ndebug=0
maxsievingdim=128
gpuvecnum=65536
enable_ggdb=0
enable_jobs=0
enable_templated_dim=0
enable_popcount=1
enable_yr=1
jobs=4

while [[ $# -gt 0 ]]; do
	case "$1" in
		-m|--maxsievingdim)
			maxsievingdim=$2
			shift
			;;
		--gpuvecnum)
			gpuvecnum=$2
			shift
			;;
		-j|--jobs)
			jobs=$2
			shift
			;;
		-y|--noyr)
			enable_yr=0
			enable_popcount=0
			;;
		-g|--ggdb)
			enable_ggdb=1
			;;
		-c|--cpucounters)
			enable_cpucounters=1
			;;
		-s|--stats)
			enable_stats=1
			;;
                -ss|--extended_stats)
                        enable_stats=2
                        ;;
		-t|--templated_dim)
			enable_templated_dim=1
			;;
		-f|--fast)
			enable_ndebug=1
			enable_cpucounters=0
			enable_stats=0
			enable_templated_dim=1
			;;
		-p|--no_popcount)
			enable_popcount=0
			;;
		--with-cuda)
			enable_cuda=$2
			shift
			;;
		--build-threshold)
			build_threshold=$2
			shift
			;;
		--sieve-threshold)
			sieve_threshold=$2
			shift
			;;
		--onlyconf)
			only_conf=1
			;;
		*)
			;;
	esac
	shift
done

EXTRAFLAGS=""
if [ ${enable_yr} -eq 0 ]; then
    EXTRAFLAGS="$EXTRAFLAGS -DNOYR=1"
fi
if [ ${enable_popcount} -eq 0 ]; then
    EXTRAFLAGS="$EXTRAFLAGS -DPOPCOUNT=0"
fi
if [ ${enable_ggdb} -eq 1 ]; then
	EXTRAFLAGS="$EXTRAFLAGS -g -ggdb"
fi
if [ ${enable_cpucounters} -eq 1 ]; then
	EXTRAFLAGS="$EXTRAFLAGS -DPERFORMANCE_COUNTING"
fi
if [ ${enable_stats} -ge 1 ]; then
	EXTRAFLAGS="$EXTRAFLAGS -DENABLE_STATS"
fi
if [ ${enable_stats} -ge 2 ]; then
	EXTRAFLAGS="$EXTRAFLAGS -DENABLE_EXTENDED_STATS"
fi
if [ ${enable_ndebug} -eq 1 ]; then
	EXTRAFLAGS="$EXTRAFLAGS -DNDEBUG"
fi
if [ ${enable_templated_dim} -eq 1 ]; then
	EXTRAFLAGS="$EXTRAFLAGS -DTEMPLATED_DIM"
fi
if [ ${maxsievingdim} -gt 0 ]; then
	EXTRAFLAGS="$EXTRAFLAGS -DMAX_SIEVING_DIM=${maxsievingdim}"
fi
if [ "$build_threshold" != "" ]; then
	EXTRAFLAGS="$EXTRAFLAGS -DXPC_BUCKET_THRESHOLD=${build_threshold}"
fi
if [ "$sieve_threshold" != "" ]; then
	EXTRAFLAGS="$EXTRAFLAGS -DXPC_THRESHOLD=${sieve_threshold}"
fi
if [ "$gpuvecnum" != "" ]; then
	EXTRAFLAGS="$EXTRAFLAGS -DGPUVECNUM=${gpuvecnum}"
fi

### CUDA: CONFIGURE OPTIONS ###
want_cuda="maybe"
have_nvcc="no"
if [ "$CUDAPATH" = "" ]; then
	cuda_path="/usr/local/cuda"
else
	cuda_path="$CUDAPATH"
fi
if [ "${enable_cuda}" = "no" ]; then
	want_cuda="no"
elif [ "${enable_cuda}" = "yes" ]; then
	want_cuda="yes"
elif [ "${enable_cuda}" != "" ]; then
	want_cuda="yes"
	cuda_path="${enable_cuda}"
fi
echo "want_cuda: ${want_cuda}" #&>> cuda_debug.log
echo "cuda_path: ${cuda_path}" #&>> cuda_debug.log
### CUDA: FIND NVCC
if [ "${want_cuda}" != "no" ]; then
	if [ "${cuda_path}" != "" ]; then
		PATH=${PATH}:${cuda_path}/bin
	fi
	NVCC=$(which nvcc)
	if [ "${NVCC}" != "" ]; then
		have_nvcc="yes"
	fi
fi
echo "nvcc_path: ${NVCC}" #&>> cuda_debug.log
if [ "${have_nvcc}" = "yes" ]; then
	NVCC_VERSION=`${NVCC} --version | grep release | awk 'gsub(/,/, "") {print $5}'`
	echo "nvcc version: ${NVCC_VERSION}" #&>> cuda_debug.log

	cuda_libdir=lib
	if [ `uname -m` = x86_64 ]; then
		cuda_libdir=lib64
	fi
	if [ `uname -m` = powerpc64le ]; then
		cuda_libdir=lib64
	fi

	CUDA_FLAGS="-lineinfo -I${cuda_path}/include -I../parallel-hashmap"
	CUDA_LIBS="-L${cuda_path}/${cuda_libdir} -Wl,-rpath=${cuda_path}/${cuda_libdir} -lcudart -L${cuda_path}/${cuda_libdir}/stubs -Wl,-rpath=${cuda_path}/${cuda_libdir}/stubs -lcuda -lcublas -lcurand"
	echo "CUDA_FLAGS: ${CUDA_FLAGS}" #&>> cuda_debug.log
	echo "CUDA_LIBS: ${CUDA_LIBS}" #&>> cuda_debug.log

	EXTRAFLAGS="$EXTRAFLAGS -DHAVE_CUDA"

	CUDA_CXX=""
	HAVE_CUDA=1
else
	HAVE_CUDA=0
fi
### END CUDA SECTION


# write Makefile.local
cat >Makefile.local <<EOF
EXTRAFLAGS=${EXTRAFLAGS}
HAVE_CUDA=${HAVE_CUDA}
CUDA_PATH=${cuda_path}
NVCC=${NVCC}
CUDA_FLAGS=${CUDA_FLAGS}
CUDA_LIBS=${CUDA_LIBS}
CUDA_CXX=${CUDA_CXX}
EOF

if [ "${only_conf}" = "1" ]; then
	exit
fi

[ -d parallel-hashmap ] || git clone https://github.com/cr-marcstevens/parallel-hashmap

rm -r build *.so g6k/*.so cuda/*.so kernel/*.so `find g6k -name "*.pyc"`
python3 setup.py clean
make -C kernel clean || exit 1

make -C kernel -j ${jobs} || exit 1

python3 setup.py build_ext --inplace -j ${jobs} || python setup.py build_ext --inplace
