ifeq (,$(wildcard ../Makefile.local))
$(shell cd ..; ./rebuild.sh --onlyconf >/dev/null; cd kernel)
endif
include ../Makefile.local

CXXFLAGS += -fPIC -Ofast -march=native -ftree-vectorize -funroll-loops -std=c++11 -pthread -Wall -Wextra $(EXTRAFLAGS) -I../parallel-hashmap
LDFLAGS += -shared -pthread -L$(realpath .) -Wl,-rpath=$(realpath .)
LIBADD += -lpthread

DEST = libG6K.so
OBJ = sieving.o control.o bgj1_sieve.o gpu_sieve.o triple_sieve.o params.o cpuperf.o triple_sieve_mt.o

# CUDA configuration
ifeq ($(HAVE_CUDA),1)
ifeq ($(CUDA_CXX),)
CUDA_CXX=$(CXX)
endif
SMS ?= 75
CUDACXXFLAGS = -fPIC -Ofast -march=native -pthread -Wall -Wextra $(EXTRAFLAGS)
NVCC_FLAGS += -ccbin $(CUDA_CXX) $(addprefix -Xcompiler ,$(CUDACXXFLAGS))
NVCC_FLAGS += -std=c++11 -O3
OBJ += ../cuda/GPUStreamGeneral.o
CXXFLAGS += $(CUDA_FLAGS)
LIBADD += $(CUDA_LIBS)
endif

ifeq ($(HAVE_CUDA),1)
# Build GENCODE FLAGS from SMS
GENCODE_FLAGS =
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))
HIGHEST_SM = $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif
NVCC_FLAGS += $(GENCODE_FLAGS)
endif


KERNELHEADERS = $(wildcard *.h *.hpp *.inl ../cuda/*.h ../cuda/*.hpp ../cuda/*.inl)

all: $(DEST)

$(DEST): $(OBJ) $(KERNELHEADERS) ../Makefile.local
	$(CXX) $(LDFLAGS) $(OBJ) -o $@ $(LIBADD)

%.o: %.cpp $(KERNELHEADERS) ../Makefile.local
	$(CXX) $(CXXFLAGS) -c $< -o $@

# CUDA build rule
%.o: %.cu $(KERNELHEADERS) ../Makefile.local
	$(NVCC) $(NVCC_FLAGS) $(CUDA_FLAGS) -c $< -o $@

clean:
	-rm $(OBJ) *.o *.so ../*.so

# for python setup.py to obtain final values of variables
printvariables:
	$(foreach v,$(.VARIABLES),$(info $(v) = $($(v))))
