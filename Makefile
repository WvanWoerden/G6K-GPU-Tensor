SUBDIRS := kernel
KERNELHEADERS := $(wildcard kernel/*.h kernel/*.hpp kernel/*.inl cuda/*.h cuda/*.hpp cuda/*.inl)
KERNELSOURCES := $(wildcard kernel/*.cpp cuda/*.cpp cuda/*.cu)
PYTHONSOURCES := $(wildcard g6k/*.pxd g6k/*.pyx)

all: Makefile.local kernel/libG6K.so g6k/siever.so

rebuild:
	./rebuild.sh $(G6KREBUILDARGS)

Makefile.local:
	./rebuild.sh --onlyconf $(G6KREBUILDARGS)

kernel/libG6K.so: Makefile.local $(KERNELHEADERS) $(KERNELSOURCES)
	$(MAKE) -C kernel libG6K.so

g6k/siever.so: Makefile.local kernel/libG6K.so $(KERNELHEADERS) $(PYTHONSOURCES)
	-rm g6k/*.cpp g6k/*.so
	python setup.py clean
	python setup.py build_ext --inplace

clean: Makefile.local
	for dir in "${SUBDIRS}"; do make -C "$${dir}" clean; done
	-rm -rf build
	-rm -f g6k/*.cpp g6k/*.so

.PHONY: all $(SUBDIRS) clean
