# Compiler
GPU_TARGET ?= CDNA3

TARGET=dsv4_decode_kernel
SRC=kernel.cpp

THUNDERKITTENS_ROOT ?= $(abspath ../../../)

ROCM_INSTALL_DIR := $(ROCM_PATH)
HIP_INCLUDE_DIR  := $(ROCM_INSTALL_DIR)/include/hip
ROCM_INCLUDE_DIR := $(ROCM_INSTALL_DIR)/include

HIPCXX ?= $(ROCM_INSTALL_DIR)/bin/hipcc

ifeq ($(GPU_TARGET),CDNA3)
HIPFLAGS += -DKITTENS_CDNA3 --offload-arch=gfx942
else ifeq ($(GPU_TARGET),CDNA4)
HIPFLAGS += -DKITTENS_CDNA4 --offload-arch=gfx950
endif

CXX_STD := c++20
ICXXFLAGS := -std=$(CXX_STD) -w
ICPPFLAGS := -I$(THUNDERKITTENS_ROOT)/include -I$(ROCM_INCLUDE_DIR) -I$(HIP_INCLUDE_DIR)
ICXXFLAGS += $(shell python3 -m pybind11 --includes) $(shell python3-config --ldflags | sed 's/-lcrypt//g') -shared -fPIC
ICXXFLAGS += -Rpass-analysis=kernel-resource-usage

all: $(TARGET)

$(TARGET): $(SRC)
	$(HIPCXX) $(SRC) $(HIPFLAGS) $(ICXXFLAGS) $(ICPPFLAGS) \
	    -o $(TARGET)$(shell python3-config --extension-suffix)

clean:
	rm -f $(TARGET)$(shell python3-config --extension-suffix)
