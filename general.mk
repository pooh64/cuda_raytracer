SELF_DIR := $(dir $(lastword $(MAKEFILE_LIST)))

OBJDIR := $(SELF_DIR)/obj
LIBDIR := $(SELF_DIR)/lib

$(OBJDIR):
	mkdir -p $(OBJDIR)

CXX = nvcc
#LDFLAGS = -pthread
#DEPFLAGS = -MT $@ -MMD -MP -MF

CPPFLAGS = -std=c++14 -I$(SELF_DIR)/include
# CPPFLAGS += -O0
# CPPFLAGS += -mavx -Ofast -march=native -mtune=native -ftree-vectorize
# CPPFLAGS += -frename-registers -funroll-loops -ffast-math -fno-signed-zeros -fno-trapping-math

#CPPFLAGS += -Wall
CPPFLAGS += -g
#CPPFLAGS += -fsanitize=address
#LDFLAGS  += -fsanitize=address
#CPPFLAGS += -fsanitize=thread
#LDFLAGS  += -fsanitize=thread

$(OBJDIR)/%.o: $(LIBDIR)/%.cu | $(OBJDIR)
	$(CXX) $(CPPFLAGS) -c $< -o $@

$(OBJDIR)/%.d: $(LIBDIR)/%.cu
	@ set -e; rm -f $@;					\
	$(CXX) -M $(CPPFLAGS) $< > $@.$$$$;			\
	sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@;	\
	rm -f $@.$$$$

lib_src := $(wildcard $(LIBDIR)/*.cu)
lib_obj := $(lib_src:$(LIBDIR)/%.cu=$(OBJDIR)/%.o)
lib_dep := $(lib_obj:$(OBJDIR)/%.o=$(OBJDIR)/%.d)
