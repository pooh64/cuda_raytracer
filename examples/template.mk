include ../../general.mk

.DEFAULT_GOAL := all
all: $(example)

example_src := $(wildcard *.cu)
example_obj := $(example_src:.cu=.o)
example_dep := $(example_obj:.o=.d)

%.o: %.cu
	$(CXX) $(CPPFLAGS) -c $< -o $@

%.d: %.cu
	@ set -e; rm -f $@; \
	$(CXX) -M $(CPPFLAGS) $< > $@.$$$$; \
	sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$

$(example): $(lib_obj) $(example_obj)
	$(CXX) -o $@ $^ $(LDFLAGS)

.PHONY: clean
clean:
	rm -f $(example_obj) $(example_dep) $(example)

-include $(lib_dep)
-include $(example_dep)
