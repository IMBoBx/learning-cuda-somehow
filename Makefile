NVCC := nvcc
CUDA_FLAGS := -arch=sm_86

%: %.cu
	@$(NVCC) $(CUDA_FLAGS) $< -o $@
	@./$@ $(ARGS)
	@rm -f ./$@

	
