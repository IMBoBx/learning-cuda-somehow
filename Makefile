NVCC := nvcc
CUDA_FLAGS := -arch=sm_86

%: %.cu
# 	@echo "Compiling $<..."
	@$(NVCC) $(CUDA_FLAGS) $< -o $@
# 	@echo "Running ./$@...\n"
	@./$@
# 	@echo "\nCleaning up ./$@...\n"
	@rm -f ./$@

	
