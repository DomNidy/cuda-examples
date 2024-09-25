run:
	nvcc ./vector_add.cu -o vector_add
	powershell ./time.ps1 vector_add.exe

run_non_cuda:
	gcc ./cpu_vector_add.c -o cpu_vector_add.exe
	powershell ./time.ps1 cpu_vector_add.exe