#define ARRAY_SIZE 16
#define OUTPUT_FILE "file1.txt"

long int shortest_time, ave_time;

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <windows.h>

#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

/* Find a GPU or CPU associated with the first available platform */
cl_device_id create_device() {

	cl_device_id dev;
	cl_platform_id platform[10];
	int devices = 0, platforms;
	int err;

	/* Identify a platform */
	err = clGetPlatformIDs(4, platform, &platforms);
	if (err < 0) {
		perror("Couldn't identify a platform");
		exit(1);
	}
	
	/* Access a device */
	err = clGetDeviceIDs(platform[2], CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
	if (err == CL_DEVICE_NOT_FOUND) {
		err = clGetDeviceIDs(platform[2], CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
	}
	if (err < 0) {
		perror("Couldn't access any devices");
		exit(1);
	}

	/*Print Device Extensions*/
	/*clGetDeviceInfo(dev, CL_DEVICE_EXTENSIONS, sizeof(buffer), buffer, NULL);
	printf("\n  DEVICE_EXTENSIONS = %s\n\n", buffer);

	/*Print Max Work Item Sizes*/
	/*size_t sizes[3];
	clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(sizes), sizes, NULL);
	printf("\n  Work Item Sizes = %d  %d  %d\n\n", sizes[0], sizes[1], sizes[2]);

	clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(sizes), sizes, NULL);
	printf("\n  Max Workgoup Size = %d\n\n", sizes[0]);
	*/
	return(dev);
}
/* Create program from a file and compile it */
cl_program build_program(cl_context ctx, cl_device_id dev, const char* filename) {

	cl_program program;
	FILE *program_handle;
	char *program_buffer, *program_log;
	size_t program_size, log_size;
	int err;
	errno_t err_s;
	/* Read program file and place content into buffer */
	//program_handle = fopen(filename, "r");
	err_s = fopen_s(&program_handle, filename, "r");
	if (program_handle == NULL) {
		perror("Couldn't find the program file");
		exit(1);
	}
	fseek(program_handle, 0, SEEK_END);
	program_size = ftell(program_handle);
	rewind(program_handle);
	program_buffer = (char*)malloc(program_size + 1);
	program_buffer[program_size] = '\0';
	fread(program_buffer, sizeof(char), program_size, program_handle);
	fclose(program_handle);

	/* Create program from file */
	program = clCreateProgramWithSource(ctx, 1,
		(const char**)&program_buffer, &program_size, &err);
	if (err < 0) {
		perror("Couldn't create the program");
		exit(1);
	}
	free(program_buffer);

	/* Build program */
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err < 0) {

		/* Find size of log and print to std output */
		clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
			0, NULL, &log_size);
		program_log = (char*)malloc(log_size + 1);
		program_log[log_size] = '\0';
		clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
			log_size + 1, program_log, NULL);
		printf("%s\n", program_log);
		free(program_log);
		getchar();
		exit(1);
	}

	return program;
}

int mmult32s(cl_device_id device, cl_context context, const char* filename, const char* kernel_func, int loops, int n,int mults){
	/*OpenCL structures*/
	cl_command_queue queue;
	cl_kernel kernel;
	cl_program program;

	cl_int i, j, err;
	size_t local_size[2], global_size[2];
	cl_event event;
	cl_command_queue_properties properties;
	
	cl_ulong time_start, time_end;
	long int total_time;
	ave_time = 0; shortest_time = 0;

	/* Data and buffers */
	cl_mem input_buffer0, input_buffer1, output_buffer;
	float *matrix0; 
	float *matrix1; 
	float *output_matrix; 
	matrix0 = (float*)calloc(ARRAY_SIZE*ARRAY_SIZE*n,sizeof(float));
	matrix1 = (float*)calloc(ARRAY_SIZE*ARRAY_SIZE*n, sizeof(float));
	output_matrix = (float*)calloc(ARRAY_SIZE*ARRAY_SIZE*n, sizeof(float));
	
	/* Build program */
	program = build_program(context, device, filename);

	/* Initialize data */
	for (int i = 0; i < ARRAY_SIZE*ARRAY_SIZE; i++) {
		matrix0[i*n] = (i % ARRAY_SIZE);
		for (int j = 0; j < n; j++){
			matrix1[i*n+j] = 0.0;
		}
		if (!(i % (ARRAY_SIZE+1))){
			matrix1[i*n] = 1;
		}
	}

	/* Create data buffer */
	global_size[0] = ARRAY_SIZE;
	global_size[1] = ARRAY_SIZE*mults;
	local_size[0] = ARRAY_SIZE;
	local_size[1] = ARRAY_SIZE;

	input_buffer0 = clCreateBuffer(context, CL_MEM_READ_ONLY |
		CL_MEM_COPY_HOST_PTR, ARRAY_SIZE*ARRAY_SIZE * n * sizeof(float), matrix0, &err);
	input_buffer1 = clCreateBuffer(context, CL_MEM_READ_ONLY |
		CL_MEM_COPY_HOST_PTR, ARRAY_SIZE*ARRAY_SIZE * n * sizeof(float), matrix1, &err);
	output_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE |
		CL_MEM_COPY_HOST_PTR, ARRAY_SIZE*ARRAY_SIZE * n * sizeof(float), output_matrix, &err);
	if (err < 0) {
		perror("Couldn't create a buffer");
		exit(1);
	};

	/* Create a command queue */
	properties = CL_QUEUE_PROFILING_ENABLE;
	//properties |= CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
	queue = clCreateCommandQueue(context, device, properties, &err);
	if (err < 0) {
		perror("Couldn't create a command queue");
		exit(1);
	};

	/* Create a kernel */
	kernel = clCreateKernel(program, kernel_func, &err);
	if (err < 0) {
		perror("Couldn't create a kernel");
		exit(1);
	};

	/* Create kernel arguments */
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buffer0);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &input_buffer1);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &output_buffer);
	err |= clSetKernelArg(kernel, 3, local_size[0] * local_size[1] * sizeof(float)*n, NULL);
	err |= clSetKernelArg(kernel, 4, local_size[0] * local_size[1] * sizeof(float)*n, NULL);
	if (err < 0) {
		perror("Couldn't create a kernel argument");
		exit(1);
	}
	for (int i = 0; i < loops; i++){
		/* Enqueue kernel */

		clFinish(queue);
		err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, &global_size,
			&local_size, 0, NULL, &event);
		if (err < 0) {
			perror("Couldn't enqueue the kernel");
			exit(1);
		}
		clWaitForEvents(1, &event);

		clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
		clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
		total_time = time_end - time_start;
		ave_time = ave_time + total_time / loops;
		if ((total_time < shortest_time) || (shortest_time==0)){
			shortest_time = total_time;
		}

		/* Read the kernel's output */
		err = clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0,
			ARRAY_SIZE*ARRAY_SIZE * sizeof(float)*n, output_matrix, 0, NULL, NULL);
		if (err < 0) {
			perror("Couldn't read the buffer");
			exit(1);
		}
	}
	/*Check Result*/
	/*for (i = 0; i < ARRAY_SIZE*ARRAY_SIZE; i++) {
	printf("%4.0f ", (output_matrix[i].x + output_matrix[i].y));
	if (i != 0){
	if (!((i + 1) % 32)){
	printf("\n");
	}
	if (!((i + 1) % 1024)){
	printf("\n");
	}
	}
	}*/
	/* Deallocate resources */
	free(matrix0);
	free(matrix1);
	free(output_matrix);
	clReleaseMemObject(output_buffer);
	clReleaseMemObject(input_buffer0);
	clReleaseMemObject(input_buffer1);
	clReleaseCommandQueue(queue);

	clReleaseKernel(kernel);
	clReleaseProgram(program);
	return 0;
}

int mmult32d(cl_device_id device, cl_context context, const char* filename, const char* kernel_func, int loops, int n, int mults){
	/*OpenCL structures*/
	cl_command_queue queue;
	cl_kernel kernel;
	cl_program program;

	cl_int i, j, err;
	size_t local_size[2], global_size[2];
	cl_event event;
	cl_command_queue_properties properties;

	cl_ulong time_start, time_end;
	long int total_time;
	ave_time = 0; shortest_time = 0;

	/* Data and buffers */
	cl_mem input_buffer0, input_buffer1, output_buffer;
	double *matrix0;
	double *matrix1;
	double *output_matrix;
	matrix0 = (double*)calloc(ARRAY_SIZE*ARRAY_SIZE*n, sizeof(double));
	matrix1 = (double*)calloc(ARRAY_SIZE*ARRAY_SIZE*n, sizeof(double));
	output_matrix = (double*)calloc(ARRAY_SIZE*ARRAY_SIZE*n, sizeof(double));

	/* Build program */
	program = build_program(context, device, filename);

	/* Initialize data */
	for (int i = 0; i < ARRAY_SIZE*ARRAY_SIZE; i++) {
		matrix0[i*n] = (i % ARRAY_SIZE);
		for (int j = 0; j < n; j++){
			matrix1[i*n + j] = 0.0;
		}
		if (!(i % (ARRAY_SIZE+1))){
			matrix1[i*n] = 1;
		}
	}

	/* Create data buffer */
	global_size[0] = ARRAY_SIZE;
	global_size[1] = ARRAY_SIZE*mults;
	local_size[0] = ARRAY_SIZE;
	local_size[1] = ARRAY_SIZE;

	input_buffer0 = clCreateBuffer(context, CL_MEM_READ_ONLY |
		CL_MEM_COPY_HOST_PTR, ARRAY_SIZE*ARRAY_SIZE * n * sizeof(double), matrix0, &err);
	input_buffer1 = clCreateBuffer(context, CL_MEM_READ_ONLY |
		CL_MEM_COPY_HOST_PTR, ARRAY_SIZE*ARRAY_SIZE * n * sizeof(double), matrix1, &err);
	output_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE |
		CL_MEM_COPY_HOST_PTR, ARRAY_SIZE*ARRAY_SIZE * n * sizeof(double), output_matrix, &err);
	if (err < 0) {
		perror("Couldn't create a buffer");
		exit(1);
	};

	/* Create a command queue */
	properties = CL_QUEUE_PROFILING_ENABLE;
	//properties |= CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
	queue = clCreateCommandQueue(context, device, properties, &err);
	if (err < 0) {
		perror("Couldn't create a command queue");
		exit(1);
	};

	/* Create a kernel */
	kernel = clCreateKernel(program, kernel_func, &err);
	if (err < 0) {
		perror("Couldn't create a kernel");
		exit(1);
	};

	/* Create kernel arguments */
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buffer0);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &input_buffer1);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &output_buffer);
	err |= clSetKernelArg(kernel, 3, local_size[0] * local_size[1] * sizeof(double)*n, NULL);
	err |= clSetKernelArg(kernel, 4, local_size[0] * local_size[1] * sizeof(double)*n, NULL);
	if (err < 0) {
		perror("Couldn't create a kernel argument");
		exit(1);
	}
	for (int i = 0; i < loops; i++){
		/* Enqueue kernel */

		clFinish(queue);
		err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, &global_size,
			&local_size, 0, NULL, &event);
		if (err < 0) {
			perror("Couldn't enqueue the kernel");
			exit(1);
		}
		clWaitForEvents(1, &event);

		clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
		clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
		total_time = time_end - time_start;
		ave_time = ave_time + total_time / loops;
		if ((total_time < shortest_time) || (shortest_time == 0)){
			shortest_time = total_time;
		}

		/* Read the kernel's output */
		err = clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0,
			ARRAY_SIZE*ARRAY_SIZE * sizeof(double)*n, output_matrix, 0, NULL, NULL);
		if (err < 0) {
			perror("Couldn't read the buffer");
			exit(1);
		}
	}
	/*Check Result*/
	/*for (i = 0; i < ARRAY_SIZE*ARRAY_SIZE; i++) {
	printf("%4.0f ", (output_matrix[i].x + output_matrix[i].y));
	if (i != 0){
	if (!((i + 1) % 32)){
	printf("\n");
	}
	if (!((i + 1) % 1024)){
	printf("\n");
	}
	}
	}*/
	/* Deallocate resources */
	free(matrix0);
	free(matrix1);
	free(output_matrix);
	clReleaseMemObject(output_buffer);
	clReleaseMemObject(input_buffer0);
	clReleaseMemObject(input_buffer1);
	clReleaseCommandQueue(queue);

	clReleaseKernel(kernel);
	clReleaseProgram(program);
	return 0;
}

int main() {
	/*Resize Console Window */
	HWND wh = GetConsoleWindow();
	MoveWindow(wh, 10, 10, 1400, 700, TRUE);
	clock_t start_time, finish_time;
	start_time = clock();

	/* OpenCL structures */
	cl_device_id device;
	cl_context context;
	cl_int err;
	int err2;
	

	/*Output File*/
	errno_t file_err;
	FILE *fp;
	file_err = fopen_s(&fp, OUTPUT_FILE, "w+");
	if (file_err != 0){
		printf("error opening file");
	}

	/* Create device */
	char buffer[240];
	device = create_device();
	clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(buffer), buffer, NULL);
	fprintf(fp,"Device:\t%s\n", buffer);

	/* Create Context */
	context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
	if (err < 0) {
		perror("Couldn't create a context");
		exit(1);
	}
	
	fprintf(fp, "Matrix Order: %d\n", ARRAY_SIZE);
	int i = 1;
	fprintf(fp, "\nSingle precision\n");
	fprintf(fp, "Multiplications\tshortest(ns)\taverage(ns)\n");
	
	for (i = 1; i < 101; i++){
		err2 = mmult32s(device, context, "mmult32s.cl", "mmult32s", 1000, 1,i);
		fprintf(fp, "%d\t%li\t%li\n",i, shortest_time, ave_time);
	}
	
	fprintf(fp, "\nDouble-Single precision\n");
	fprintf(fp, "Multiplications\tshortest(ns)\taverage(ns)\n");
	for (i = 1; i < 101; i++){
		err2 = mmult32s(device, context, "mmult32ds.cl", "mmult32ds", 1000, 2, i);
		fprintf(fp, "%d\t%li\t%li\n", i, shortest_time, ave_time);
	}

	fprintf(fp, "\nQuad-Single precision\n");
	fprintf(fp, "Multiplications\tshortest(ns)\taverage(ns)\n");
	for (i = 1; i < 101; i++){
		err2 = mmult32s(device, context, "mmult32qs.cl", "mmult32qs", 1000, 4, i);
		fprintf(fp, "%d\t%li\t%li\n", i, shortest_time, ave_time);
	}

	fprintf(fp, "\nDouble precision\n");
	fprintf(fp, "Multiplications\tshortest(ns)\taverage(ns)\n");
	for (i = 1; i < 101; i++){
		err2 = mmult32d(device, context, "mmult32d.cl", "mmult32d", 1000, 1, i);
		fprintf(fp, "%d\t%li\t%li\n", i, shortest_time, ave_time);
	}

	fprintf(fp, "\nDouble-Double precision\n");
	fprintf(fp, "Multiplications\tshortest(ns)\taverage(ns)\n");
	for (i = 1; i < 101; i++){		
		err2 = mmult32d(device, context, "mmult32dd.cl", "mmult32dd", 1000, 2, i);
		fprintf(fp, "%d\t%li\t%li\n", i, shortest_time, ave_time);
	}
	
	fprintf(fp, "\Quad-Double precision\n");
	fprintf(fp, "Multiplications\tshortest(ns)\taverage(ns)\n");
	for (i = 1; i < 101; i++){
		err2 = mmult32d(device, context, "mmult32qd.cl", "mmult32qd", 1000, 4, i);
		fprintf(fp, "%d\t%li\t%li\n", i, shortest_time, ave_time);
	}
	/* Finish */
	finish_time = clock();
	clReleaseContext(context);
	fclose(fp);
	printf("DONE\n");
	printf("Total time taken (s): %d", ((finish_time-start_time)/CLOCKS_PER_SEC));
	getchar();
	return 0;
}
