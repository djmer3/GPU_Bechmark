#define ARRAY_SIZE 16
#pragma OPENCL EXTENSION cl_khr_fp64: enable

__kernel void mmult32d(__global double* data0, __global double* data1, __global double* output_data,
	__local double* matrix0, __local double* matrix1) {

	uint global_addr0, global_addr1, local_addr0, local_addr1, i;
	double out = 0;

	global_addr0 = get_global_id(0);
	global_addr1 = get_global_id(1);
	local_addr0 = get_local_id(0);
	local_addr1 = get_local_id(1);

	matrix0[local_addr0 + local_addr1*ARRAY_SIZE] = data0[local_addr0 + local_addr1*ARRAY_SIZE];
	matrix1[local_addr0 + local_addr1*ARRAY_SIZE] = data1[local_addr0 + local_addr1*ARRAY_SIZE];

	barrier(CLK_LOCAL_MEM_FENCE);

	for (i = 0; i<ARRAY_SIZE; i++){
		out = out + matrix0[i + local_addr1*ARRAY_SIZE] * matrix1[local_addr0 + i*ARRAY_SIZE];
	}
	output_data[local_addr0 + local_addr1*ARRAY_SIZE] = out;
}
