#define ARRAY_SIZE 16

__kernel void mmult32s(__global float* data0, __global float* data1, __global float* output_data,
	__local float* matrix0, __local float* matrix1) {

	uint global_addr0, global_addr1, local_addr0, local_addr1, i;
	float out = 0;

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
