#pragma OPENCL EXTENSION cl_khr_select_fprounding_mode : enable
#define __ROUNDING_MODE__ rte
#define ARRAY_SIZE 16
typedef float type;
typedef float2 type2;

type2 make_dd(type a, type b){
	type2 c;
	c.s0 = a;
	c.s1 = b;

	return c;
}

type2 add_d_to_dd(type a, type b){
	type t1, t2;
	type2 c;

	c.s1 = a + b;
	t1 = c.s1 - a;
	t2 = c.s1 - t1;
	t1 = a - t2;
	t2 = t1 + t2;
	c.s0 = t1 + t2;

	return c;
}

type2 mul_d_to_dd(type a, type b){
	type2 c;

	c.s1 = a*b;
	c.s0 = fma(a, b, -c.s1);

	return c;
}

type2 add_dd(type2 a, type2 b){
	type2 c;
	type t1, t2, t3, t4, t5, e;

	t1 = a.s1 + b.s1;
	t2 = t1 - a.s1;
	t3 = (a.s1 + (t2 - t1)) + (b.s1 - t2);
	t4 = a.s0 + b.s0;
	t2 = t4 - a.s0;
	t5 = (a.s0 + (t2 - t4)) + (b.s0 - t2);
	t3 = t3 + t4;
	t4 = t1 + t3;
	t3 = (t1 - t4) + t3;
	t3 = t3 + t5;
	c.s1 = e = t4 + t3;
	c.s0 = (t4 - e) + t3;

	return c;
}

type2 neg_dd(type2 a){
	type2 b;
	b.s1 = -a.s1;
	b.s0 = -a.s0;

	return b;
}

type2 sub_dd(type2 a, type2 b){
	type2 c;
	type t1, t2, t3, t4, t5, e;

	t1 = a.s1 - b.s1;
	t2 = t1 - a.s1;
	t3 = (a.s1 + (t2 - t1)) - (b.s1 + t2);
	t4 = a.s0 - b.s0;
	t2 = t4 - a.s0;
	t5 = (a.s0 + (t2 - t4)) - (b.s0 + t2);
	t3 = t3 + t4;
	t4 = t1 + t3;
	t3 = (t1 - t4) + t3;
	t3 = t3 + t5;
	c.s1 = e = t4 + t3;
	c.s0 = (t4 - e) + t3;

	return c;
}

type2 mul_dd(type2 a, type2 b){
	type2 t, c;
	type e;

	t.s1 = a.s1 * b.s1;
	t.s0 = fma(a.s1, b.s1, -t.s1);
	t.s0 = fma(a.s0, b.s0, t.s0);
	t.s0 = fma(a.s1, b.s0, t.s0);
	t.s0 = fma(a.s0, b.s1, t.s0);
	c.s1 = e = t.s1 + t.s0;
	c.s0 = (t.s1 - e) + t.s0;

	return c;
}

type2 div_dd(type2 a, type2 b){
	type2 t, c;
	type e, r;

	r = 1.0 / b.s1;
	t.s1 = a.s1 * r;
	e = fma(b.s1, -t.s1, a.s1);
	t.s1 = fma(r, e, t.s1);
	t.s0 = fma(b.s1, -t.s1, a.s1);
	t.s0 = a.s0 + t.s0;
	t.s0 = fma(b.s0, -t.s1, t.s0);
	e = r * t.s0;
	t.s0 = fma(b.s1, -e, t.s0);
	t.s0 = fma(r, t.s0, e);
	c.s1 = e = t.s1 + t.s0;
	c.s0 = (t.s1 - e) + t.s0;

	return c;
}

__kernel void mmult32ds(__global type2* data0, __global type2* data1, __global type2* output_data,
	__local type2* matrix0, __local type2* matrix1) {

	uint global_addr0, global_addr1, local_addr0, local_addr1, i;
	type2 out = 0;

	global_addr0 = get_global_id(0);
	global_addr1 = get_global_id(1);
	local_addr0 = get_local_id(0);
	local_addr1 = get_local_id(1);

	matrix0[local_addr0 + local_addr1*ARRAY_SIZE] = data0[local_addr0 + local_addr1*ARRAY_SIZE];
	matrix1[local_addr0 + local_addr1*ARRAY_SIZE] = data1[local_addr0 + local_addr1*ARRAY_SIZE];

	barrier(CLK_LOCAL_MEM_FENCE);

	for (i = 0; i<ARRAY_SIZE; i++){
		out = add_dd(out, mul_dd(matrix0[i + local_addr1*ARRAY_SIZE], matrix1[local_addr0 + i*ARRAY_SIZE]));
	}
	output_data[local_addr0 + local_addr1*ARRAY_SIZE] = out;
}