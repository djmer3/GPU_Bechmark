#pragma OPENCL EXTENSION cl_khr_select_fprounding_mode : enable
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#define __ROUNDING_MODE__ rte
#define ARRAY_SIZE 16

float two_sum(float a, float b, float *err){
	if ((a == 0.0) || (b == 0.0)){
		*err = 0.0;
		return (a + b);
	}
	float s = a + b;
	float bb = s - a;
	*err = (a - (s - bb)) + (b - bb);

	return s;
}
float quick_two_sum(float a, float b, float *err){

	float s = a + b;

	if (b == 0.0){
		*err = 0.0;
		return s;
	}

	*err = b - (s - a);
	return s;
}

void three_sum(float *a, float *b, float *c){
	float t1, t2, t3;
	t1 = two_sum(*a, *b, &t2);
	*a = two_sum(*c, t1, &t3);
	*b = two_sum(t2, t3, c);
}

void three_sum2(float *a, float *b, float c){
	float t1, t2, t3;
	t1 = two_sum(*a, *b, &t2);
	*a = two_sum(c, t1, &t3);
	*b = t2 + t3;
}

/*float4 renorm_qd(float4 a){
float t0, t1, t2, t3;
float4 out;

t0 = quick_two_sum(a.s2, a.s3, &t3);
t0 = quick_two_sum(a.s1, t0, &t2);
t0 = quick_two_sum(a.s0, t0, &t1);

if (t1 != 0.0) {
t1 = quick_two_sum(t1, t2, &t2);
if (t2 != 0.0){
t2 = quick_two_sum(t2, t3, &t3);
}
else {
t1 = quick_two_sum(t1, t3, &t2);
t3 = 0.0;
}
}
else {
t0 = quick_two_sum(t0, t2, &t1);
if (t1 != 0.0){
t1 = quick_two_sum(t1, t3, &t2);
t3 = 0.0;
}
else {
t0 = quick_two_sum(t0, t3, &t1);
t3 = 0.0;
t2 = 0.0;
}
}

out.s0 = t0;
out.s1 = t1;
out.s2 = t2;
out.s3 = t3;

return out;
}

float4 renorm_qd5(float a0, float a1, float a2, float a3, float a4){
float t0, t1, t2, t3, t4;
float4 out;

t0 = quick_two_sum(a3, a4, &t4);
t0 = quick_two_sum(a2, t0, &t3);
t0 = quick_two_sum(a1, t0, &t2);
t0 = quick_two_sum(a0, t0, &t1);

if (t1 != 0.0) {
t1 = quick_two_sum(t1, t2, &t2);
if (t2 != 0.0){
t2 = quick_two_sum(t2, t3, &t3);
if (t3 != 0.0){
t3 = t3 + t4;
}
else {
t2 = t2 + t4;
}
}
else {
t1 = quick_two_sum(t1, t3, &t2);
if (t2 != 0.0){
t2 = quick_two_sum(t2, t4, &t3);
}
else {
t1 = quick_two_sum(t1, t4, &t2);
t3 = 0.0;
}
}
}
else {
t0 = quick_two_sum(t0, t2, &t1);
if (t1 != 0.0){
t1 = quick_two_sum(t1, t3, &t2);
if (t2 != 0.0){
t2 = quick_two_sum(t2, t4, &t3);
}
else {
t1 = quick_two_sum(t1, t4, &t2);
t3 = 0.0;
}
}
else {
t0 = quick_two_sum(t0, t3, &t1);
if (t1 != 0.0){
t1 = quick_two_sum(t1, t4, &t2);
t3 = 0.0;
}
else {
t0 = quick_two_sum(t0, t4, &t1);
t2 = 0.0;
t3 = 0.0;
}
}
}

out.s0 = t0;
out.s1 = t1;
out.s2 = t2;
out.s3 = t3;

return out;
}*/

float4 renorm_qd5(float a0, float a1, float a2, float a3, float a4){
	float s, e, c, f, g;
	float t[5], b[4] = { 0.0 };
	float4 out;

	s = a4;
	s = quick_two_sum(a3, s, &t[4]);
	s = quick_two_sum(a2, s, &t[3]);
	s = quick_two_sum(a1, s, &t[2]);
	s = quick_two_sum(a0, s, &t[1]);

	t[0] = s;
	b[0] = t[0];
	int k = 0;
	for (int i = 1; i < 5; i++){
		s = quick_two_sum(s, t[i], &e);
		b[k] = s;
		if (e != 0.0){
			int l = k - 1;
			k++;
			while (l >= 0){
				f = quick_two_sum(b[l], b[l + l], &g);
				if (g == 0){
					b[l] = f;
					l = l - 1;
					k = k - 1;
				}
				else{
					l = 0;
				}
			}
		}
	}
	out.x = b[0];
	out.y = b[1];
	out.z = b[2];
	out.w = b[3];
	return out;
}

float4 renorm_qd4(float4 a){
	float t = 0.0;
	return renorm_qd5(a.s0, a.s1, a.s2, a.s3, t);
}

float4 add_qd(float4 a, float4 b){
	float t0, t1, t2, t3, t4;
	float c0, c1, c2, c4;

	t0 = two_sum(a.s0, b.s0, &c0);

	t1 = two_sum(a.s1, b.s1, &c1);
	t1 = two_sum(t1, c0, &c0);

	t2 = two_sum(a.s2, b.s2, &c2);
	three_sum(&t2, &c1, &c0);

	t3 = two_sum(a.s3, b.s3, &c4);
	three_sum2(&t3, &c1, c2);

	t4 = c4 + c1 + c0;

	return renorm_qd5(t0, t1, t2, t3, t4);
}

float4 mul_qd(float4 a, float4 b){
	float p10, p02, p11, p20;
	float q00, q01, q10, q02, q11, q20;
	float t0, t1;
	float4 out;

	//eps^0 terms
	out.s0 = a.s0 * b.s0;
	q00 = fma(a.s0, b.s0, -out.s0);

	//eps^1 terms
	t0 = out.s1 = a.s0*b.s1;
	q01 = fma(a.s0, b.s1, -out.s1);
	p10 = a.s1 *b.s0;
	q10 = fma(a.s1, b.s0, -p10);

	three_sum(&t0, &p10, &q00);

	//eps ^2 terms
	p02 = a.s0 *b.s2;
	q02 = fma(a.s0, b.s2, -p02);
	p11 = a.s1*b.s1;
	q11 = fma(a.s1, b.s1, -p11);
	p20 = a.s2 *b.s0;
	q20 = fma(a.s2, b.s0, -p20);

	//six-three_sum(p10,q01,q10,p02,p11,p20)
	three_sum(&p10, &q01, &q10);
	three_sum(&p02, &p11, &p20);
	//(s0,s1,s2) = (p2,q1,q2) + (p3,p4,p5)
	out.s2 = two_sum(p10, p02, &t0);
	out.s3 = two_sum(q01, p11, &t1);
	out.s3 = two_sum(out.s3, t0, &t0);

	//eps ^3 terms
	out.s3 = out.s3 + (a.s0*b.s3 + a.s1*b.s2 + a.s2*b.s1 + a.s3*b.s0 + q00 + q02 + q11 + q20);

	//eps ^4 terms are skipped for speed

	return renorm_qd4(out);
}

__kernel void mmult32qs(__global float4* data0, __global float4* data1, __global float4* output_data, __local float4* matrix0, __local float4* matrix1){

	uint global_addr0, global_addr1, local_addr0, local_addr1;
	float4 out = { 0.0, 0.0, 0.0, 0.0 };

	global_addr0 = get_global_id(0);
	global_addr1 = get_global_id(1);
	local_addr0 = get_local_id(0);
	local_addr1 = get_local_id(1);

	matrix0[local_addr0 + local_addr1 * ARRAY_SIZE] = renorm_qd4(data0[local_addr0 + local_addr1 * ARRAY_SIZE]);
	matrix1[local_addr0 + local_addr1 * ARRAY_SIZE] = renorm_qd4(data1[local_addr0 + local_addr1 * ARRAY_SIZE]);

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 0; i < ARRAY_SIZE; i++){
		out = add_qd(out, mul_qd(matrix0[i + local_addr1*ARRAY_SIZE], matrix1[local_addr0 + i*ARRAY_SIZE]));
	}
	output_data[local_addr0 + local_addr1 * ARRAY_SIZE] = out;
}
