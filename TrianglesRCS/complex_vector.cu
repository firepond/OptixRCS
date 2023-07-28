#pragma once

#include <cuda/helpers.h>
#include <cuda_runtime.h>
#include <vector_types.h>

#include <cuda/std/complex>

using cuda::std::complex;


struct complexFloat3 {
    complex<float> x, y, z;
};

__device__ complexFloat3 makeComplexFloat3(complex<float> a, complex<float> b,
                                           complex<float> c) {
    complexFloat3 res;
    res.x = a;
    res.y = b;
    res.z = c;
    return res;
}

__device__ __forceinline void printComplexFloat3(complexFloat3 cf3) {
    printf("((%7f,%7f),(%7f,%7f),(%7f,%7f))\n", cf3.x.real(), cf3.x.imag(),
           cf3.y.real(), cf3.y.imag(), cf3.z.real(), cf3.z.imag());
}


__device__ complexFloat3 operator-(complexFloat3 cf3) {
    return makeComplexFloat3(-cf3.x, -cf3.y, -cf3.z);
}


__device__ complexFloat3 operator*(complex<float> cf, float3 f3) {
    complex<float> a = cf * f3.x;
    complex<float> b = cf * f3.y;
    complex<float> c = cf * f3.z;
    return makeComplexFloat3(a, b, c);
}

__device__ complexFloat3 operator-(complexFloat3 a, complexFloat3 b) {
    complex<float> cfa = a.x - b.x;
    complex<float> cfb = a.y - b.y;
    complex<float> cfc = a.z - b.z;
    return makeComplexFloat3(cfa, cfb, cfc);
}

__device__ complexFloat3 operator*(complexFloat3 cf3, float f) {
    complex<float> a = cf3.x * f;
    complex<float> b = cf3.y * f;
    complex<float> c = cf3.z * f;
    return makeComplexFloat3(a, b, c);
}

__device__ complexFloat3 operator+(complexFloat3 cf3a, complexFloat3 cf3b) {
    complex<float> a = cf3a.x + cf3b.x;
    complex<float> b = cf3a.y + cf3b.y;
    complex<float> c = cf3a.z + cf3b.z;
    return makeComplexFloat3(a, b, c);
}

__device__ complexFloat3 cross(complexFloat3 a, float3 b) {

    complex<float> cfa = a.y * b.z - a.z * b.y;
    complex<float> cfb = a.z * b.x - a.x * b.z;
    complex<float> cfc = a.x * b.y - a.y * b.x;
    return makeComplexFloat3(cfa, cfb, cfc);
}


__device__ complex<float> dot(complexFloat3 cf3, float3 f3) {
    return cf3.x * f3.x + cf3.y * f3.y + cf3.z * f3.z;
}

// CtsToSph
// Physics convention
// X Y Z --> R Phi(x-y angle, 0 to 2pi) Theta(z-xy angle, 0 to pi)
__device__ float3 CtsToSph(float3 f3) {
    float x = f3.x;
    float y = f3.y;
    float z = f3.z;
    float r = sqrt(x * x + y * y + z * z);
    float theta = acosf(z / r);
    float phi = atan2f(y, x);

    return make_float3(r, phi, theta);
}