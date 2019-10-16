// Adapted from: https://stackoverflow.com/questions/1389712/getting-started-with-intel-x86-sse-simd-instructions
#include <chrono>
#include <stdio.h>
#include <immintrin.h>

float* addFloats(float* a, float *b) {
  
  float* sum = new float[4];

  for (int i = 0; i < 4; i++) {
    sum[i] = a[i] + b[i];
  }

  return sum;
}

__m128 addFloatsVectorized(float* a, float *b) {

  // The high element appears first (big endian), opposite of C array order.
  /*
  __m128 vector1 = _mm_set_ps(a[3], a[2], a[1], a[0]);
  __m128 vector2 = _mm_set_ps(b[3], b[2], b[1], b[0]);
  */

  // Use _mm_setr_ps to use little endian element order.
  __m128 vector1 = _mm_setr_ps(a[0], a[1], a[2], a[3]);
  __m128 vector2 = _mm_setr_ps(b[0], b[1], b[2], b[3]);

  __m128 sum = _mm_add_ps(vector1, vector2); // result = vector1 + vector 2

  return sum;
}

int main()
{
    //   [ 4.50,   3.00,  2.00,  1.00]
    // + [ 9.00,   7.00,  6.25,  5.00]
    // = [13.50,  10.00,  8.25,  6.00]
    float a[4] = {4.5, 3.0, 2.0, 1.0};
    float b[4] = {9.0, 7.0, 6.25, 5.0};

    // Normal sum
    float *sum = addFloats(a, b);
    printf("Sum: %.2f, %.2f, %.2f, %.2f\n", sum[0], sum[1], 
      sum[2], sum[3]);

    // Vector sum
    __m128 sumVector = addFloatsVectorized(a, b);

    // Convert to float[] to display components
    float *resultVector = (float*) &sumVector;

    printf("Sum (vector): %.2f, %.2f, %.2f, %.2f\n", resultVector[0], resultVector[1], 
      resultVector[2], resultVector[3]);

    printf("\n");

    // 100,000 iterations of non-vectorized
    auto start1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100000; i++) {
      float *sum = addFloats(a, b);
    }
    auto stop1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(stop1 - start1);

    printf("Non-vectorized time (microsec): %.2lld\n", duration1.count());


    // 100,000 iterations of vectorized code
    auto start2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100000; i++) {
      __m128 sumVector = addFloatsVectorized(a, b);
    }
    auto stop2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(stop2 - start2);

    printf("Vectorized time (microsec): %.2lld\n", duration2.count());

    printf("\n");
    
    return 0;
}