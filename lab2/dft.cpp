#include <math.h>
#include <string>
#include <stdio.h>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <math.h>
#include <complex>
#include <algorithm>
#include <omp.h>

#include "utils/bmp.cpp"

void compress(
  const uint32_t valuesCount, 
  const int accuracy,
  const uint8_t *values, 
  float *Xreal, 
  float *Ximag
) {
  // values, Xreal and Ximag are values describing single color of single row of bitmap. 
  // This function will be called once per each (color, row) combination.
  for (int k = 0; k < accuracy; k++) {
    for (int i = 0; i < valuesCount; i++) {
      float theta = (2 * M_PI * k * i) / valuesCount;
      Xreal[k] += values[i] * cos(theta);
      Ximag[k] -= values[i] * sin(theta);
    }
  }
}

void decompress(
  const uint32_t valuesCount, 
  const int accuracy,
  uint8_t *values, 
  const float *Xreal, 
  const float *Ximag
) {
  // values, Xreal and Ximag are values describing single color of single row of bitmap.
  // This function will be called once per each (color, row) combination.
  std::vector<float> rawValues(valuesCount, 0);

  for (int i = 0; i < valuesCount; i++) {
    for (int k = 0; k < accuracy; k++) {
      float theta = (2 * M_PI * k * i) / valuesCount;
      rawValues[i] += Xreal[k] * cos(theta) + Ximag[k] * sin(theta);
    }
    values[i] = rawValues[i] / valuesCount;
  }
}

void compressPar(
  const uint32_t valuesCount, 
  const int accuracy,
  const uint8_t *values, 
  float *Xreal, 
  float *Ximag
) {
  // PUT YOUR IMPLEMENTATION HERE
  #pragma omp parallel for 
  for (int k = 0; k < accuracy; k++) {
    #pragma omp parallel for reduction(+:Xreal[:accuracy]) reduction(-:Ximag[:accuracy])
    for (int i = 0; i < valuesCount; i++) {
        float theta = (2 * M_PI * k * i) / valuesCount;
        Xreal[k] += values[i] * cos(theta);
        Ximag[k] -= values[i] * sin(theta);
    }
  }
}

void decompressPar(
  const uint32_t valuesCount, 
  const int accuracy,
  uint8_t *values, 
  const float *Xreal, 
  const float *Ximag
) {
  // PUT YOUR IMPLEMENTATION HERE
  // We treat the trig functions to be an "expensive computation".
  float cosines[accuracy][valuesCount];
  float sines[accuracy][valuesCount];

  #pragma omp for collapse(2)
  for (int k = 0; k < accuracy; k++) {
    for (int i = 0; i < valuesCount; i++) {
      float theta = (2 * M_PI * k * i) / valuesCount;
      cosines[k][i] = cos(theta);
      sines[k][i] = sin(theta);
    }
  }

  for (int i = 0; i < valuesCount; i++) {
    for (int k = 0; k < accuracy; k++) {
      values[i] += (Xreal[k] * cosines[k][i] + Ximag[k] * sines[k][i]);
    }
    values[i] /= valuesCount;
  }
}

void test(std::string filename, const CompressFun* comp, const DecompressFun* decomp) {
  // bmp.{compress,decompress} will run provided function on every bitmap row and color.
  static BMP bmp;
  bmp.read("example.bmp");
  static size_t accuracy = 16; // We are interested in values from range [8; 32]

  float compressTime = bmp.compress(comp, accuracy);
  float decompressTime = bmp.decompress(decomp);
  printf("Compress time: %.2lfs\nDecompress time: %.2lfs\nTotal: %.2lfs\n\n", 
    compressTime, decompressTime, compressTime + decompressTime);
  bmp.write(filename + "_result.bmp");
}

int main() {    
  test("seq", compress, decompress);
  test("par", compressPar, decompressPar);

  return 0;
}

