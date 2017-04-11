#define _USE_MATH_DEFINES //compatibility
#include <math.h>
#include "filter.h"

/* guassian function */
static float gaussian(float x, float y, float sigma);

void filter_gauss2d(float *filter, unsigned int n, float sigma){
  const float coefficient = (1.0/(2*M_PI*sigma*sigma));
  const int offset = (n-1) / 2;
  float sum = 0.0f;
  /* construct gaussian */
  for(int i = 0; i < n; ++i){
    for(int j = 0; j < n; ++j){
      const float next  = coefficient * gaussian(i - offset,j-offset,sigma);
      filter[i*n+j] = next;
      sum += next;
    }
  }

  /* normalize gaussian */
  for(int i = 0; i < n; ++i){
    for(int j = 0; j < n; ++j){
      filter[i*n+j] /= sum;
    }
  }
}
float gaussian(float x, float y, float sigma){
  return exp(-(x*x + y*y)/(2*sigma*sigma));
}
