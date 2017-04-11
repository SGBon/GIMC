#define _USE_MATH_DEFINES //compatibility
#include <math.h>
#include "filter.h"

/* guassian function */
static double gaussian(double x, double y, double sigma);

void filter_gauss2d(double *filter, unsigned int n, double sigma){
  const double coefficient = (1.0/(2*M_PI*sigma*sigma));
  const int offset = (n-1) / 2;
  double sum = 0.0;
  /* construct gaussian */
  for(int i = 0; i < n; ++i){
    for(int j = 0; j < n; ++j){
      const double next  = coefficient * gaussian(i - offset,j-offset,sigma);
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

double gaussian(double x, double y, double sigma){
  return exp(-(x*x + y*y)/(2*sigma*sigma));
}
