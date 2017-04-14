#define _USE_MATH_DEFINES //compatibility
#include <math.h>
#include "filter.h"

/* guassian function */
static double Gaussian(double x, double y, double sigma);

/* create a bank of 2d Gaussian filters
 * use sigma to vary the gaussians
 */
void filter_Gauss2dbank(double *bank,unsigned int num_filters, unsigned int filter_width){
  double step = num_filters > 3 ? 10.0/num_filters : 1.0;
  for(unsigned int i = 0; i < num_filters; ++i){
    filter_Gauss2d(&bank[i*filter_width],filter_width,i*step);
  }
}

void filter_Gauss2d(double *filter, unsigned int n, double sigma){
  const double coefficient = (1.0/(2*M_PI*sigma*sigma));
  const int offset = (n-1) / 2;
  double sum = 0.0;
  /* construct Gaussian */
  for(int i = 0; i < n; ++i){
    for(int j = 0; j < n; ++j){
      const double next  = coefficient * Gaussian(i - offset,j-offset,sigma);
      filter[i*n+j] = next;
      sum += next;
    }
  }

  /* normalize Gaussian */
  for(int i = 0; i < n; ++i){
    for(int j = 0; j < n; ++j){
      filter[i*n+j] /= sum;
    }
  }
}

double Gaussian(double x, double y, double sigma){
  return exp(-(x*x + y*y)/(2*sigma*sigma));
}
