/* functions related to generating filters */

#ifndef GIMC_FILTER_H
#define GIMC_FILTER_H

/* create a 2d gaussian
 * filter: array to put guassian into
 * n: side lenghts of kernel
 * sigma: standard deviation
 * mu: offset for center of gaussian ie. mean
 * n is assumed to be odd
 */
extern void filter_gauss2d(float *filter, unsigned int n, float sigma);

#endif
