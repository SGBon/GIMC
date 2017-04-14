/* functions related to generating filters */

#ifndef GIMC_FILTER_H
#define GIMC_FILTER_H

/* create a bank of 2d Gaussian filters
 * bank: array to put Gaussians into
 * num_filters: number of filters to create
 * filter_width: width of each filter
 * filter_width is assumed to be odd
 */
extern void filter_Gauss2dbank(float *bank,unsigned int num_filters, unsigned int filter_width);

/* create a 2d Gaussian
 * filter: array to put Guassian into
 * n: side lengths of kernel
 * sigma: standard deviation
 * n is assumed to be odd
 */
extern void filter_Gauss2d(float *filter, unsigned int n, float sigma);


#endif
