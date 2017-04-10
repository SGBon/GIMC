#ifndef GIMC_CLUTIL_H
#define GIMC_CLUTIL_H

#include <CL/cl.h>

/* Prints an error with error code to stderr */
void print_error(const char *function, cl_int error);

/* read a source file for kernel code from filename into source
 * source is malloc so it's necessary to free the memory using free_cl_source
 */
void read_cl_source(const char *filename,char **source);

/* frees opencl kernel source read with read_cl_source */
void free_cl_source(char *source);

#endif
