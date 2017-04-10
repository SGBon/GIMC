#include "clutil.h"
#include <stdio.h>
#include <stdlib.h>

void print_error(const char *function, cl_int error){
  fprintf(stderr,"Error when calling %s, code: %d\n",function,error);
}

/* determine length of file, allocate memory for the output pointer
 * and read in the bytes into pointer */
void read_cl_source(const char *filename, char **source_out){
  FILE * const fp = fopen(filename,"rb");
  if(!fp){
    fprintf(stderr,"Failed to load kernel\n");
    exit(EXIT_FAILURE);
  }

  fseek(fp,0,SEEK_END);
  const size_t len = ftell(fp);
  rewind(fp);

  char *source;
  source = malloc(len+1);
  source[len] = '\0'; /* null terminate the source */
  fread(source,sizeof(char),len,fp);
  fclose(fp);
  *source_out = source;
}

void free_cl_source(char *source){
  free(source);
}
