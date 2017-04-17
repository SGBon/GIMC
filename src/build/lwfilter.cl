/* 1st kernel: convolves an image with many filters
 * for simplicity all filters have the same size and are square
 * image and result are assumed to be grayscale with a depth of 8 bits
 * takes advantage of local work groups to further accelerate computation.
 * work groups produce partial sums which are then reduced
 * image: buffer containing image to perform convolution on
 * filter: buffer containing bank of filters
 * result: buffer where resulting images are created
 * scratch: local workspace
 * image_width/height, filter_width: sizes of image and filters
 * num_filters: number of filters in bank.
 */
__kernel
void convolve2d(__global unsigned char *image,
  __global float *filter,
  __global unsigned char *result,
  __local float *scratch,
  unsigned long image_width,
  unsigned long image_height,
  unsigned int filter_width,
  unsigned int num_filters)
{
  const unsigned int pixel = get_global_id(0); /* current pixel */
  const unsigned int fid = get_global_id(1); /* index of filter in bank */
  const unsigned int fchunk = get_global_id(2); /* chunk of filter to compute */

  const unsigned int lid = get_local_id(2);
  const unsigned int local_size = get_local_size(2);
  const unsigned int image_size = image_width * image_height;

  scratch[lid] = 0.0;

  if(pixel < (image_width*image_height) && fid < num_filters){

    const int px = pixel % image_width;
    const int py = ((pixel - px)/image_width);
    const unsigned int filter_len = filter_width * filter_width;
    const unsigned int chunk_size = filter_len < local_size ? filter_len : filter_len/local_size;
    const int offset = (filter_len - 1)/2;
    /* top left corner of filter window on image */
    const int cornerx = px - offset;
    const int cornery = py - offset;

    const unsigned int start = chunk_size*lid;
    unsigned int end = start + chunk_size;
    /* give last chunk rest of remaining work */
    if(filter_len % local_size != 0 && fchunk == local_size - 1){
      end = filter_len;
    }
    float sum = 0.0f;

    /* iterate over the filter */
    for(unsigned int i = start; i < end && i < filter_len; ++i){
      int col = (cornerx) + (i % filter_width);
      int row = (cornery) + ((i -(i%filter_width))/filter_width);

      /* zero the pixels if they are out of bounds */
      float source;
      if(row < 0 || row > image_height || col < 0 || col > image_width){
        source = 0.0f;
        }else{
          source = image[row*image_width + col];
        }

        /* convolution uses the filter backwards */
        const unsigned int findex = filter_len - i - 1 + fid*filter_len;
        const float weight = filter[findex];
        sum += source*weight;
        //printf("%f %f %f\n",sum,source, weight);
    }
    scratch[lid] = sum;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  /* perform commutative reduction */
  for(unsigned int offset = local_size / 2; offset > 0; offset >>=1){
    if(lid < offset){
      const float other = scratch[lid + offset];
      scratch[lid] += other;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  /* send final result up */
  if(lid == 0){
    result[pixel + fid*image_size] = scratch[0];
  }
}


/* create a gaussian filter bank in the bank buffer
 * bank: buffer to create bank in
 * num_filters: number of filters to create
 * filter_width: size of filters
 * 2 dimensional, first dimension is the filter
 * second dimension is the cell of that filter
 */
__kernel
void filter_Gauss2dbank(__global float *bank,
  unsigned int num_filters,
  unsigned int filter_width)
{
  const unsigned int filter = get_global_id(0);
  const unsigned int cell = get_global_id(1);

  const unsigned long filter_len = filter_width*filter_width;
  if(filter < num_filters && cell < filter_len){
    const float sigma = (filter + 1) * 25.0/num_filters;
    const int offset = (filter_width - 1)/2;
    int y = (cell % filter_width);
    int x = ((cell - y)/filter_width);
    y -= offset;
    x -= offset;


    const float value = (1.0/(2*M_PI_F*sigma*sigma)) * (exp(-(x*x + y*y)/(2.0*sigma*sigma)));

    bank[filter_len*filter + cell] = value;
  }
}

/* normalize a filter so that all values sum up to 1
 * bank: buffer which filters reside in
 * num_filters: maximum number of filters in bank
 * filter_width: side lengths of filters
 */
__kernel
void filter_normalize(__global float *bank,
  unsigned int num_filters,
  unsigned int filter_width)
{
  const unsigned int filter = get_global_id(0);
  const unsigned int filter_len = filter_width*filter_width;

  if(filter < num_filters){
    float sum = 0;
    for(unsigned int i = 0; i < filter_len; ++i){
      sum += bank[filter_len*filter+i];
    }

    for(unsigned int i = 0; i < filter_len; ++i){
      bank[filter_len*filter+i] = bank[filter_len*filter+i] / sum;
    }
  }
}
