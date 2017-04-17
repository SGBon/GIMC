/* 1st kernel: convolves an image with many filters
 * for simplicity all filters have the same size and are square
 * image and result are assumed to be grayscale with a depth of 8 bits
 * takes advantage of local work groups to further accelerate computation.
 * work groups produce partial sums which are resolved in the 2nd kernel
 * image: buffer containing image to perform convolution on
 * filter: buffer containing bank of filters
 * scratch: local workspace
 * image_width/height, filter_width/height: sizes of image and filters
 * num_filters: number of filters in bank.
 */
__kernel
void convolve2d(__global unsigned char *image,
  __global float *filter,
  __local float *scratch,
  __global float *psum,
  unsigned long image_width,
  unsigned long image_height,
  unsigned int filter_width,
  unsigned int filter_height,
  unsigned int num_filters)
{
  const unsigned int pixel = get_global_id(0); /* current pixel */
  const unsigned int fid = get_global_id(1); /* index of filter in bank */
  const unsigned int fcell = get_global_id(2); /* index of cell in filter */

  const unsigned int lp = get_local_id(0);
  const unsigned int lf = get_local_id(1);
  const unsigned int lc = get_local_id(2);

  const unsigned int filter_len = filter_width * filter_height;
  const unsigned int image_size = image_width * image_height;

  scratch[lc] = 0.0;
  barrier(CLK_LOCAL_MEM_FENCE);

  if(pixel < image_size && fid < num_filters && fcell < filter_len){
    const int px = pixel % image_width;
    const int py = ((pixel - px)/image_width);
    const int offset = (filter_len - 1)/2;
    /* top left corner of filter window on image */
    const int cornerx = px - offset;
    const int cornery = py - offset;

    const int col = (cornerx) + (fcell % filter_width);
    const int row = (cornery) + ((fcell - (fcell %filter_width))/filter_width);

    /* zero the pixels if they are out of bounds */
    float source;
    if(row < 0 || row > image_height || col < 0 || col > image_width){
      source = 0;
    }else{
      source = image[row*image_width+col];
    }
    const unsigned int findex = filter_len - fcell - 1 + fid*filter_len;
    const float weight = filter[findex];
    scratch[lc] = source * weight;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  /* perform commutative reduction within the workgroup */
  for(unsigned int offset = get_local_size(2) / 2; offset > 0; offset >>= 1){
    if(lc < offset){
      const float other = scratch[lc + offset];
      scratch[lc] += other;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  /* first thread in workgroup sends partial sum up */
  if(lc == 0){
    psum[pixel - get_global_offset(0) + get_group_id(2)] = scratch[0];
  }
}

/* second kernel of 2d convolution: reduce partial sums to full sums
 * and put the final result into the result buffer
 * psum: buffer of partial sums from first kernel
 * result: buffer to put result of convolution into
 * image_width, image_height: dimensions of image
 * psum_per_pixel: amount of partial sums which correspond to each pixel in result
 */
__kernel
void convolve2d_reduce(__global float *psum,
  __global unsigned char *result,
  unsigned long image_width,
  unsigned long image_height,
  unsigned long psum_per_pixel)
{
  const unsigned int pixel = get_global_id(0);
  const unsigned int fid = get_global_id(1);
  const unsigned int offset = pixel - get_global_offset(0);
  const unsigned long image_size = image_width * image_height;

  if(pixel < image_size){
    float sum = 0.0;
    for(unsigned int i = 0; i < psum_per_pixel; ++i){
      sum += psum[offset*psum_per_pixel + i];
    }
    //printf("%u %u\n",pixel,fid*image_size);
    result[pixel + fid*image_size] = sum;
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
  const int cell = get_global_id(1);

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
