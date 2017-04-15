/* convolves an image with many filters
 * for simplicity all filters have the same size and are square
 * image and result are assumed to be grayscale with a depth of 8 bits
 * takes advantage of local work greeps to further accelerate
 */
__kernel
void convolve2d(__constant unsigned char *image,
  __global float *filter,
  __global unsigned char *result,
  unsigned int image_width,
  unsigned int image_height,
  unsigned int filter_width,
  unsigned int filter_height,
  unsigned int num_filters)
{
  int pixel = get_global_id(0); /* current pixel */
  int fid = get_global_id(1); /* index of filter */

  if(pixel < (image_width*image_height) && fid < num_filters){

    const int px = pixel % image_width;
    const int py = ((pixel - px)/image_width);
    const unsigned int image_size = image_width * image_height;
    const unsigned long filter_len = filter_width * filter_height;
    const int offset = (filter_len - 1)/2;
    /* top left corner of filter window on image */
    const int cornerx = px - offset;
    const int cornery = py - offset;

    float sum = 0;

    /* iterate over the filter */
    for(unsigned int i = 0; i < filter_len; ++i){
      int col = (cornerx) + (i % filter_width);
      int row = (cornery) + ((i -(i%filter_width))/filter_width);

      /* zero the pixels if they are out of bounds */
      float source;
      if(row < 0 || row > image_height || col < 0 || col > image_width){
        source = 0;
      }else{
        source = image[row*image_width + col];
      }

      /* convolution uses the filter backwards */
      const unsigned int findex = filter_len - i - 1 + fid*filter_len;
      const float weight = filter[findex];
      sum += source*weight;
    }
    result[py*image_width + px + fid*image_size] = sum;
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
