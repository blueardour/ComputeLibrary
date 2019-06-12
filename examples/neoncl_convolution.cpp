/*
 * Copyright (c) 2016-2018 ARM Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#ifndef ARM_COMPUTE_CL /* Needed by Utils.cpp to handle OpenCL exceptions properly */
#error "This example needs to be built with -DARM_COMPUTE_CL"
#endif /* ARM_COMPUTE_CL */

#include "arm_compute/runtime/CL/CLFunctions.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/CL/CLTuner.h"

#include "arm_compute/runtime/NEON/NEFunctions.h"

#include "arm_compute/core/Types.h"
#include "utils/ImageLoader.h"
#include "utils/Utils.h"

using namespace arm_compute;
using namespace utils;

static const bool debug = false;

class LayerParameter
{
  public:
    int width, height, cin, cout, pad, stride, dilation, kernel;

    int seed;
    bool buffer_ready;
    float *input, *weight, *cpu_buffer, *gpu_buffer;

    ~LayerParameter() {
      free(input);
      free(weight);
      free(cpu_buffer);
      free(gpu_buffer);
      input = weight = cpu_buffer = gpu_buffer = NULL;
    }

    LayerParameter(int sd = 1) {
      width = height = 56;
      cin = cout = 64;
      kernel = 3;
      pad = 1;
      stride = dilation = 1;

      input = weight = cpu_buffer = gpu_buffer = NULL;
      buffer_ready = false;
      seed = sd;
    }

    void print()
    {
      std::cout << "layer info: " << std::endl;
      std::cout << "width and height: " << width << ", " << height << std::endl;
      std::cout << "cin and cout: " << cin << ", " << cout << std::endl;
      std::cout << "pad, stride, dilation: " << pad << ", " << stride << ", " << dilation << std::endl;
      std::cout << "kernel: " << kernel << std::endl;
    }

    int outw() { return (width  + 2*pad - kernel) / stride + 1; }
    int outh() { return (height + 2*pad - kernel) / stride + 1; }

    void init(int sd = -1)
    {
      if(sd <= 0) sd = seed;
      srandom(sd);
      int len, i, bs;
      bs = sizeof(float);
      float *tmp;

      len = cin * width * height;
      tmp = input = (float *)malloc(bs * len);
      for(i=0; i<len; i++) tmp[i] = tanh(((random() % 1000) -500) * 0.001f);

      len = cin * cout * kernel * kernel;
      tmp = weight = (float *)malloc(bs * len);
      for(i=0; i<len; i++) tmp[i] = tanh(((random() % 1000) -500) * 0.001f);

      len = cout * outw() * outh();
      cpu_buffer = (float *)malloc(bs * len);
      gpu_buffer = (float *)malloc(bs * len);
      buffer_ready = true;
    }

    void save(float *buffer, int len, char *workspace) {
      std::ofstream txt;
      txt.open(workspace);
      for(int i=0; i<len; i++) {
        txt << buffer[i] << "\n";
      }
      txt.close();
    }

    void save(char *workspace) {
      char * ptr = strrchr(workspace, '/') + 1;
      int len;
      strcpy(ptr, "input.txt");
      len = cin * width * height;
      save((float *)input, len, workspace);
      strcpy(ptr, "weight.txt");
      len = cin * cout * kernel * kernel;
      save((float *)weight, len, workspace);
      strcpy(ptr, "cpu.txt");
      len = cout * outw() * outh();
      save((float *)cpu_buffer, len, workspace);
      strcpy(ptr, "gpu.txt");
      len = cout * outw() * outh();
      save((float *)gpu_buffer, len, workspace);
    }
};

int test_cpu(LayerParameter *lp, DataType dtype=DataType::F32)
{
  std::cout << "Enter " << __FUNCTION__ << std::endl;
  // init data tensor
  Tensor* data  = new Tensor();
  const TensorShape input_shape(lp->width, lp->height, lp->cin);
  data->allocator()->init(TensorInfo(input_shape, 1, dtype));

  // init weight tensor
  Tensor* weight = new Tensor();
  const TensorShape weight_shape(lp->kernel, lp->kernel, lp->cin, lp->cout);
  weight->allocator()->init(TensorInfo(weight_shape, 1, dtype));

  // init output tensor
  Tensor* output = new Tensor();
  const TensorShape output_shape(lp->outw(), lp->outh(), lp->cout);
  output->allocator()->init(TensorInfo(output_shape, 1, dtype));

  // init conv layer
  PadStrideInfo conv_info(lp->stride, lp->stride, lp->pad, lp->pad);

  NEConvolutionLayer *conv = new NEConvolutionLayer();
  conv->configure(data, weight, nullptr, output, conv_info);

  Status status = conv->validate(data->info(), weight->info(), nullptr, output->info(), conv_info);
  if((bool)status != true)
    std::cout << "Return Error: " << status.error_description() << std::endl;

  ConvolutionMethod method = conv->get_convolution_method(data->info(), weight->info(), output->info(), conv_info);
  if(method == arm_compute::ConvolutionMethod::GEMM)
    std::cout << "GEMM" << std::endl;
  else if(method == arm_compute::ConvolutionMethod::WINOGRAD)
    std::cout << "WINOGRAD" << std::endl;
  else if(method == arm_compute::ConvolutionMethod::DIRECT)
    std::cout << "DIRECT" << std::endl;
  else
    std::cout << "Unkown" << std::endl;

  // allocate and assgin value
  data->allocator()->allocate();
  weight->allocator()->allocate();
  output->allocator()->allocate();
  if(lp->buffer_ready)
  {
    Window window;
    window.use_tensor_dimensions(data->info()->tensor_shape());
    //std::cout << " Dimensions of the input's iterator:\n";
    //std::cout << " X = [start=" << window.x().start() << ", end=" << window.x().end() << ", step=" << window.x().step() << "]\n";
    //std::cout << " Y = [start=" << window.y().start() << ", end=" << window.y().end() << ", step=" << window.y().step() << "]\n";
    //std::cout << " Z = [start=" << window.z().start() << ", end=" << window.z().end() << ", step=" << window.z().step() << "]\n";

    Iterator input_iter(data, window);
    execute_window_loop(window,
        // lambda function
        [&](const Coordinates & id) {
            //std::cout << "Setting item [" << id.x() << "," << id.y() << "," << id.z() << "]\n";
            *reinterpret_cast<float *>(input_iter.ptr()) = lp->input[(id.z()*lp->height + id.y())*lp->width + id.x()];
        },
        input_iter);

    window.use_tensor_dimensions(weight->info()->tensor_shape());
    Iterator weight_iter(weight, window);
    execute_window_loop(window,
        // lambda function
        [&](const Coordinates & id) {
            //std::cout << "Setting item [" << id.x() << "," << id.y() << "," << id.z() << "]\n";
            *reinterpret_cast<float *>(weight_iter.ptr()) = lp->weight[((id[3]*lp->cin + id[2]) * lp->kernel + id[1]) * lp->kernel + id[0]];
        },
        weight_iter);

  } else {
    // # the following manner is wrong the assgin the value, never call this
    //for (size_t i = 0; i < input_shape.total_size(); i++) ptr[i] = 1.0;
  }

  // run
  struct timeval begin, end;
  gettimeofday(&begin, NULL);
  conv->run();
  gettimeofday(&end, NULL);
  std::cout << "time cost for execution kernel " << (end.tv_sec - begin.tv_sec)*1000000 + (end.tv_usec - begin.tv_usec) << " us\n";

  if(lp->buffer_ready)
  {
    Window window;
    window.use_tensor_dimensions(output->info()->tensor_shape());
    Iterator output_iter(output, window);
    execute_window_loop(window,
        // lambda function
        [&](const Coordinates & id) {
            //std::cout << "Setting item [" << id.x() << "," << id.y() << "," << id.z() << "]\n";
            memcpy(lp->cpu_buffer + (id.z() * lp->outh() + id.y()) * lp->outw(), output_iter.ptr(), lp->outw() * sizeof(float));
        },
        output_iter);
  }

  std::cout << "Leaving " << __FUNCTION__ << std::endl;
  return 0;
}

int test_gpu(LayerParameter *lp, DataType dtype=DataType::F32)
{
  std::cout << "Enter " << __FUNCTION__ << std::endl;

  CLTuner cl_tuner;
  arm_compute::CLScheduler::get().default_init();

  // init data tensor
  CLTensor* data  = new CLTensor();
  const TensorShape input_shape(lp->width, lp->height, lp->cin);
  data->allocator()->init(TensorInfo(input_shape, 1, dtype));

  // init weight tensor
  CLTensor* weight = new CLTensor();
  const TensorShape weight_shape(lp->kernel, lp->kernel, lp->cin, lp->cout);
  weight->allocator()->init(TensorInfo(weight_shape, 1, dtype));

  // init output tensor
  CLTensor* output = new CLTensor();
  const TensorShape output_shape(lp->outw(), lp->outh(), lp->cout);
  output->allocator()->init(TensorInfo(output_shape, 1, dtype));

  // init conv layer
  PadStrideInfo conv_info(lp->stride, lp->stride, lp->pad, lp->pad);
  WeightsInfo weight_info = WeightsInfo();
  ActivationLayerInfo act_info = ActivationLayerInfo();

  CLConvolutionLayer* conv = new CLConvolutionLayer();
  conv->configure(data, weight, nullptr, output, conv_info);

  Status status = conv->validate(data->info(), weight->info(), nullptr, output->info(), conv_info);
  if((bool)status != true)
    std::cout << "Return Error: " << status.error_description() << std::endl;

  ConvolutionMethod method = conv->get_convolution_method(data->info(), weight->info(), output->info(),
      conv_info, weight_info, act_info, arm_compute::GPUTarget::BIFROST);
  if(method == arm_compute::ConvolutionMethod::GEMM)
    std::cout << "GEMM" << std::endl;
  else if(method == arm_compute::ConvolutionMethod::WINOGRAD)
    std::cout << "WINOGRAD" << std::endl;
  else if(method == arm_compute::ConvolutionMethod::DIRECT)
    std::cout << "DIRECT" << std::endl;
  else
    std::cout << "Unkown" << std::endl;

  // allocate and assgin value
  data->allocator() -> allocate();
  weight->allocator()->allocate();
  output->allocator()->allocate();
  weight->map(true);
  data->map(true);
  if(lp->buffer_ready)
  {
    Window window;
    window.use_tensor_dimensions(data->info()->tensor_shape());
    //std::cout << " Dimensions of the input's iterator:\n";
    //std::cout << " X = [start=" << window.x().start() << ", end=" << window.x().end() << ", step=" << window.x().step() << "]\n";
    //std::cout << " Y = [start=" << window.y().start() << ", end=" << window.y().end() << ", step=" << window.y().step() << "]\n";
    //std::cout << " Z = [start=" << window.z().start() << ", end=" << window.z().end() << ", step=" << window.z().step() << "]\n";

    Iterator input_iter(data, window);
    execute_window_loop(window,
        // lambda function
        [&](const Coordinates & id) {
            //std::cout << "Setting item [" << id.x() << "," << id.y() << "," << id.z() << "]\n";
            *reinterpret_cast<float *>(input_iter.ptr()) = lp->input[(id.z()*lp->height + id.y())*lp->width + id.x()];
        },
        input_iter);

    window.use_tensor_dimensions(weight->info()->tensor_shape());
    Iterator weight_iter(weight, window);
    execute_window_loop(window,
        // lambda function
        [&](const Coordinates & id) {
            //std::cout << "Setting item [" << id.x() << "," << id.y() << "," << id.z() << "]\n";
            *reinterpret_cast<float *>(weight_iter.ptr()) = lp->weight[((id[3]*lp->cin + id[2]) * lp->kernel + id[1]) * lp->kernel + id[0]];
        },
        weight_iter);

  }
  data->unmap();
  weight->unmap();

  // run
  CLScheduler::get().sync();

  struct timeval begin, end;
  gettimeofday(&begin, NULL);
  conv->run();
  CLScheduler::get().sync();
  gettimeofday(&end, NULL);
  std::cout << "time cost for execution kernel " << (end.tv_sec - begin.tv_sec)*1000000 + (end.tv_usec - begin.tv_usec) << " us\n";

  // print result
  output->map(true);
  if(lp->buffer_ready)
  {
    Window window;
    window.use_tensor_dimensions(output->info()->tensor_shape());
    Iterator output_iter(output, window);
    execute_window_loop(window,
        // lambda function
        [&](const Coordinates & id) {
            //std::cout << "Setting item [" << id.x() << "," << id.y() << "," << id.z() << "]\n";
            memcpy(lp->gpu_buffer + (id.z() * lp->outh() + id.y()) * lp->outw(), output_iter.ptr(), lp->outw() * sizeof(float));
        },
        output_iter);
  }
  output->unmap();

  std::cout << "Leaving " << __FUNCTION__ << std::endl;
  return 0;
}

int main(int argc, char **argv)
{
  if(debug) printf("[debug] @ line %d: \n", __LINE__);
  char workspace[256];
  strcpy(workspace, argv[0]);
  int i;

  LayerParameter layers[7];
  layers[1].width = layers[1].height = 112;
  layers[2].width = layers[2].height = 224;
  layers[3].width = layers[3].height = 448;
  layers[4].cin = layers[4].cout = 32;
  layers[5].cin = layers[5].cout = 128;
  layers[6].cin = layers[6].cout = 256;
  for(i=0; i<7; i++) {
    layers[i].init();
    layers[i].print();
    test_cpu(layers + i, DataType::F32);
    test_gpu(layers + i, DataType::F16);
  }
  //lp.save(workspace);
  return 0;
}

