#ifndef __NNLIB_H_INCLUDED__
#define __NNLIB_H_INCLUDED__

struct neural_network{
  float* weights;
  int num_layers;
  int* layer_sizes;
  float **neuron_values;
  float **formatted_weights;
  float *output;
  float **predicted_outputs;
};

struct test_data{
  int num_test_data;
  int num_input;
  int num_output;
  float **input;
  float **output;
};

float *run_blas(int num_layers,int *layer_sizes,float** weights,float** neuron_values,int num_test_data);

void destroy_network(neural_network *mynet);

void destroy_test_data(test_data *data);

test_data *read_test_data_from_binary(char *binary_name);

neural_network *read_net_from_binary(char *net_name);

void run_net(struct neural_network *&mynet,struct test_data *data);

#endif
