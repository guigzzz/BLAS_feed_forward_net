#include <iostream>
#include <cblas.h>
#include <chrono>
#include "nnlib.h" //own header in same dir as cpp
#include <boost/simd/function/exp.hpp>
#include <boost/simd/function/load.hpp>
#include <boost/simd/function/store.hpp>
#include <boost/simd/function/is_greater.hpp>
#include <boost/simd/function/if_else.hpp>
#include <boost/simd/constant/zero.hpp>
#include <boost/simd/constant/one.hpp>
#include <boost/simd/constant/mone.hpp>
#include <boost/simd/pack.hpp>

namespace bs = boost::simd;
using std::cout;
using std::endl;

void run_net(struct neural_network *&mynet,struct test_data *data){

	int num_test_data=data->num_test_data;
	int num_input=data->num_input;
	int num_output=data->num_output;
	float **input=data->input;
	float **output=data->output;
	int num_layers = mynet->num_layers;
	int *layer_sizes = mynet->layer_sizes;
	float **neuron_values = mynet->neuron_values;
	float **weights = mynet->formatted_weights;
	float **predicted_outputs = mynet->predicted_outputs;
	//fill input matrix

	for(int j = 0;j<num_test_data;j++){
		int k = 0;
		for(;k<layer_sizes[0]-1;k++){
			neuron_values[0][j * layer_sizes[0] + k] = input[j][k];
		}
		neuron_values[0][j*layer_sizes[0] + k] = 1; //set bias neuron
	}

	float *raw_output;
	raw_output = run_blas(num_layers,layer_sizes,weights,neuron_values,num_test_data);
	//remove biases
	for(int i = 0;i<data->num_test_data*layer_sizes[num_layers-1];i++){
		if(i%layer_sizes[num_layers-1] == 0){
			predicted_outputs[i/layer_sizes[num_layers-1]] = &raw_output[i];
		}
	}
}


float *run_blas(int num_layers,int *layer_sizes,float** weights,float** neuron_values,int num_test_data)
{
	using pack_t = bs::pack<float>;
	double blas_time = 0;
	namespace chr = std::chrono;
	using hrc = chr::high_resolution_clock;

	for(int i = 1;i<num_layers;i++){

		auto begin = hrc::now();

		cblas_sgemm(CblasColMajor,CblasTrans,CblasNoTrans,layer_sizes[i]-1,num_test_data,layer_sizes[i-1], 
				0.5,weights[i-1],layer_sizes[i-1],neuron_values[i-1],layer_sizes[i-1],0.0,neuron_values[i],layer_sizes[i]);

		auto end = hrc::now();
		auto time_taken = chr::duration_cast<chr::milliseconds>(end-begin).count();
		blas_time+=time_taken;

		for(int k = 0;k<num_test_data;k++){
			int j = 0;
			pack_t max_sum(300);
			pack_t ones = bs::One<pack_t>();
			pack_t twos(-2.f);
			for(;j + pack_t::static_size <=layer_sizes[i]-1;j+= pack_t::static_size){
				pack_t neuron_sum = bs::load<pack_t>(&neuron_values[i][k * layer_sizes[i] + j]);
				auto mask = bs::abs(neuron_sum) > max_sum;
				auto zero_mask = bs::is_greater(neuron_sum,bs::Zero<pack_t>());
				pack_t v0 = bs::if_else(mask, bs::if_else(zero_mask,bs::One<pack_t>(),bs::Mone<pack_t>()) * max_sum, neuron_sum);

				pack_t result = ones/(ones + bs::exp(twos * v0));
				bs::store(result,&neuron_values[i][k * layer_sizes[i] + j]);
			}

			for(;j<layer_sizes[i]-1;j++){ //-1 to not do bias neuron
				float neuron_sum_scalar = neuron_values[i][k * layer_sizes[i] + j];
				float max_sum_scalar = 300; //max sum = 150/activation_steepness

				if(neuron_sum_scalar > max_sum_scalar)
					neuron_sum_scalar = max_sum_scalar;
				else if(neuron_sum_scalar < -max_sum_scalar)
					neuron_sum_scalar = -max_sum_scalar;

				neuron_values[i][k * layer_sizes[i] + j] = 1.0f/(1.0f + bs::exp(-2.0f * neuron_sum_scalar)); //activation function
			}
			neuron_values[i][k * layer_sizes[i] + j] = 1; //set bias neuron
		}
	}
	cout << "total blas time: " << blas_time/1000 << endl;
	return neuron_values[num_layers-1];
}




void destroy_network(neural_network *mynet){
	for(int i = 0;i<mynet->num_layers;i++){
		free(mynet->neuron_values[i]);
	}
	free(mynet->neuron_values);
	free(mynet->formatted_weights);
	free(mynet->weights);
	free(mynet->layer_sizes);
	free(mynet->predicted_outputs);
	delete(mynet);
}




void destroy_test_data(test_data *data){
	for(int i = 0;i<data->num_test_data;i++){
		free(data->input[i]);
		free(data->output[i]);
	}
	free(data->input);
	free(data->output);
	delete(data);
}




test_data *read_test_data_from_binary(char *binary_name){

	FILE* fp = fopen(binary_name,"rb");
	test_data *data = new test_data;

	fread(&data->num_test_data,sizeof(int),1,fp); 
	fread(&data->num_output,sizeof(int),1,fp); 
	fread(&data->num_input,sizeof(int),1,fp); 

	data->input = (float **)malloc(data->num_test_data*sizeof(float *));
	data->output =(float **)malloc(data->num_test_data*sizeof(float *));

	for(int i = 0;i<data->num_test_data;i++){
		data->input[i]=(float *)malloc(data->num_input*sizeof(float));
		data->output[i]=(float *)malloc(data->num_output*sizeof(float));
	}

	for(int i = 0;i<data->num_test_data;i++){
		fread(data->input[i],sizeof(float),data->num_input,fp);
		fread(data->output[i],sizeof(float),data->num_output,fp);
	}

	fclose(fp);
	return data;
}




neural_network *read_net_from_binary(char *net_name){

	FILE* fp = fopen(net_name,"rb");
	neural_network *mynet = new neural_network;
	int total_connections;
	fread(&total_connections,sizeof(int),1,fp); //number of weights;

	fread(&mynet->num_layers,sizeof(int),1,fp);

	mynet->layer_sizes = (int *)malloc(mynet->num_layers*sizeof(int));

	fread(mynet->layer_sizes,sizeof(int),mynet->num_layers,fp);
	mynet->weights = (float *)malloc(total_connections*sizeof(float));
	fread(mynet->weights,sizeof(float),total_connections,fp);

	mynet->neuron_values = (float **)malloc(mynet->num_layers*sizeof(float *));

	mynet->formatted_weights = (float **)malloc((mynet->num_layers-1)*sizeof(float *));

	int cur_weight = 0;
	for(int i = 1;i<mynet->num_layers;i++){
		mynet->formatted_weights[i-1] = &mynet->weights[cur_weight];
		cur_weight += mynet->layer_sizes[i-1]*(mynet->layer_sizes[i]-1);
	}

	fclose(fp);
	return mynet;
}

