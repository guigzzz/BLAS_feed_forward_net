#include <iostream>
#include <chrono>
#include "nnlib.h" //own header in same dir as cpp
#include <math.h>
using std::cout;
using std::endl;

int main(int argc, char* argv[])
{
	if(argc < 2){
		fprintf(stderr, "%s <network binary> <validation binary> \n", argv[0]);
		return -1;
	}

	test_data *data;
	data = read_test_data_from_binary(argv[2]);
	cout << "data reading done" << endl;

	neural_network *mynet = read_net_from_binary(argv[1]); 
	cout << "net reading done" << endl;
	cout << "running net" << endl;

	namespace chr = std::chrono;
	using hrc = chr::high_resolution_clock;
	int num_loops = 5;
	double avg_time;

	for(int i = 0;i<mynet->num_layers;i++){
		mynet->neuron_values[i] = (float *)malloc(data->num_test_data*mynet->layer_sizes[i]*sizeof(float));
	}

	mynet->predicted_outputs = (float **)malloc(data->num_test_data*sizeof(float *));

	//link predicted outputs to output values
	for(int i = 0;i<data->num_test_data*mynet->layer_sizes[mynet->num_layers-1];i++){
		if(i%mynet->layer_sizes[mynet->num_layers-1] == 0){
			mynet->predicted_outputs[i/mynet->layer_sizes[mynet->num_layers-1]] = &mynet->neuron_values[mynet->num_layers-1][i];
		}
	}

	for(int i = 0;i<num_loops+0;i++){
		auto begin = hrc::now();

		run_net(mynet,data);

		auto end = hrc::now();
		auto time_taken = chr::duration_cast<chr::milliseconds>(end-begin).count();
		std::cout << "iteration " << i+1 << ", time: " << time_taken/1000.0 << std::endl;
		avg_time += time_taken; 
	}
	avg_time = avg_time/num_loops;

	cout << "avg time over " << num_loops << " iterations: " << avg_time/1000 << endl;
	/*
	for(int i = 0;i<data->num_test_data;i++){
		for(int j = 0;j<data->num_output;j++){
		//	cout << "predicted: "<< mynet->predicted_outputs[i][j] << " actual: " << data->output[i][j] << endl;
		}
	}
	*/
	destroy_network(mynet);
	destroy_test_data(data);
	return 0;
}


