#include <stdio.h>
#include <fann.h>

int main(int argc,char **argv){

	if(argc<4){
		printf("%s <net name> <binary net name>\n",argv[0]);
		return -1;
	}
	struct fann *ann = fann_create_standard(4,2,4000,4000,1);
	//struct fann *ann = fann_create_from_file(argv[1]);
	int num_layers = ann->last_layer-ann->first_layer;
	struct fann_layer *layer_it = ann->first_layer;
	FILE* fp = fopen(argv[2],"wb");
	int layer_size;

	fwrite(&ann->total_connections,sizeof(int),1,fp); //number of weights;
	fwrite(&num_layers,sizeof(int),1,fp);

	for(int i = 0;i<num_layers;i++,layer_it++){
		layer_size = layer_it->last_neuron-layer_it->first_neuron;
		fwrite(&layer_size,sizeof(int),1,fp);
	} 
	fwrite(ann->weights,sizeof(float),ann->total_connections,fp);

	fclose(fp);

	return 0;

}
