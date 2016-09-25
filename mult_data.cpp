#include "nnlib.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc,char **argv){
	if(argc<3){
		printf("%s <data bin name> <num mult>\n",argv[0]);
		return -1;
	}
	test_data *data = read_test_data_from_binary(argv[1]);	
	int num_test_data_overall = data->num_test_data*atoi(argv[2]);

	FILE* fp = fopen(argv[1],"wb");
	fwrite(&num_test_data_overall,sizeof(int),1,fp); 
	fwrite(&data->num_output,sizeof(int),1,fp); 
	fwrite(&data->num_input,sizeof(int),1,fp); 

	for(int i = 0;i<atoi(argv[2]);i++){
	for(int i = 0;i<data->num_test_data;i++){
		fwrite(data->input[i],sizeof(float),data->num_input,fp);
		fwrite(data->output[i],sizeof(float),data->num_output,fp);
	}
	}
	fclose(fp);
	destroy_test_data(data);

	return 0;
}
