#include "fann.h"
#include <iostream>
#include <stdio.h>

int main(int argc, char **argv){

    if(argc<3){
      std::cout << argv[0] << " <test data name> <output binary name>" << std::endl;
      return -1;
    }

    FILE* fp = fopen(argv[2],"wb");
    struct fann_train_data *test_data;
    test_data = fann_read_train_from_file(argv[1]);
    int num_test_data = fann_length_train_data(test_data);
    int num_input = fann_num_input_train_data(test_data);
    int num_output = fann_num_output_train_data(test_data);
 
    fwrite(&num_test_data,sizeof(int),1,fp); 
    fwrite(&num_output,sizeof(int),1,fp); 
    fwrite(&num_input,sizeof(int),1,fp); 

    for(int i = 0;i<num_test_data;i++){
      fwrite(test_data->input[i],sizeof(float),num_input,fp);
      fwrite(test_data->output[i],sizeof(float),num_output,fp);
    }
    
    fann_destroy_train(test_data);
    fclose(fp);

    return 0;
}
