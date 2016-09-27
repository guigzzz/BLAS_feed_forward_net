#include <fann.h>
#include <iostream>

using std::cout;
using std::endl;

void fann_write_net_to_binary(struct fann* orig,char *name);
int main(int argc,char **argv){
	if(argc<3){
		cout << argv[0] << " <net txt> <net bin>" << endl;
		return -1;
	}	
	struct fann *ann = fann_create_from_file(argv[1]);
	fann_write_net_to_binary(ann,argv[2]);
	fann_destroy(ann);

	return 0;
}





//FANN_EXTERNAL struct fann* FANN_API fann_write_net_to_binary(struct fann* orig)
void fann_write_net_to_binary(struct fann* orig,char *name)
{

	FILE* fp = fopen(name,"wb");

	unsigned int num_layers = (unsigned int)(orig->last_layer - orig->first_layer);
	struct fann_layer *orig_layer_it, *copy_layer_it;
	unsigned int layer_size;
	struct fann_neuron *last_neuron,*orig_neuron_it,*copy_neuron_it;
	struct fann_neuron *orig_first_neuron,*copy_first_neuron;
	int do_read = 1;
	int i=0;
	int idx;
	int dont_read = 0;
//	copy->errno_f = orig->errno_f;
	fwrite(&orig->errno_f,sizeof(enum fann_errno_enum),1,fp);
	//write
	if (orig->errstr)
	{
		//strcpy(copy->errstr,orig->errstr);
		fwrite(&do_read,sizeof(int),1,fp);
		fwrite(&orig->errstr,sizeof(FANN_ERRSTR_MAX),1,fp);
		//write
	}
	else{
		//write null
		fwrite(&dont_read,sizeof(int),1,fp);
	}
	//copy->error_log = orig->error_log;
	//fwrite(&orig->error_log,sizeof(float),1,fp);
	//write
	// block write
	//copy->learning_rate = orig->learning_rate;
	fwrite(&orig->learning_rate,sizeof(float),1,fp);

	//copy->learning_momentum = orig->learning_momentum;
	fwrite(&orig->learning_momentum,sizeof(float),1,fp);

	//copy->connection_rate = orig->connection_rate;
	fwrite(&orig->connection_rate,sizeof(float),1,fp);

	//copy->network_type = orig->network_type;
	fwrite(&orig->network_type,sizeof(enum fann_nettype_enum),1,fp);

	//copy->num_MSE = orig->num_MSE;
	fwrite(&orig->num_MSE,sizeof(unsigned int),1,fp);

	//copy->MSE_value = orig->MSE_value;
	fwrite(&orig->MSE_value,sizeof(float),1,fp);

	//copy->num_bit_fail = orig->num_bit_fail;
	fwrite(&orig->num_bit_fail,sizeof(unsigned int),1,fp);

	//copy->bit_fail_limit = orig->bit_fail_limit;
	fwrite(&orig->bit_fail_limit,sizeof(fann_type),1,fp);

	//copy->train_error_function = orig->train_error_function;
	fwrite(&orig->train_error_function,sizeof(enum fann_errorfunc_enum),1,fp);

	//copy->train_stop_function = orig->train_stop_function;
	fwrite(&orig->train_stop_function,sizeof(enum fann_stopfunc_enum),1,fp);

	//copy->training_algorithm = orig->training_algorithm;
	fwrite(&orig->training_algorithm,sizeof(enum fann_train_enum),1,fp);

	//copy->callback = orig->callback;
	fwrite(&orig->callback,sizeof(fann_callback_type),1,fp);



	fwrite(&num_layers,sizeof(num_layers),1,fp);

	fwrite(&orig->num_input,sizeof(int),1,fp);

	fwrite(&orig->num_output,sizeof(int),1,fp);

	fwrite(&orig->total_neurons,sizeof(int),1,fp);

	fwrite(&orig->total_connections,sizeof(int),1,fp);
	// end block write
#ifndef FIXEDFANN
	//block write
	//copy->cascade_output_change_fraction = orig->cascade_output_change_fraction;
	fwrite(&orig->cascade_output_change_fraction,sizeof(float),1,fp);
	//copy->cascade_output_stagnation_epochs = orig->cascade_output_stagnation_epochs;
	fwrite(&orig->cascade_output_stagnation_epochs,sizeof(unsigned int),1,fp);
	//copy->cascade_candidate_change_fraction = orig->cascade_candidate_change_fraction;
	fwrite(&orig->cascade_candidate_change_fraction,sizeof(float),1,fp);
	//copy->cascade_candidate_stagnation_epochs = orig->cascade_candidate_stagnation_epochs;
	fwrite(&orig->cascade_candidate_stagnation_epochs,sizeof(unsigned int),1,fp);
	//copy->cascade_best_candidate = orig->cascade_best_candidate;
	fwrite(&orig->cascade_best_candidate,sizeof(unsigned int),1,fp);
	//copy->cascade_candidate_limit = orig->cascade_candidate_limit;
	fwrite(&orig->cascade_candidate_limit,sizeof(fann_type),1,fp);
	//copy->cascade_weight_multiplier = orig->cascade_weight_multiplier;
	fwrite(&orig->cascade_weight_multiplier,sizeof(fann_type),1,fp);
	//copy->cascade_max_out_epochs = orig->cascade_max_out_epochs;
	fwrite(&orig->cascade_max_out_epochs,sizeof(unsigned int),1,fp);
	//copy->cascade_max_cand_epochs = orig->cascade_max_cand_epochs;
	fwrite(&orig->cascade_max_cand_epochs,sizeof(unsigned int),1,fp);
	//end block write

	//copy->cascade_activation_functions_count = orig->cascade_activation_functions_count;
	fwrite(&orig->cascade_activation_functions_count,sizeof(int),1,fp);
	//write

	//memcpy(copy->cascade_activation_functions,orig->cascade_activation_functions,
	//		copy->cascade_activation_functions_count * sizeof(enum fann_activationfunc_enum));
	fwrite(orig->cascade_activation_functions,sizeof(enum fann_activationfunc_enum),orig->cascade_activation_functions_count,fp);
	//write

	//copy->cascade_activation_steepnesses_count = orig->cascade_activation_steepnesses_count;
	fwrite(&orig->cascade_activation_steepnesses_count,sizeof(int),1,fp);
	//write

	//memcpy(copy->cascade_activation_steepnesses,orig->cascade_activation_steepnesses,copy->cascade_activation_steepnesses_count * sizeof(fann_type));
	fwrite(&orig->cascade_activation_steepnesses,sizeof(fann_type),orig->cascade_activation_steepnesses_count,fp);
	//write
	//copy->cascade_num_candidate_groups = orig->cascade_num_candidate_groups;
	fwrite(&orig->cascade_num_candidate_groups,sizeof(unsigned int),1,fp);
	//write

	if (orig->cascade_candidate_scores == NULL)
	{
		//copy->cascade_candidate_scores = NULL;
		//write
		fwrite(&dont_read,sizeof(int),1,fp);
	}
	else
	{
		//write not null
		fwrite(&do_read,sizeof(int),1,fp);
		//memcpy(copy->cascade_candidate_scores,orig->cascade_candidate_scores,fann_get_cascade_num_candidates(copy) * sizeof(fann_type));
		fwrite(orig->cascade_candidate_scores,sizeof(fann_type),fann_get_cascade_num_candidates(orig),fp);
		//write
	}
#endif /* FIXEDFANN */
	// block write
	//copy->quickprop_decay = orig->quickprop_decay;
	fwrite(&orig->quickprop_decay,sizeof(float),1,fp);

	//copy->quickprop_mu = orig->quickprop_mu;
	fwrite(&orig->quickprop_mu,sizeof(float),1,fp);

	//copy->rprop_increase_factor = orig->rprop_increase_factor;
	fwrite(&orig->rprop_increase_factor,sizeof(float),1,fp);

	//copy->rprop_decrease_factor = orig->rprop_decrease_factor;
	fwrite(&orig->rprop_decrease_factor,sizeof(float),1,fp);

	//copy->rprop_delta_min = orig->rprop_delta_min;
	fwrite(&orig->rprop_delta_min,sizeof(float),1,fp);

	//copy->rprop_delta_max = orig->rprop_delta_max;
	fwrite(&orig->rprop_delta_max,sizeof(float),1,fp);

	//copy->rprop_delta_zero = orig->rprop_delta_zero;
	fwrite(&orig->rprop_delta_zero,sizeof(float),1,fp);
	//end block write

#ifdef FIXEDFANN
	//block write
	//copy->decimal_point = orig->decimal_point;
	fwrite(&orig->decimal_point,sizeof(unsigned int),1,fp);

	//copy->multiplier = orig->multiplier;
	fwrite(&orig->multiplier,sizeof(unsigned int),1,fp);

	//memcpy(copy->sigmoid_results,orig->sigmoid_results,6*sizeof(fann_type));
	fwrite(&orig->sigmoid_results,sizeof(fann_type),6,fp);

	//memcpy(copy->sigmoid_values,orig->sigmoid_values,6*sizeof(fann_type));
	fwrite(&orig->sigmoid_values,sizeof(fann_type),6,fp);

	//memcpy(copy->sigmoid_symmetric_results,orig->sigmoid_symmetric_results,6*sizeof(fann_type));
	fwrite(orig->sigmoid_symmetric_results,sizeof(fann_type),6,fp);

	//memcpy(copy->sigmoid_symmetric_values,orig->sigmoid_symmetric_values,6*sizeof(fann_type));
	fwrite(orig->sigmoid_symmetric_values,sizeof(fann_type),6,fp);
	//end block write
#endif

	//write layer sizes
	for (orig_layer_it = orig->first_layer;
			orig_layer_it != orig->last_layer; orig_layer_it++)
	{
		layer_size = (unsigned int)(orig_layer_it->last_neuron - orig_layer_it->first_neuron);
		fwrite(&layer_size,sizeof(unsigned int),1,fp);
	}

	/* copy scale parameters, when used */
#ifndef FIXEDFANN
	//block write
	if (orig->scale_mean_in != NULL)
	{
		//write that this isnt null
		fwrite(&do_read,sizeof(int),1,fp);

		for (i=0; i < orig->num_input ; i++) {

			//copy->scale_mean_in[i] = orig->scale_mean_in[i];
			fwrite(&orig->scale_mean_in[i],sizeof(float),1,fp);

			//copy->scale_deviation_in[i] = orig->scale_deviation_in[i];
			fwrite(&orig->scale_deviation_in[i],sizeof(float),1,fp);

			//copy->scale_new_min_in[i] = orig->scale_new_min_in[i];
			fwrite(&orig->scale_new_min_in[i],sizeof(float),1,fp);

			//copy->scale_factor_in[i] = orig->scale_factor_in[i];
			fwrite(&orig->scale_factor_in[i],sizeof(float),1,fp);
		}
		for (i=0; i < orig->num_output ; i++) {
			//copy->scale_mean_in[i] = orig->scale_mean_in[i];
			fwrite(&orig->scale_mean_in[i],sizeof(float),1,fp);

			//copy->scale_deviation_in[i] = orig->scale_deviation_in[i];
			fwrite(&orig->scale_deviation_in[i],sizeof(float),1,fp);

			//copy->scale_new_min_in[i] = orig->scale_new_min_in[i];
			fwrite(&orig->scale_new_min_in[i],sizeof(float),1,fp);

			//copy->scale_factor_in[i] = orig->scale_factor_in[i];
			fwrite(&orig->scale_factor_in[i],sizeof(float),1,fp);

		}
	}
	else{
		//write that is null
		fwrite(&dont_read,sizeof(int),1,fp);
	}
	//end block write
#endif
	//write weights
	for(i = 0; i<orig->total_connections;i++){
		idx = orig->connections[i] - orig->connections[0];
		fwrite(&idx,sizeof(int),1,fp);
		fwrite(&orig->weights[i],sizeof(fann_type),1,fp);
	}

	//write neuron data	
	last_neuron = (orig->last_layer - 1)->last_neuron;
	for (orig_neuron_it = orig->first_layer->first_neuron;
			orig_neuron_it != last_neuron; orig_neuron_it++)
	{
		fwrite(&orig_neuron_it->first_con,sizeof(unsigned),1,fp);
		fwrite(&orig_neuron_it->last_con,sizeof(unsigned),1,fp);
		fwrite(&orig_neuron_it->activation_function,sizeof(int),1,fp);
		fwrite(&orig_neuron_it->activation_steepness,sizeof(fann_type),1,fp);
	}



	if (orig->train_slopes)
	{
		fwrite(&do_read,sizeof(int),1,fp);
		//memcpy(copy->train_slopes,orig->train_slopes,copy->total_connections_allocated * sizeof(fann_type));
		fwrite(orig->train_slopes,sizeof(fann_type),orig->total_connections,fp); //total_connections_allocated??
	}
	else{
		//write if not
		fwrite(&dont_read,sizeof(int),1,fp);
	}

	if (orig->prev_steps)
	{
		fwrite(&do_read,sizeof(int),1,fp);
		//memcpy(copy->prev_steps, orig->prev_steps, copy->total_connections_allocated * sizeof(fann_type));
		fwrite(orig->prev_steps,sizeof(fann_type),orig->total_connections,fp);
	}
	else{
		//write if not
		fwrite(&dont_read,sizeof(int),1,fp);
	}

	if (orig->prev_train_slopes)
	{
		fwrite(&do_read,sizeof(int),1,fp);
		//memcpy(copy->prev_train_slopes,orig->prev_train_slopes, copy->total_connections_allocated * sizeof(fann_type));
		fwrite(orig->prev_train_slopes,sizeof(fann_type),orig->total_connections,fp);
	}
	else{
		//write if not
		fwrite(&dont_read,sizeof(int),1,fp);
	}


	if (orig->prev_weights_deltas)
	{
		fwrite(&do_read,sizeof(int),1,fp);
		//memcpy(copy->prev_weights_deltas, orig->prev_weights_deltas,copy->total_connections_allocated * sizeof(fann_type));
		fwrite(orig->prev_weights_deltas,sizeof(fann_type),orig->total_connections,fp);
	}
	else{

		fwrite(&dont_read,sizeof(int),1,fp);
	}

	fclose(fp);
}


