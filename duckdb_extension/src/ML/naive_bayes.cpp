//
// Created by Massimo Perini on 10/01/2024.
//
#include <ML/naive_bayes.h>
#include <utils.h>
#include <ML/utils.h>

#include <duckdb/function/scalar/nested_functions.hpp>

void ML::nb_train(duckdb::DataChunk &args, duckdb::ExpressionState &state,
              duckdb::Vector &result){

  duckdb::idx_t size = args.size();
  duckdb::vector<duckdb::Vector> &in_data = args.data;

  duckdb::Vector &in_cofactors = args.data[0];
  duckdb::Vector &in_labels = args.data[1];

  duckdb::RecursiveFlatten(in_cofactors, size);//these are lists
  duckdb::RecursiveFlatten(in_labels, size);//these are lists

  int n_aggregates = duckdb::ListVector::GetListSize(in_cofactors);

  //std::cout<<"a"<<std::endl;

  struct cofactor *cofactor = new struct cofactor[n_aggregates];
  extract_data(duckdb::ListVector::GetEntry(in_cofactors), cofactor, n_aggregates); // duckdb::value passed as
  //std::cout<<"---"<<std::endl;
  int drop_first = 0;//PG_GETARG_INT64(2);
  const auto &labels = duckdb::ConstantVector::GetData<int>(duckdb::ListVector::GetEntry(in_labels));

  uint64_t *cat_array = NULL; //max. size
  uint32_t *cat_vars_idxs = NULL; // track start each cat. variable
  //std::cout<<"bbb"<<std::endl;
  size_t num_params = n_cols_1hot_expansion(cofactor, n_aggregates, &cat_vars_idxs, &cat_array, drop_first);//enable drop first

  float total_tuples = 0;
  for(size_t k=0; k<n_aggregates; k++) {
    total_tuples += cofactor[k].N;
  }
  //std::cout<<"b"<<std::endl;

  //compute mean and variance for every numerical feature
  //result = n. aggregates (classes), n. cat. values (cat_array size), cat_array, probs for each class, mean, variance for every num. feat. in 1st aggregate, prob. each cat. value 1st aggregate, ...

  size_t res_size = 0;

  if (cofactor[0].num_categorical_vars > 0) {
    res_size = ( ((2 * cofactor[0].num_continuous_vars * n_aggregates)//continuous
                                               + (cat_vars_idxs[cofactor[0].num_categorical_vars] *
                                                  n_aggregates)//categoricals
                                               + cofactor[0].num_categorical_vars + 1 //cat. vars. idxs
                                               + n_aggregates + n_aggregates + 1 + 1 +//init
                                               cat_vars_idxs[cofactor[0].num_categorical_vars]));//copy categorical
  }
  else{
    res_size =  ((2 * cofactor[0].num_continuous_vars * n_aggregates)//continuous
                                               + n_aggregates + n_aggregates + 1 + 1);
  }


  result.SetVectorType(duckdb::VectorType::CONSTANT_VECTOR);

  duckdb::ListVector::Reserve(result, res_size);
  duckdb::ListVector::SetListSize(result, res_size);
  auto out_data = duckdb::ConstantVector::GetData<float>(duckdb::ListVector::GetEntry(result));
  auto metadata_out = duckdb::ListVector::GetData(result);
  //auto metadata_out = duckdb::ListVector::GetData(duckdb::ListVector::GetEntry(result));
  metadata_out[0].offset = 0;
  metadata_out[0].length = res_size;

  //std::cout<<"res size "<<res_size<<std::endl;


  out_data[0] = n_aggregates;

  size_t k=2;
  if (cofactor[0].num_categorical_vars > 0) {
    out_data[1] = cofactor[0].num_categorical_vars + 1;

    //stores here cat_vars_idxs and cat_array. Label is not part of the columns
    for(size_t i=0; i<cofactor[0].num_categorical_vars + 1; i++)
      out_data[i+2] = (cat_vars_idxs[i]);

    for(size_t i=0; i<cat_vars_idxs[cofactor[0].num_categorical_vars]; i++)
      out_data[i+2+cofactor[0].num_categorical_vars+1] = cat_array[i];
    k=2+cat_vars_idxs[cofactor[0].num_categorical_vars]+cofactor[0].num_categorical_vars+1;


  }
  else
    out_data[1] = 0;

  //now insert labels
  for(size_t i=0; i<n_aggregates; i++) {
    out_data[k + i] = labels[i];
  }


  //start storing NB parameters

  k += n_aggregates + n_aggregates;
  size_t start_priors = k-n_aggregates;
  for(size_t i=0; i<n_aggregates; i++) {//each NB aggregate contains training data for a specific class
    //save here the frequency for categorical NB
    //these are the first NB parameters stored
    //std::cout<<"i "<<i<<std::endl;

    out_data[start_priors+i] = cofactor[i].N / total_tuples;

    for (size_t j=0; j<cofactor[i].num_continuous_vars; j++){//save numeric params (mean, variance) after n_aggregates values
      float mean = (float)cofactor[i].lin[j] / (float)cofactor[i].N;
      float variance = ((float)cofactor[i].quad[j] / (float)cofactor[i].N) - (mean * mean);
      out_data[k] = mean;
      out_data[k+1] = variance;
      k+=2;
    }
    if (cofactor[0].num_categorical_vars > 0)
      k += cat_vars_idxs[cofactor[0].num_categorical_vars];
  }
  if (cofactor[0].num_categorical_vars > 0) {
    k = n_aggregates + 2 + cat_vars_idxs[cofactor[0].num_categorical_vars] +
        (cofactor[0].num_continuous_vars * 2) + cofactor[0].num_categorical_vars + 1;
    for (size_t i = 0; i < n_aggregates; i++) {
      auto &v_map_cat = cofactor[i].lin_cat;//this is over nb aggregates
      for (size_t j = 0; j < cofactor[i].num_categorical_vars; j++) {
        auto &map_cat = v_map_cat[j];
        for (auto &item : map_cat) {
          size_t index = find_in_array(item.first, cat_array, cat_vars_idxs[j], cat_vars_idxs[j + 1]);
          out_data[index + k] = ((float) item.second / (float) cofactor[i].N);//after mean and variance stores prior for num. columns
        }
      }
      k += cat_vars_idxs[cofactor[i].num_categorical_vars] +
           (cofactor[i].num_continuous_vars * 2);//jump to the next space where cat. values need to be written
    }
  }

  delete[] cofactor;
  delete[] cat_vars_idxs;
  delete[] cat_array;


}

duckdb::unique_ptr<duckdb::FunctionData> ML::nb_train_bind(
    duckdb::ClientContext &context, duckdb::ScalarFunction &function,
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> &arguments){

  function.return_type = duckdb::LogicalType::LIST(duckdb::LogicalType::FLOAT);;
  return duckdb::make_uniq<duckdb::VariableReturnBindData>(function.return_type);
}

void ML::nb_impute(duckdb::DataChunk &args, duckdb::ExpressionState &state,
               duckdb::Vector &result){

  //params, feat cont, cat

  idx_t size = args.size();//n. of rows to return
  duckdb::vector<duckdb::Vector> &in_data = args.data;
  int columns = in_data.size();
  size_t num_cols = 0;
  size_t cat_cols = 0;

  duckdb::UnifiedVectorFormat input_data[1024];//todo vary length (columns)
  RecursiveFlatten(in_data[0], size);//means_covariance in postgresql
  bool normalize = args.data[1].GetValue(0).GetValue<bool>();

  for (idx_t j=2;j<columns;j++){
    auto col_type = in_data[j].GetType();
    in_data[j].ToUnifiedFormat(size, input_data[j-1]);
    if (col_type == duckdb::LogicalType::FLOAT || col_type == duckdb::LogicalType::DOUBLE)
      num_cols++;
    else if (col_type == duckdb::LogicalType::INTEGER)
      cat_cols++;
  }

  duckdb::ListVector::GetEntry(in_data[0]).ToUnifiedFormat(size, input_data[0]);

  //extract train data

  idx_t params_size = duckdb::ListVector::GetListSize(in_data[0]);
  float *params = new float[params_size];
  for(size_t i=0; i<params_size; i++){//copy params
    params[i] = duckdb::UnifiedVectorFormat::GetData<float>(input_data[0])[input_data[0].sel->get_index(i)];
  }

  ///
  int n_classes = (int) (params[0]);
  int size_idxs = (int) (params[1]);
  size_t k=2+n_classes + n_classes;//2+priors+labels
  size_t prior_offset = 2 + n_classes;
  uint64_t *cat_vars_idxs;
  uint64_t *cat_vars;


  //----------

  if(size_idxs > 0) {
    cat_vars_idxs = new uint64_t [size_idxs];//max. size
    for (size_t i = 0; i < size_idxs; i++)
      cat_vars_idxs[i] = params[i + 2];

    cat_vars = new uint64_t [cat_vars_idxs[size_idxs - 1]];//max. size
    for (size_t i = 0; i < cat_vars_idxs[size_idxs - 1]; i++)
      cat_vars[i] = params[i + 2 + size_idxs];

    k=2+cat_vars_idxs[size_idxs-1]+size_idxs+n_classes+n_classes;
    prior_offset = 2+cat_vars_idxs[size_idxs-1]+size_idxs+n_classes;
  }

  size_t idx_start_params = k;

  for(size_t row=0; row<size; row++){
    k = idx_start_params;
    int best_class = 0;
    double max_prob = 0;
    for(size_t i=0; i<n_classes; i++){
      double total_prob = params[prior_offset+i];
      //elog(WARNING, "total prob %f", total_prob);

      for(size_t j=0; j<num_cols; j++){
        double variance = (params[k+(j*2)+1]);
        variance += 0.000000001;//avoid division by 0
        double mean = (params[k+(j*2)]);
        float in_cont_data = duckdb::UnifiedVectorFormat::GetData<float>(input_data[j+1])[input_data[j+1].sel->get_index(row)];
        total_prob *= ((double)1 / sqrt(2*M_PI*variance)) * exp( -(pow((in_cont_data - mean), 2)) / ((double)2*variance));
        //elog(WARNING, "total prob %f (normal mean %lf var %lf)", total_prob, mean, variance);
      }

      k += (2*num_cols);
      if (size_idxs > 0) {//if categorical features
        for (size_t j = 0; j < cat_cols; j++) {
          int in_cat_class = duckdb::UnifiedVectorFormat::GetData<int>(input_data[j+1+num_cols])[input_data[j+1+num_cols].sel->get_index(row)];//cat_padding (1+n cont + value)
          size_t index = find_in_array(in_cat_class, cat_vars, cat_vars_idxs[j], cat_vars_idxs[j + 1]);
          //elog(WARNING, "class %zu index %zu", class, index);
          if (index == cat_vars_idxs[j + 1])//class not found in train dataset
            total_prob *= 0;
          else {
            total_prob *= (params[k + index]);//cat. feats need to be monot. increasing
            //elog(WARNING, "multiply %lf", DatumGetFloat4(arrayContent1[k + index]));
            //elog(WARNING, "total prob %lf", total_prob);
          }
        }
        k += cat_vars_idxs[cat_cols];
      }
      //elog(WARNING, "final prob for class %d %lf", i, total_prob);

      if (total_prob > max_prob){
        max_prob = total_prob;
        best_class = i;
      }
    }
    result.SetValue(row, duckdb::Value(params[prior_offset-n_classes+best_class]));

  }

  delete[] params;
  if(size_idxs > 0) {
    delete[] cat_vars_idxs;
    delete[] cat_vars;
  }

}

duckdb::unique_ptr<duckdb::FunctionData> ML::nb_impute_bind(
    duckdb::ClientContext &context, duckdb::ScalarFunction &function,
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> &arguments){


  auto struct_type = duckdb::LogicalType::INTEGER;
  function.return_type = struct_type;
  function.varargs = duckdb::LogicalType::ANY;
  return duckdb::make_uniq<duckdb::VariableReturnBindData>(function.return_type);


}