

#include <ML/utils.h>


void extract_data(const duckdb::Vector &triple, cofactor *cofactor, size_t n_cofactor){
  //extract data
  std::cerr<<"extract data"<<std::endl;
  //triple needs to be a struct! if a list pass underlying struct
  auto &triple_struct = duckdb::StructVector::GetEntries(triple);

  duckdb::Vector v_num_lin = duckdb::ListVector::GetEntry(*triple_struct[1]);
  duckdb::Vector v_num_quad = duckdb::ListVector::GetEntry(*triple_struct[2]);
  duckdb::Vector v_cat_lin = duckdb::ListVector::GetEntry(*triple_struct[3]);
  //std::cerr<<"a"<<std::endl;

  for(size_t i=0; i<n_cofactor; i++)
    cofactor[i].N = duckdb::FlatVector::GetData<int32_t>(*triple_struct[0])[i];

  //std::cerr<<"err"<<std::endl;

  //duckdb::child_list_t<duckdb::Value> struct_values;
  //const duckdb::vector<duckdb::Value> &linear = duckdb::ListValue::GetChildren(first_triple_children[1]);
  size_t cont_cols = duckdb::ListVector::GetListSize(*triple_struct[1])/n_cofactor;
  for(size_t i=0; i<n_cofactor; i++) {
    cofactor[i].lin.reserve(cont_cols);
    cofactor[i].num_continuous_vars = cont_cols;
  }
  //std::cerr<<"b"<<std::endl;

  //const duckdb::vector<duckdb::Value> &categorical = duckdb::ListValue::GetChildren(first_triple_children[3]);
  size_t cat_cols = duckdb::ListVector::GetListSize(*triple_struct[3])/n_cofactor;

  for(size_t i=0; i<n_cofactor; i++) {
    cofactor[i].lin_cat.reserve(cat_cols);
    cofactor[i].num_categorical_vars = cat_cols;
  }

  //duckdb::ListVector::GetData<float>(lin_data[0]);
  auto lin_res_num = duckdb::FlatVector::GetData<float>(v_num_lin);
  //lin numerical
  for (size_t j=0; j<n_cofactor; j++) {
    for (idx_t i = 0; i < cont_cols; i++) {
      //std::cout<<"pushing "<<lin_res_num[i + (j * cont_cols)]<<std::endl;
      cofactor[j].lin.push_back(lin_res_num[i + (j * cont_cols)]);
    }
  }

  //copy lin. categorical
  if (duckdb::ListVector::GetListSize(*triple_struct[3]) > 0){
    //a list of struct
    duckdb::vector<duckdb::unique_ptr<duckdb::Vector>> &lin_struct_vector = duckdb::StructVector::GetEntries(duckdb::ListVector::GetEntry(v_cat_lin));//expands the struct data
    //std::cerr<<"d"<<std::endl;
    auto cat_lin_keys = duckdb::FlatVector::GetData<int>(*((lin_struct_vector)[0]));
    auto cat_lin_vals = duckdb::FlatVector::GetData<float>(*((lin_struct_vector)[1]));//raw data
    //std::cerr<<"d"<<std::endl;
    auto sublist_meta = duckdb::ListVector::GetData(v_cat_lin);
    std::cout<<"cat cols "<<cat_cols<<std::endl;
    for(size_t k =0; k<n_cofactor; k++){
      for(idx_t i=0;i<cat_cols;i++) {//for each cat variable
        cofactor[k].lin_cat.emplace_back();//add new map
        for(size_t j=0; j<sublist_meta[i+ (k*cat_cols)].length; j++){//how many records in this struct
            std::cout<<"key:  "<<cat_lin_keys[j + sublist_meta[i + (k*cat_cols)].offset]<<"value:  "<<cat_lin_vals[j + sublist_meta[i+(k*cat_cols)].offset]<<" index "<<j + sublist_meta[i + (k*cat_cols)].offset<<std::endl;
            std::cout<<"cofactor "<<k<<"col "<<i<<std::endl;
            cofactor[k].lin_cat[i][cat_lin_keys[j + sublist_meta[i + (k*cat_cols)].offset]] = cat_lin_vals[j + sublist_meta[i+(k*cat_cols)].offset];
        }
      }
    }
  }
  //std::cerr<<"d"<<std::endl;

  if(triple_struct.size() == 4){//nb agg

    auto quad_res_num = duckdb::FlatVector::GetData<float>(v_num_quad);
    for(size_t j=0; j<n_cofactor; j++){
      //cofactor[j].quad.reserve(duckdb::ListVector::GetListSize(*triple_struct[2]));
      for(idx_t i=0;i<cont_cols;i++)
        cofactor[j].quad.push_back(quad_res_num[i + (j*cont_cols)]);
    }
    return;
  }
  else{
    //quad numerical
    auto quad_res_num = duckdb::FlatVector::GetData<float>(v_num_quad);
    for(size_t j=0; j<n_cofactor; j++){
      //cofactor[j].quad.reserve(duckdb::ListVector::GetListSize(*triple_struct[2]));
      for(idx_t i=0;i<(cont_cols*(cont_cols+1))/2;i++)
        cofactor[j].quad.push_back(quad_res_num[i + (j*(cont_cols*(cont_cols+1))/2)]);
    }
  }

  duckdb::Vector v_num_cat_quad_1 = duckdb::ListVector::GetEntry(*triple_struct[4]);
  duckdb::Vector v_cat_cat_quad_1 = duckdb::ListVector::GetEntry(*triple_struct[5]);


  //num*cat
  //std::cerr<<"e"<<std::endl;
  //all should have the same columns
  if(cofactor[0].lin.size() > 0 && cofactor[0].lin_cat.size() > 0){
    duckdb::Vector v_num_cat_structs = duckdb::ListVector::GetEntry(v_num_cat_quad_1);//each list eleemnt is a struct
    duckdb::vector<duckdb::unique_ptr<duckdb::Vector>> &num_cat_struct_vector = duckdb::StructVector::GetEntries(v_num_cat_structs);//expands the struct data
    auto num_cat_keys = duckdb::FlatVector::GetData<int>(*((num_cat_struct_vector)[0]));
    auto num_cat_vals = duckdb::FlatVector::GetData<float>(*((num_cat_struct_vector)[1]));//raw data
    auto num_cat_meta = duckdb::ListVector::GetData(v_num_cat_quad_1);

      for(size_t i=0; i<n_cofactor; i++){
      cofactor[i].num_cat.reserve(cofactor[i].lin.size() * cofactor[i].lin_cat.size());
      for(idx_t num=0;num<cofactor[i].lin.size();num++) {
        for (idx_t cat = 0; cat < cofactor[i].lin_cat.size(); cat++) {
          //copy cat*num values
          size_t idx = (i*(cofactor[i].lin_cat.size()*cofactor[i].lin.size()))+(num*cofactor[i].lin_cat.size()) + cat;
          cofactor[i].num_cat.emplace_back();
          for (size_t j = 0; j < num_cat_meta[idx].length; j++) {//copy actual data
            cofactor[i].num_cat[(num*cofactor[i].lin_cat.size()) + cat][num_cat_keys[j + num_cat_meta[idx].offset]] = num_cat_vals[j + num_cat_meta[idx].offset];
          }
        }
      }
    }
  }

  //std::cerr<<"f"<<std::endl;

  if(cofactor[0].lin_cat.size() > 0){

    duckdb::Vector v_cat_cat_structs = duckdb::ListVector::GetEntry(v_cat_cat_quad_1);//each list eleemnt is a struct
    duckdb::vector<duckdb::unique_ptr<duckdb::Vector>> &cat_cat_struct_vector = duckdb::StructVector::GetEntries(v_cat_cat_structs);//expands the struct data
    auto cat_cat_key_1 = duckdb::FlatVector::GetData<int>(*((cat_cat_struct_vector)[0]));
    auto cat_cat_key_2 = duckdb::FlatVector::GetData<int>(*((cat_cat_struct_vector)[1]));
    auto cat_cat_vals = duckdb::FlatVector::GetData<float>(*((cat_cat_struct_vector)[2]));//raw data
    auto cat_cat_meta = duckdb::ListVector::GetData(v_cat_cat_quad_1);

    //cat*cat
    for(size_t i=0; i<n_cofactor; i++){
      cofactor[i].cat_cat.reserve((cofactor[i].lin_cat.size() * (cofactor[i].lin_cat.size()+1))/2);
      size_t idx = 0;
      for(idx_t cat_1=0; cat_1<cofactor[i].lin_cat.size(); cat_1++) {
        for (idx_t cat_2 = cat_1; cat_2 < cofactor[i].lin_cat.size(); cat_2++) {
          cofactor[i].cat_cat.emplace_back();
          for (size_t j = 0; j < cat_cat_meta[idx].length; j++) {
            cofactor[i].cat_cat[idx][std::pair<int, int>(cat_cat_key_1[j + cat_cat_meta[idx].offset], cat_cat_key_2[j + cat_cat_meta[idx].offset])] = cat_cat_vals[j + cat_cat_meta[idx].offset];

            std::cout<<"CAT CAT key1 "<<cat_cat_key_1[j + cat_cat_meta[idx].offset]<<" key2 "<<cat_cat_key_2[j + cat_cat_meta[idx].offset]<<" value "<<cofactor[i].cat_cat[idx][std::pair<int, int>(cat_cat_key_1[j + cat_cat_meta[idx].offset], cat_cat_key_2[j + cat_cat_meta[idx].offset])]<<" index "<<j + cat_cat_meta[idx].offset<<std::endl;

          }
          idx++;
        }
      }
    }
  }
}

size_t find_in_array(uint64_t a, const uint64_t *array, size_t start, size_t end)
{
  size_t index = start;
  while (index < end)
  {
    if (array[index] == a)
      break;
    index++;
  }
  return index;
}



/**
 * Generates a sigma matrix, used for all the ML models.
 * @param cofactor Input cofactor
 * @param matrix_size size (dimension 1-hot encoded) of matrix
 * @param label_categorical_sigma if >=0 excludes i-th CATEGORICAL attribute from matrix (usually cat. label)
 * @param cat_array Array of (unique) categorical values for each categorical column
 * @param cat_vars_idxs for each categorical column, stores begin:end indices of cat_array
 * @param drop_first If true, skips first category in 1-hot encoding (useful to avoid collinearity in the matrix, QDA can't invert the matrix otherwise)
 * @param sigma Output matrix
 */
void build_sigma_matrix(const cofactor &cofactor, size_t matrix_size, int label_categorical_sigma, uint64_t *cat_array, uint32_t *cat_vars_idxs, int drop_first,
                        /* out */ double *sigma)
{
  // start numerical data:
  // count
  sigma[0] = cofactor.N;
  //matrix_size already removes a label if needs to be removed
  // sum1
  const std::vector<float> &sum1_scalar_array = cofactor.lin;
  for (size_t i = 0; i < cofactor.num_continuous_vars; i++){
    sigma[i + 1] = sum1_scalar_array[i];
    sigma[(i + 1) * matrix_size] = sum1_scalar_array[i];
  }

  //sum2 full matrix (from half)
  const std::vector<float> &sum2_scalar_array = cofactor.quad;
  for (size_t row = 0; row < cofactor.num_continuous_vars; row++){
    for (size_t col = 0; col < cofactor.num_continuous_vars; col++){
      if (row > col)
        sigma[((row + 1) * matrix_size) + (col + 1)] = sum2_scalar_array[(col * cofactor.num_continuous_vars) - (((col) * (col + 1)) / 2) + row];
      else
        sigma[((row + 1) * matrix_size) + (col + 1)] = sum2_scalar_array[(row * cofactor.num_continuous_vars) - (((row) * (row + 1)) / 2) + col];
    }
  }


  //create sorted array
  // (numerical_params) * (numerical_params) allocated
  // start relational data:
  cat_vars_idxs[0] = 0;
  size_t skipped_var_categories = 0;
  // count * categorical (group by A, group by B, ...)
  for (size_t i = 0; i < cofactor.num_categorical_vars; i++)
  {
    std::map<int, float> relation_data = cofactor.lin_cat[i];
    if (label_categorical_sigma >= 0 && ((size_t)label_categorical_sigma) == i )
    {
      //skip this variable
      skipped_var_categories = (cat_vars_idxs[i + 1] - cat_vars_idxs[i]);
      continue;
    }

    for (auto &item:relation_data)
    {
      // search key index
      size_t key_index = find_in_array(item.first, cat_array, cat_vars_idxs[i], cat_vars_idxs[i + 1]);
      if (drop_first && key_index == cat_vars_idxs[i + 1])
        continue;//skip

      assert(key_index < cat_vars_idxs[i + 1]);
      // add to sigma matrix
      key_index += cofactor.num_continuous_vars + 1 - skipped_var_categories;
      sigma[key_index] = item.second;
      sigma[key_index * matrix_size] = item.second;
      sigma[(key_index * matrix_size) + key_index] = item.second;
    }
  }
  // categorical * numerical
  for (size_t numerical = 1; numerical < cofactor.num_continuous_vars + 1; numerical++){
    skipped_var_categories = 0;
    for (size_t categorical = 0; categorical < cofactor.num_categorical_vars; categorical++){
      std::map<int, float> relation_data = cofactor.num_cat[(numerical-1)*cofactor.num_categorical_vars + categorical];
      if (label_categorical_sigma >= 0 && ((size_t)label_categorical_sigma) == categorical ){
        //skip this variable
        skipped_var_categories = (cat_vars_idxs[categorical + 1] - cat_vars_idxs[categorical]);
        continue;
      }

      for (auto &item:relation_data)
      {
        //search in the right categorical var
        size_t key_index = find_in_array(item.first, cat_array, cat_vars_idxs[categorical], cat_vars_idxs[categorical + 1]);
        if (drop_first && key_index == cat_vars_idxs[categorical + 1])
          continue;//skip

        assert(key_index < cat_vars_idxs[categorical + 1]);
        // add to sigma matrix
        key_index += cofactor.num_continuous_vars + 1 - skipped_var_categories;
        sigma[(key_index * matrix_size) + numerical] = item.second;
        sigma[(numerical * matrix_size) + key_index] = item.second;
      }
    }
  }
  size_t skip_sec_var_categories = 0;
  skipped_var_categories = 0;

  // categorical * categorical
  size_t curr_element = 0;
  for (size_t curr_cat_var = 0; curr_cat_var < cofactor.num_categorical_vars; curr_cat_var++){
    if (((size_t)label_categorical_sigma) == curr_cat_var) {
      skip_sec_var_categories = (cat_vars_idxs[curr_cat_var + 1] - cat_vars_idxs[curr_cat_var]);//row skip, won't be changed once setted
      skipped_var_categories = skip_sec_var_categories;
    }
    else if (skip_sec_var_categories == 0)//not yet seen skip vars in rows
      skipped_var_categories = 0;//reset if "column loop" iterates over a new row, unless the var to ignore has already been seen in row

    for (size_t other_cat_var = curr_cat_var; other_cat_var < cofactor.num_categorical_vars; other_cat_var++){//todo fix this in postgres
      if(((size_t)label_categorical_sigma) == other_cat_var)
        skipped_var_categories = (cat_vars_idxs[other_cat_var + 1] - cat_vars_idxs[other_cat_var]);

      std::map<std::pair<int, int>, float> relation_data = cofactor.cat_cat[curr_element];
      curr_element++;
      if (label_categorical_sigma >= 0 && (((size_t)label_categorical_sigma) == curr_cat_var || ((size_t)label_categorical_sigma) == other_cat_var))
      {
        //skip this variable
        continue;
      }

      for (auto &item:relation_data)
      {
        size_t key_index_curr_var = find_in_array(item.first.first, cat_array, cat_vars_idxs[curr_cat_var], cat_vars_idxs[curr_cat_var + 1]);
        if (drop_first && key_index_curr_var == cat_vars_idxs[curr_cat_var + 1])
          continue;//skip

        assert(key_index_curr_var < cat_vars_idxs[curr_cat_var + 1]);

        size_t key_index_other_var = find_in_array(item.first.second, cat_array, cat_vars_idxs[other_cat_var], cat_vars_idxs[other_cat_var + 1]);
        if (drop_first && key_index_other_var == cat_vars_idxs[other_cat_var + 1])
          continue;//skip

        assert(key_index_other_var < cat_vars_idxs[other_cat_var + 1]);

        // add to sigma matrix

        key_index_curr_var += cofactor.num_continuous_vars + 1 - skip_sec_var_categories;
        key_index_other_var += cofactor.num_continuous_vars + 1 - skipped_var_categories;

          std::cout<<"sigma: adding in "<<key_index_curr_var<<", "<<key_index_other_var<<"(original key: "<<item.first.first<<" - "<<item.first.second<<" with val: "<<item.second<<std::endl;

          sigma[(key_index_curr_var * matrix_size) + key_index_other_var] = item.second;
        sigma[(key_index_other_var * matrix_size) + key_index_curr_var] = item.second;
      }
    }
  }
}


void build_sigma_matrix(const cofactor &cofactor, size_t matrix_size, int label_categorical_sigma,
                        /* out */ double *sigma)
{
  // start numerical data:
  // numerical_params = cofactor->num_continuous_vars + 1

  // count
  sigma[0] = cofactor.N;

  // sum1
  const std::vector<float> &sum1_scalar_array = cofactor.lin;
  for (size_t i = 0; i < cofactor.num_continuous_vars; i++){
    sigma[i + 1] = sum1_scalar_array[i];
    sigma[(i + 1) * matrix_size] = sum1_scalar_array[i];
  }

  //sum2 full matrix (from half)
  const std::vector<float> &sum2_scalar_array = cofactor.quad;
  for (size_t row = 0; row < cofactor.num_continuous_vars; row++){
    for (size_t col = 0; col < cofactor.num_continuous_vars; col++){
      if (row > col)
        sigma[((row + 1) * matrix_size) + (col + 1)] = sum2_scalar_array[(col * cofactor.num_continuous_vars) - (((col) * (col + 1)) / 2) + row];
      else
        sigma[((row + 1) * matrix_size) + (col + 1)] = sum2_scalar_array[(row * cofactor.num_continuous_vars) - (((row) * (row + 1)) / 2) + col];
    }
  }
  // (numerical_params) * (numerical_params) allocated

  // start relational data:

  size_t num_categories = matrix_size - cofactor.num_continuous_vars - 1;
  uint64_t *cat_array = new uint64_t [num_categories]; // array of categories
  uint32_t *cat_vars_idxs = new uint32_t[cofactor.num_categorical_vars + 1]; // track start each cat. variable


  //linear categorical
  size_t search_start = 0;        // within one category class
  size_t search_end = search_start;

  cat_vars_idxs[0] = 0;

  for (size_t i = 0; i < cofactor.num_categorical_vars; i++) {
    std::map<int, float> relation_data = cofactor.lin_cat[i];
    if (label_categorical_sigma >= 0 && ((size_t)label_categorical_sigma) == i ){
      cat_vars_idxs[i + 1] = cat_vars_idxs[i];
      continue;
    }
    //create sorted array
    for (auto &element:relation_data) {
      size_t key_index = find_in_array(element.first, cat_array, search_start, search_end);
      if (key_index == search_end){
        uint64_t value_to_insert = element.first;
        uint64_t tmp;
        for (size_t k = search_start; k < search_end; k++){
          if (value_to_insert < cat_array[k]){
            tmp = cat_array[k];
            cat_array[k] = value_to_insert;
            value_to_insert = tmp;
          }
        }
        cat_array[search_end] = value_to_insert;
        search_end++;
      }
    }
    search_start = search_end;
    cat_vars_idxs[i + 1] = cat_vars_idxs[i] + relation_data.size();
  }


  // count * categorical (group by A, group by B, ...)
  for (size_t i = 0; i < cofactor.num_categorical_vars; i++)
  {
    std::map<int, float> relation_data = cofactor.lin_cat[i];
    if (label_categorical_sigma >= 0 && ((size_t)label_categorical_sigma) == i )
    {
      //skip this variable
      continue;
    }

    for (auto &item:relation_data)
    {
      // search key index
      size_t key_index = find_in_array(item.first, cat_array, cat_vars_idxs[i], cat_vars_idxs[i + 1]);
      assert(key_index < search_end);
      /*if (key_index == search_end)    // not found
      {
          cat_array[search_end] = r->tuples[j].key;
          search_end++;
      }*/

      // add to sigma matrix
      key_index += cofactor.num_continuous_vars + 1;
      sigma[key_index] = item.second;
      sigma[key_index * matrix_size] = item.second;
      sigma[(key_index * matrix_size) + key_index] = item.second;
    }
    search_start = search_end;
    //cat_vars_idxs[i + 1] = cat_vars_idxs[i] + r->num_tuples;
  }

  // categorical * numerical
  for (size_t numerical = 1; numerical < cofactor.num_continuous_vars + 1; numerical++)
  {
    for (size_t categorical = 0; categorical < cofactor.num_categorical_vars; categorical++)
    {
      std::map<int, float> relation_data = cofactor.num_cat[(numerical-1)*cofactor.num_categorical_vars + categorical];
      if (label_categorical_sigma >= 0 && ((size_t)label_categorical_sigma) == categorical )
      {
        //skip this variable
        continue;
      }

      for (auto &item:relation_data)
      {
        //search in the right categorical var
        search_start = cat_vars_idxs[categorical];
        search_end = cat_vars_idxs[categorical + 1];

        size_t key_index = find_in_array(item.first, cat_array, search_start, search_end);
        assert(key_index < search_end);

        // add to sigma matrix
        key_index += cofactor.num_continuous_vars + 1;
        sigma[(key_index * matrix_size) + numerical] = item.second;
        sigma[(numerical * matrix_size) + key_index] = item.second;
      }
    }
  }

  // categorical * categorical
  size_t curr_element = 0;
  for (size_t curr_cat_var = 0; curr_cat_var < cofactor.num_categorical_vars; curr_cat_var++)
  {
    for (size_t other_cat_var = curr_cat_var; other_cat_var < cofactor.num_categorical_vars; other_cat_var++)
    {
      std::map<std::pair<int, int>, float> relation_data = cofactor.cat_cat[curr_element];
      curr_element++;

      if (label_categorical_sigma >= 0 && (((size_t)label_categorical_sigma) == curr_cat_var || ((size_t)label_categorical_sigma) == other_cat_var))
      {
        //skip this variable
        continue;
      }

      for (auto &item:relation_data)
      {
        search_start = cat_vars_idxs[curr_cat_var];
        search_end = cat_vars_idxs[curr_cat_var + 1];

        size_t key_index_curr_var = find_in_array(item.first.first, cat_array, search_start, search_end);
        assert(key_index_curr_var < search_end);

        search_start = cat_vars_idxs[other_cat_var];
        search_end = cat_vars_idxs[other_cat_var + 1];

        size_t key_index_other_var = find_in_array(item.first.second, cat_array, search_start, search_end);
        assert(key_index_other_var < search_end);

        // add to sigma matrix
        key_index_curr_var += cofactor.num_continuous_vars + 1;
        key_index_other_var += cofactor.num_continuous_vars + 1;
        sigma[(key_index_curr_var * matrix_size) + key_index_other_var] = item.second;
        sigma[(key_index_other_var * matrix_size) + key_index_curr_var] = item.second;
      }
    }
  }


  delete [] cat_array;
  delete [] cat_vars_idxs;
}

size_t get_num_categories(const cofactor &cofactor, int label_categorical_sigma)
{
  size_t num_categories = 0;

  for (size_t i = 0; i < cofactor.num_categorical_vars; i++)
  {
    std::map<int, float> relation_data = cofactor.lin_cat[i];
    if (label_categorical_sigma >= 0 && ((size_t)label_categorical_sigma) == i)
    {
      //skip this variable
      continue;
    }

    num_categories += relation_data.size();
  }
  return num_categories;
}

size_t sizeof_sigma_matrix(const cofactor &cofactor, int label_categorical_sigma)
{
  // count :: numerical :: 1-hot_categories
  return 1 + cofactor.num_continuous_vars + get_num_categories(cofactor, label_categorical_sigma);
}



/**
 * Computes the n. of columns of a cofactor matrix 1-hot encoded from a sequence of aggregates
 * @param cofactors input aggregates ()
 * @param n_aggregates n. of group by aggregates
 * @param cat_idxs OUTPUT: Indices of cat_unique_array. For each column stores begin:end of cat. values inside cat_unique_array
 * @param cat_unique_array OUTPUT: for every categorical column, stores sorted categorical values
 * @param drop_first If true, remove first entry for each categorical attribute (avoids multicollinearity in case of QDA)
 * @return number of values in a 1-hot encoded vector given the aggregates
 */
size_t n_cols_1hot_expansion(const cofactor *cofactors, size_t n_aggregates, uint32_t **cat_idxs, uint64_t **cat_unique_array, int drop_first)
{
  size_t num_categories = 0;
  for(size_t k=0; k<n_aggregates; k++) {
    num_categories += get_num_categories(cofactors[k], -1);//potentially overestimate
  }


  uint32_t *cat_vars_idxs = new uint32_t[(cofactors[0].num_categorical_vars + 1)]; // track start each cat. variable
  uint64_t *cat_array = new uint64_t[num_categories];//max. size

  (*cat_idxs) = cat_vars_idxs;
  (*cat_unique_array) = cat_array;

  cat_vars_idxs[0] = 0;
  size_t search_start = 0;        // within one category class
  size_t search_end = search_start;

  for (size_t i = 0; i < cofactors[0].num_categorical_vars; i++) {
    for(size_t k=0; k<n_aggregates; k++) {
      //create sorted array
      for(auto &cat_val:cofactors[k].lin_cat[i]){
        size_t key_index = find_in_array(cat_val.first, cat_array, search_start, search_end);

        if (key_index == search_end) {
            std::cout<<"Inserting "<<cat_val.first<<std::endl;
          uint64_t value_to_insert = cat_val.first;
          uint64_t tmp;
          for (size_t k = search_start; k < search_end; k++) {
            if (value_to_insert < cat_array[k]) {
              tmp = cat_array[k];
              cat_array[k] = value_to_insert;
              value_to_insert = tmp;
            }
          }
          cat_array[search_end] = value_to_insert;
          search_end++;
        }
      }
    }
      std::cout<<"Done elements inserted: "<<search_end - search_start<<std::endl;
      cat_vars_idxs[i + 1] = cat_vars_idxs[i] + (search_end - search_start);
    search_start = search_end;
  }

  if (drop_first){//remove first entry for each categorical attribute (avoids multicollinearity when inverting matrix in QDA)
    for (size_t i = 0; i < cofactors[0].num_categorical_vars; i++){
      cat_vars_idxs[i+1]-= (i+1);
      for(size_t j=cat_vars_idxs[i]; j<cat_vars_idxs[i+1]; j++){
        cat_array[j] = cat_array[j+i+1];
      }
    }
  }

  // count + numerical + 1-hot_categories
  return 1 + cofactors[0].num_continuous_vars + cat_vars_idxs[cofactors[0].num_categorical_vars];
}



void standardize_sigma(double *sigma, size_t num_params, double *means, double *std){
  //compute mean and variance of every column
  //and standardize the data
  for(size_t i=0; i<num_params; i++)
    means[i] = sigma[i] / sigma[0];
  for(size_t i=0; i<num_params; i++)
    std[i] = sqrt((sigma[(i*num_params)+i]/sigma[0]) - pow(sigma[i]/sigma[0], 2));//0 variance for first col

  //standardize sigma matrix
  for(size_t i=1; i<num_params; i++){
    for (size_t j=1; j<num_params; j++){
      sigma[(i*num_params)+j] = (sigma[(i*num_params)+j] - (means[i]*sigma[j]) - (means[j]*sigma[i]) + (sigma[0]*means[j]*means[i])) / (std[i]*std[j]);
    }
  }
  //standarize sums. Sum of standardized values is 0 (and avoid division by 0)
  for(size_t i=1; i<num_params; i++){
    sigma[i] = 0;
    sigma[(i*num_params)] = 0;
  }
}