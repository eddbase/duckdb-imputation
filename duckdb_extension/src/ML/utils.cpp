

#include <ML/utils.h>

void extract_data(const duckdb::Vector &triple, cofactor &cofactor){
  //extract data
  std::cerr<<"extract data"<<std::endl;

  auto &triple_struct = duckdb::StructVector::GetEntries(triple);

  duckdb::Vector v_num_lin = duckdb::ListVector::GetEntry(*triple_struct[1]);
  duckdb::Vector v_num_quad = duckdb::ListVector::GetEntry(*triple_struct[2]);
  duckdb::Vector v_cat_lin = duckdb::ListVector::GetEntry(*triple_struct[3]);
  duckdb::Vector v_num_cat_quad_1 = duckdb::ListVector::GetEntry(*triple_struct[4]);
  duckdb::Vector v_cat_cat_quad_1 = duckdb::ListVector::GetEntry(*triple_struct[5]);
  std::cerr<<"a"<<std::endl;

  cofactor.N = duckdb::FlatVector::GetData<int32_t>(*triple_struct[0])[0];
  std::cerr<<"err"<<std::endl;

  //duckdb::child_list_t<duckdb::Value> struct_values;
  //const duckdb::vector<duckdb::Value> &linear = duckdb::ListValue::GetChildren(first_triple_children[1]);
  size_t cont_cols = duckdb::ListVector::GetListSize(*triple_struct[1]);
  cofactor.lin.reserve(cont_cols);
  cofactor.num_continuous_vars = cont_cols;
  std::cerr<<"b"<<std::endl;

  //const duckdb::vector<duckdb::Value> &categorical = duckdb::ListValue::GetChildren(first_triple_children[3]);
  size_t cat_cols = duckdb::ListVector::GetListSize(*triple_struct[3]);
  cofactor.lin_cat.reserve(cat_cols);
  cofactor.num_categorical_vars = cat_cols;

  //duckdb::ListVector::GetData<float>(lin_data[0]);
  auto lin_res_num = duckdb::FlatVector::GetData<float>(v_num_lin);
  //lin numerical
  for(idx_t i=0;i<cont_cols;i++)
    cofactor.lin.push_back(lin_res_num[i]);
  std::cerr<<"c"<<std::endl;

  //copy lin. categorical

  if (duckdb::ListVector::GetListSize(*triple_struct[3]) > 0){
    duckdb::vector<duckdb::unique_ptr<duckdb::Vector>> &lin_struct_vector = duckdb::StructVector::GetEntries(duckdb::ListVector::GetEntry(v_cat_lin));//expands the struct data
    std::cerr<<"d"<<std::endl;
    auto cat_lin_keys = duckdb::FlatVector::GetData<int>(*((lin_struct_vector)[0]));
    auto cat_lin_vals = duckdb::FlatVector::GetData<float>(*((lin_struct_vector)[1]));//raw data
    std::cerr<<"d"<<std::endl;

    auto sublist_meta = duckdb::ListVector::GetData(v_cat_lin);
    std::cerr<<"d"<<std::endl;
    for(idx_t i=0;i<duckdb::ListVector::GetListSize(*triple_struct[3]);i++) {//for each cat variable
      cofactor.lin_cat.emplace_back();//add new map
      for(size_t j=0; j<sublist_meta[i].length; j++){//how many records this struct
        cofactor.lin_cat[i][cat_lin_keys[j + sublist_meta[i].offset]] = cat_lin_vals[j + sublist_meta[i].offset];
      }
    }
  }
  std::cerr<<"d"<<std::endl;


  //quad numerical

  cofactor.quad.reserve(duckdb::ListVector::GetListSize(*triple_struct[2]));
  auto quad_res_num = duckdb::FlatVector::GetData<float>(v_num_quad);
  for(idx_t i=0;i<duckdb::ListVector::GetListSize(*triple_struct[2]);i++)
    cofactor.quad.push_back(quad_res_num[i]);


  //num*cat
  std::cerr<<"e"<<std::endl;

  if(cofactor.lin.size() > 0 && cofactor.lin_cat.size() > 0){
    duckdb::Vector v_num_cat_structs = duckdb::ListVector::GetEntry(v_num_cat_quad_1);//each list eleemnt is a struct
    duckdb::vector<duckdb::unique_ptr<duckdb::Vector>> &num_cat_struct_vector = duckdb::StructVector::GetEntries(v_num_cat_structs);//expands the struct data
    auto num_cat_keys = duckdb::FlatVector::GetData<int>(*((num_cat_struct_vector)[0]));
    auto num_cat_vals = duckdb::FlatVector::GetData<float>(*((num_cat_struct_vector)[1]));//raw data
    auto num_cat_meta = duckdb::ListVector::GetData(v_num_cat_quad_1);

    cofactor.num_cat.reserve(cofactor.lin.size() * cofactor.lin_cat.size());
    for(idx_t num=0;num<cofactor.lin.size();num++) {
      for (idx_t cat = 0; cat < cofactor.lin_cat.size(); cat++) {
        //copy cat*num values
        size_t idx = (num*cofactor.lin_cat.size()) + cat;
        cofactor.num_cat.emplace_back();
        for (size_t j = 0; j < num_cat_meta[idx].length; j++) {//copy actual data
          cofactor.num_cat[idx][num_cat_keys[j + num_cat_meta[idx].offset]] = num_cat_vals[j + num_cat_meta[idx].offset];
        }
      }
    }
  }

  std::cerr<<"f"<<std::endl;

  if(cofactor.lin_cat.size() > 0){

    duckdb::Vector v_cat_cat_structs = duckdb::ListVector::GetEntry(v_cat_cat_quad_1);//each list eleemnt is a struct
    duckdb::vector<duckdb::unique_ptr<duckdb::Vector>> &cat_cat_struct_vector = duckdb::StructVector::GetEntries(v_cat_cat_structs);//expands the struct data
    auto cat_cat_key_1 = duckdb::FlatVector::GetData<int>(*((cat_cat_struct_vector)[0]));
    auto cat_cat_key_2 = duckdb::FlatVector::GetData<int>(*((cat_cat_struct_vector)[1]));
    auto cat_cat_vals = duckdb::FlatVector::GetData<float>(*((cat_cat_struct_vector)[2]));//raw data
    auto cat_cat_meta = duckdb::ListVector::GetData(v_cat_cat_quad_1);

    //cat*cat
    cofactor.cat_cat.reserve((cofactor.lin_cat.size() * (cofactor.lin_cat.size()+1))/2);
    size_t idx = 0;
    for(idx_t cat_1=0; cat_1<cofactor.lin_cat.size(); cat_1++) {
      for (idx_t cat_2 = cat_1; cat_2 < cofactor.lin_cat.size(); cat_2++) {
        cofactor.cat_cat.emplace_back();
        for (size_t j = 0; j < cat_cat_meta[idx].length; j++) {
          cofactor.cat_cat[idx][std::pair<int, int>(cat_cat_key_1[j + cat_cat_meta[idx].offset], cat_cat_key_2[j + cat_cat_meta[idx].offset])] = cat_cat_vals[j + cat_cat_meta[idx].offset];
        }
        idx++;
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
