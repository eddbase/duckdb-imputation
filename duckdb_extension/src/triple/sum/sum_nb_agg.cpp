

#include <triple/sum/sum.h>
//#include <triple/From_duckdb.h>
#include "duckdb/function/scalar/nested_functions.hpp"
#include <triple/sum/sum_state.h>
#include <triple/sum/sum_nb_agg.h>
#include <iostream>
#include <memory>
#include <utils.h>



duckdb::unique_ptr<duckdb::FunctionData>
Triple::sum_nb_agg_bind(duckdb::ClientContext &context, duckdb::AggregateFunction &function,
        duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> &arguments) {

  D_ASSERT(arguments.size() == 1);
  D_ASSERT(function.arguments.size() == 1);

  //std::cout<<"SUM ARGS: "<<arguments[0]->ToString();
  //std::cout<<"SUM ARGS 2: "<<function.arguments[0].ToString();


  duckdb::child_list_t<duckdb::LogicalType> struct_children;
  struct_children.emplace_back("N", duckdb::LogicalType::INTEGER);
  struct_children.emplace_back("lin_agg", duckdb::LogicalType::LIST(duckdb::LogicalType::FLOAT));
  struct_children.emplace_back("quad_agg", duckdb::LogicalType::LIST(duckdb::LogicalType::FLOAT));

  //categorical structures
  duckdb::child_list_t<duckdb::LogicalType> lin_cat;
  lin_cat.emplace_back("key", duckdb::LogicalType::INTEGER);
  lin_cat.emplace_back("value", duckdb::LogicalType::FLOAT);

  struct_children.emplace_back("lin_cat", duckdb::LogicalType::LIST(duckdb::LogicalType::LIST(duckdb::LogicalType::STRUCT(lin_cat))));

  auto struct_type = duckdb::LogicalType::STRUCT(struct_children);
  function.return_type = struct_type;
  //set arguments
  return duckdb::make_uniq<duckdb::VariableReturnBindData>(function.return_type);
}


//Updates the state with a new value. E.g, For SUM, this adds the value to the state
void Triple::sum_nb_agg(duckdb::Vector inputs[], duckdb::AggregateInputData &aggr_input_data, idx_t input_count,
         duckdb::Vector &state_vector, idx_t count) {
  D_ASSERT(input_count == 1);

  duckdb::UnifiedVectorFormat sdata;
  state_vector.ToUnifiedFormat(count, sdata);

  auto states = (SumState **) sdata.data;

  auto &input = inputs[0];
  RecursiveFlatten(input, count);//not sure it's needed
  D_ASSERT(input.GetVectorType() == duckdb::VectorType::FLAT_VECTOR);

  //SUM N
  auto &children = duckdb::StructVector::GetEntries(input);
  duckdb::Vector &c1 = *children[0];//this should be the vector of integers....

  duckdb::UnifiedVectorFormat input_data;
  c1.ToUnifiedFormat(count, input_data);
  auto c1_u = duckdb::UnifiedVectorFormat::GetData<int32_t>(input_data);

  //auto input_data = (int32_t *) FlatVector::GetData(c1);
  //auto state = states[sdata.sel->get_index(0)];//because of parallelism???

  for (int j = 0; j < count; j++) {
    auto state = states[sdata.sel->get_index(j)];
    state->count += c1_u[input_data.sel->get_index(j)];
    state->is_nb_aggregates = true;
  }
  //auto N =  input_data;

  //SUM LINEAR AGGREGATES:

  duckdb::Vector &c2 = *children[1];//this should be the linear aggregate....
  duckdb::UnifiedVectorFormat input_data_c2;
  duckdb::ListVector::GetEntry(c2).ToUnifiedFormat(count, input_data_c2);
  auto list_entries = duckdb::UnifiedVectorFormat::GetData<float>(input_data_c2);

  auto num_attr_size = duckdb::ListVector::GetListSize(c2);
  auto cat_attr_size = duckdb::ListVector::GetListSize(*children[3]);

  //auto list_entries = (float *) ListVector::GetEntry(c2).GetData();//entries are float
  int num_attr = num_attr_size / count;//cols = total values / num. rows
  int cat_attr = cat_attr_size;
  if (cat_attr > 0)
    cat_attr = duckdb::ListVector::GetData(*children[3])[0].length;


  for (idx_t j = 0; j < count; j++) {
    auto state = states[sdata.sel->get_index(j)];

    if (state->lin_agg == nullptr && state->quad_num_cat == nullptr){//initialize
      state->num_attributes = num_attr;
      state->cat_attributes = cat_attr;
      if(num_attr > 0) {
        state->lin_agg = new float[num_attr + num_attr];
        state->quadratic_agg = &(state->lin_agg[num_attr]);
        for(idx_t k=0; k<num_attr; k++)
          state->lin_agg[k] = 0;

        for(idx_t k=0; k<num_attr; k++)
          state->quadratic_agg[k] = 0;
      }
      if(cat_attr > 0) {//todo
        state->quad_num_cat = new std::map<int, std::vector<float>>[cat_attr];//boost::container::flat_map<int, std::vector<float>>[cat_attr];

        state->quad_cat_cat = nullptr;
      }
    }

    for(idx_t k=0; k<num_attr; k++) {
      state->lin_agg[k] += list_entries[input_data_c2.sel->get_index(k + (j * num_attr))];
    }
  }

  //SUM QUADRATIC AGGREGATES:
  duckdb::Vector &c3 = *children[2];//this should be the linear aggregate....
  duckdb::UnifiedVectorFormat input_data_c3;
  duckdb::ListVector::GetEntry(c3).ToUnifiedFormat(count, input_data_c3);
  auto list_entries_2 = duckdb::UnifiedVectorFormat::GetData<float>(input_data_c3);

  int cols2 = num_attr;//quadratic aggregates columns

  for (idx_t j = 0; j < count; j++) {
    auto state = states[sdata.sel->get_index(j)];
    for(idx_t k=0; k<cols2; k++)
      state->quadratic_agg[k] += list_entries_2[input_data_c3.sel->get_index(k + (j*cols2))];
  }

  //sum cat lin. data

  int *cat_set_val_key_lin_cat = nullptr;
  float *cat_set_val_val_lin_cat = nullptr;
  duckdb::list_entry_t *sublist_metadata_lin_cat = nullptr;

  if(cat_attr > 0){
    duckdb::Vector v_cat_lin_1 = duckdb::ListVector::GetEntry(*children[3]);
    sublist_metadata_lin_cat = duckdb::ListVector::GetData(v_cat_lin_1);
    duckdb::Vector v_cat_lin_1_values = duckdb::ListVector::GetEntry(v_cat_lin_1);
    duckdb::vector<duckdb::unique_ptr<duckdb::Vector>> &v_cat_lin_1_struct = duckdb::StructVector::GetEntries(
        v_cat_lin_1_values);
    //todo flatvector assumption
    cat_set_val_key_lin_cat = duckdb::FlatVector::GetData<int>(*((v_cat_lin_1_struct)[0]));
    cat_set_val_val_lin_cat = duckdb::FlatVector::GetData<float>(*((v_cat_lin_1_struct)[1]));

    //todo make this vary num_attr+1
    float sum_vals[1024];//sum of values for key

    for (idx_t j = 0; j < count; j++) {
      auto state = states[sdata.sel->get_index(j)];

      for(idx_t k=0; k<cat_attr; k++){
        auto &col_vals = state->quad_num_cat[k];//get the map
        //they have the same key length for each categorical, so get length from first categorical*numerical
        //auto size_keys_cat_col = sublist_metadata_num_cat[k + (j*(cat_attr * num_attr))];
        auto curr_metadata_lin_cat = sublist_metadata_lin_cat[(j*cat_attr) + k];

        for(int key_idx=0; key_idx<curr_metadata_lin_cat.length; key_idx++){//for each key of the map
          int key = cat_set_val_key_lin_cat[key_idx + curr_metadata_lin_cat.offset];//key value
          //need to sum counts
          sum_vals[0] = cat_set_val_val_lin_cat[curr_metadata_lin_cat.offset + key_idx];
          auto pos = col_vals.find(key);
          if (pos == col_vals.end())
            col_vals[key] = std::vector<float>(sum_vals, sum_vals+1);
          else
              pos->second[0] += sum_vals[0];
        }
      }
    }
  }
}

duckdb::vector<duckdb::Value> sum_list_of_structs(const duckdb::vector<duckdb::Value> &v1, const duckdb::vector<duckdb::Value> &v2){
  std::map<int, float> content;
  duckdb::vector<duckdb::Value> col_cat_lin = {};
  for(size_t k=0; k<v1.size(); k++){
    //extract struct
    auto children = duckdb::StructValue::GetChildren(v1[k]);//vector of pointers to childrens
    content[children[0].GetValue<int>()] = children[1].GetValue<float>();
  }

  for(size_t k=0; k<v2.size(); k++){
    //extract struct
    auto children = duckdb::StructValue::GetChildren(v2[k]);//vector of pointers to childrens
    auto pos = content.find(children[0].GetValue<int>());
    if (pos == content.end())
      content[children[0].GetValue<int>()] = children[1].GetValue<float>();
    else
      pos->second += children[1].GetValue<float>();
  }
  for (auto const& value : content){
    duckdb::child_list_t<duckdb::Value> struct_values;
    struct_values.emplace_back("key", duckdb::Value(value.first));
    struct_values.emplace_back("value", duckdb::Value(value.second));
    col_cat_lin.push_back(duckdb::Value::STRUCT(struct_values));
  }
  return col_cat_lin;
}

duckdb::vector<duckdb::Value> sum_list_of_structs_key2(const duckdb::vector<duckdb::Value> &v1, const duckdb::vector<duckdb::Value> &v2){
  std::map<std::pair<int, int>, float> content;
  duckdb::vector<duckdb::Value> col_cat_lin = {};
  for(size_t k=0; k<v1.size(); k++){
    //extract struct
    auto children = duckdb::StructValue::GetChildren(v1[k]);//vector of pointers to childrens
    content[std::pair<int, int>(children[0].GetValue<int>(), children[1].GetValue<int>())] = children[2].GetValue<float>();
  }

  for(size_t k=0; k<v2.size(); k++){
    //extract struct
    auto children = duckdb::StructValue::GetChildren(v2[k]);//vector of pointers to childrens
    auto key = std::pair<int, int>(children[0].GetValue<int>(), children[1].GetValue<int>());
    auto pos = content.find(key);
    if (pos == content.end())
      content[key] = children[2].GetValue<float>();
    else
      pos->second += children[2].GetValue<float>();
  }
  for (auto const& value : content){
    duckdb::child_list_t<duckdb::Value> struct_values;
    struct_values.emplace_back("key1", duckdb::Value(value.first.first));
    struct_values.emplace_back("key2", duckdb::Value(value.first.second));
    struct_values.emplace_back("value", duckdb::Value(value.second));
    col_cat_lin.push_back(duckdb::Value::STRUCT(struct_values));
  }
  return col_cat_lin;
}

duckdb::Value sum_nb_aggregates(const duckdb::Value &triple_1, const duckdb::Value &triple_2){

  auto first_triple_children = duckdb::StructValue::GetChildren(triple_1);//vector of pointers to childrens
  auto sec_triple_children = duckdb::StructValue::GetChildren(triple_2);

  auto N_1 = first_triple_children[0].GetValue<int>();;
  auto N_2 = sec_triple_children[0].GetValue<int>();

  duckdb::child_list_t<duckdb::Value> struct_values;
  struct_values.emplace_back("N", duckdb::Value(N_1 + N_2));

  const duckdb::vector<duckdb::Value> &linear_1 = duckdb::ListValue::GetChildren(first_triple_children[1]);
  const duckdb::vector<duckdb::Value> &linear_2 = duckdb::ListValue::GetChildren(sec_triple_children[1]);
  duckdb::vector<duckdb::Value> lin = {};
  if (!linear_1.empty() && !linear_2.empty()) {
    for (idx_t i = 0; i < linear_1.size(); i++)
      lin.push_back(duckdb::Value(linear_1[i].GetValue<float>() + linear_2[i].GetValue<float>()));
  }
  else if (!linear_1.empty()){
    for (idx_t i = 0; i < linear_1.size(); i++)
      lin.push_back(duckdb::Value(linear_1[i].GetValue<float>()));
  }
  else if (!linear_2.empty()){
    for (idx_t i = 0; i < linear_2.size(); i++)
      lin.push_back(duckdb::Value(linear_2[i].GetValue<float>()));
  }

  struct_values.emplace_back("lin_num", duckdb::Value::LIST(duckdb::LogicalType::FLOAT, lin));

  const duckdb::vector<duckdb::Value> &quad_1 = duckdb::ListValue::GetChildren(first_triple_children[2]);
  const duckdb::vector<duckdb::Value> &quad_2 = duckdb::ListValue::GetChildren(sec_triple_children[2]);
  duckdb::vector<duckdb::Value> quad = {};

  if (!quad_1.empty() && !quad_2.empty()) {
    for(idx_t i=0;i<quad_1.size();i++)
      quad.push_back(duckdb::Value(quad_1[i].GetValue<float>() + quad_2[i].GetValue<float>()));
  }
  else if (!quad_1.empty()){
    for (idx_t i = 0; i < quad_1.size(); i++)
      quad.push_back(duckdb::Value(quad_1[i].GetValue<float>()));
  }
  else if (!quad_2.empty()){
    for (idx_t i = 0; i < quad_2.size(); i++)
      quad.push_back(duckdb::Value(quad_2[i].GetValue<float>()));
  }

  struct_values.emplace_back("quad_num", duckdb::Value::LIST(duckdb::LogicalType::FLOAT, quad));

  //categorical linear
  const duckdb::vector<duckdb::Value> &cat_linear_1 = duckdb::ListValue::GetChildren(first_triple_children[3]);
  const duckdb::vector<duckdb::Value> &cat_linear_2 = duckdb::ListValue::GetChildren(sec_triple_children[3]);
  duckdb::vector<duckdb::Value> cat_lin = {};
  if (!cat_linear_1.empty() && !cat_linear_2.empty()) {
    for (idx_t i = 0; i < cat_linear_1.size(); i++) {//for each cat. column copy into map
      const duckdb::vector<duckdb::Value> &pairs_cat_col_1 = duckdb::ListValue::GetChildren(cat_linear_1[i]);
      const duckdb::vector<duckdb::Value> &pairs_cat_col_2 = duckdb::ListValue::GetChildren(cat_linear_2[i]);
      cat_lin.push_back(duckdb::Value::LIST(sum_list_of_structs(pairs_cat_col_1, pairs_cat_col_2)));
    }
  }
  else if (!cat_linear_1.empty()){
    for (idx_t i = 0; i < cat_linear_1.size(); i++) {//for each cat. column copy into map
      const duckdb::vector<duckdb::Value> &pairs_cat_col_1 = duckdb::ListValue::GetChildren(cat_linear_1[i]);
      cat_lin.push_back(duckdb::Value::LIST(pairs_cat_col_1));
    }
  }
  else if (!cat_linear_2.empty()){
    for (idx_t i = 0; i < cat_linear_2.size(); i++) {//for each cat. column copy into map
      const duckdb::vector<duckdb::Value> &pairs_cat_col_1 = duckdb::ListValue::GetChildren(cat_linear_2[i]);
      cat_lin.push_back(duckdb::Value::LIST(pairs_cat_col_1));
    }
  }

  duckdb::child_list_t<duckdb::LogicalType> struct_values_l;
  struct_values_l.emplace_back("key", duckdb::LogicalType::INTEGER);
  struct_values_l.emplace_back("value", duckdb::LogicalType::FLOAT);

  struct_values.emplace_back("lin_cat", duckdb::Value::LIST(duckdb::LogicalType::LIST(duckdb::LogicalType::STRUCT(struct_values_l)),cat_lin));
  auto ret = duckdb::Value::STRUCT(struct_values);
  return ret;
}

