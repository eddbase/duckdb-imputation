

#include <triple/sum/sum_no_lift.h>
#include <duckdb/function/scalar/nested_functions.hpp>
#include <triple/sum/sum_state.h>

#include <triple/sum/sum.h>

#include <iostream>
#include <map>
#include <unordered_map>


duckdb::unique_ptr<duckdb::FunctionData>
Triple::SumNoLiftBind(duckdb::ClientContext &context, duckdb::AggregateFunction &function,
                      duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> &arguments) {

  //std::cout<<arguments.size()<<"\n";
  //std::cout<<(function.arguments.size())<<"\n";

  duckdb::child_list_t<duckdb::LogicalType> struct_children;
  struct_children.emplace_back("N", duckdb::LogicalType::INTEGER);
  struct_children.emplace_back("lin_agg", duckdb::LogicalType::LIST(duckdb::LogicalType::FLOAT));
  struct_children.emplace_back("quad_agg", duckdb::LogicalType::LIST(duckdb::LogicalType::FLOAT));

  //categorical structures
  duckdb::child_list_t<duckdb::LogicalType> lin_cat;
  lin_cat.emplace_back("key", duckdb::LogicalType::INTEGER);
  lin_cat.emplace_back("value", duckdb::LogicalType::FLOAT);

  struct_children.emplace_back("lin_cat", duckdb::LogicalType::LIST(duckdb::LogicalType::LIST(duckdb::LogicalType::STRUCT(lin_cat))));

  duckdb::child_list_t<duckdb::LogicalType> quad_num_cat;
  quad_num_cat.emplace_back("key", duckdb::LogicalType::INTEGER);
  quad_num_cat.emplace_back("value", duckdb::LogicalType::FLOAT);
  struct_children.emplace_back("quad_num_cat", duckdb::LogicalType::LIST(duckdb::LogicalType::LIST(duckdb::LogicalType::STRUCT(quad_num_cat))));

  duckdb::child_list_t<duckdb::LogicalType> quad_cat_cat;
  quad_cat_cat.emplace_back("key1", duckdb::LogicalType::INTEGER);
  quad_cat_cat.emplace_back("key2", duckdb::LogicalType::INTEGER);
  quad_cat_cat.emplace_back("value", duckdb::LogicalType::FLOAT);
  struct_children.emplace_back("quad_cat", duckdb::LogicalType::LIST(duckdb::LogicalType::LIST(duckdb::LogicalType::STRUCT(quad_cat_cat))));

  auto struct_type = duckdb::LogicalType::STRUCT(struct_children);
  function.return_type = struct_type;
  //set arguments
  //function.varargs = LogicalType::ANY;
  return duckdb::make_uniq<duckdb::VariableReturnBindData>(function.return_type);
}


//Updates the state with a new value. E.g, For SUM, this adds the value to the state. Input count: n. of columns
void Triple::SumNoLift(duckdb::Vector inputs[], duckdb::AggregateInputData &aggr_input_data, duckdb::idx_t cols,
                       duckdb::Vector &state_vector, duckdb::idx_t count) {
  //count: count of tuples to process
  //inputs: inputs to process

  duckdb::UnifiedVectorFormat sdata;
  state_vector.ToUnifiedFormat(count, sdata);

  auto states = (Triple::SumState **) sdata.data;

  size_t num_cols = 0;
  size_t cat_cols = 0;
  duckdb::UnifiedVectorFormat input_data[1024];//todo vary size (cols)
  for (idx_t j=0;j<cols;j++){
    auto col_type = inputs[j].GetType();
    inputs[j].ToUnifiedFormat(count, input_data[j]);
    if (col_type == duckdb::LogicalType::FLOAT || col_type == duckdb::LogicalType::DOUBLE)
      num_cols++;
    else
      cat_cols++;
  }

  //auto &input = inputs[1];
  //RecursiveFlatten(input, count);
  //std::cout<<FlatVector::

  //auto value = (float *)input.GetData();
  //SUM N
  //auto state = states[sdata.sel->get_index(0)];//because of parallelism???

  for (int j = 0; j < count; j++) {
    auto state = states[sdata.sel->get_index(j)];
    state->count += 1;
  }

  //SUM LINEAR AGGREGATES:
  //std::cout<<"Start SUMNOLIFT";

  for (idx_t j = 0; j < count; j++) {
    size_t curr_numerical = 0;
    size_t curr_cateogrical = 0;
    auto state = states[sdata.sel->get_index(j)];
    //initialize
    if (state->lin_agg == nullptr && state->quad_num_cat == nullptr){
      state->num_attributes = num_cols;
      state->cat_attributes = cat_cols;

      if(num_cols > 0) {
        state->lin_agg = new float[num_cols + ((num_cols * (num_cols + 1)) / 2)];
        state->quadratic_agg = &(state->lin_agg[num_cols]);
        for (idx_t k = 0; k < num_cols; k++)
          state->lin_agg[k] = 0;
        for (idx_t k = 0; k < (num_cols * (num_cols + 1)) / 2; k++)
          state->quadratic_agg[k] = 0;
      }

      if(cat_cols > 0) {
        state->quad_num_cat = new std::map<int, std::vector<float>>[cat_cols];//new boost::container::flat_map<int, std::vector<float>>[cat_cols];
        //todo
        //for(int kk=0; kk<cat_cols; kk++)
        //  state->quad_num_cat[kk].reserve(4);
        state->quad_cat_cat = new std::map<std::pair<int, int>, float>[cat_cols * (cat_cols + 1) / 2];//boost::container::flat_map<std::pair<int, int>, float>[cat_cols * (cat_cols + 1) / 2];
      }
    }


    for(idx_t k=0; k<num_cols; k++) {
      state->lin_agg[k] += duckdb::UnifiedVectorFormat::GetData<float>(input_data[k])[input_data[k].sel->get_index(j)];
    }
  }

  //std::cout<<"End INIT + lin NEW"<<std::endl;


  //SUM QUADRATIC NUMERICAL AGGREGATES:
  int col_idx = 0;

  for(idx_t j=0; j<cols; j++) {
    auto col_type = inputs[j].GetType();
    if (col_type == duckdb::LogicalType::FLOAT || col_type == duckdb::LogicalType::DOUBLE) {//numericals
      const float *first_column_data = duckdb::UnifiedVectorFormat::GetData<float>(input_data[j]);
      for (idx_t k = j; k < cols; k++) {
        auto col_type = inputs[k].GetType();
        if (col_type == duckdb::LogicalType::FLOAT || col_type == duckdb::LogicalType::DOUBLE) {//numerical
          const float *sec_column_data = duckdb::UnifiedVectorFormat::GetData<float>(input_data[k]);
          for (idx_t i = 0; i < count; i++) {
            auto state = states[sdata.sel->get_index(i)];
            state->quadratic_agg[col_idx] += first_column_data[input_data[j].sel->get_index(i)] * sec_column_data[input_data[k].sel->get_index(i)];
          }
          col_idx++;
        }
      }
    }
  }
  //std::cout<<"Start NUMCAT"<<std::endl;

  //count unique keys in each categorical
  //and allocate appropriate float vector



  //SUM NUM*CAT

  //NUM*CAT
  float payload[1024];//todo make this vary, num_cols+1
  if (cat_cols > 0) {
    for (idx_t i = 0; i < count; i++) {
      col_idx = 0;
      auto state = states[sdata.sel->get_index(i)];
      for (idx_t j = num_cols; j < cols; j++) {
        const int *sec_column_data = duckdb::UnifiedVectorFormat::GetData<int>(input_data[j]);
        //boost::container::flat_map<int, std::vector<float>> &vals
        std::map<int, std::vector<float>> &vals = state->quad_num_cat[col_idx];//map of the column
        int key = sec_column_data[input_data[j].sel->get_index(i)];
        auto pos = vals.find(key);
        if (pos == vals.end()) {

          //add num_cat columns
          for(idx_t k = 0; k<num_cols; k++) {
            const float *first_column_data = duckdb::UnifiedVectorFormat::GetData<float>(input_data[k]);
            payload[k+1] = first_column_data[input_data[k].sel->get_index(i)];
          }
          payload[0] = 1;
          vals[key] = std::vector<float>(payload, payload+num_cols+1);
        }
        else {
          auto &payload = pos->second;
          for(idx_t k = 0; k<num_cols; k++) {
            const float *first_column_data = duckdb::UnifiedVectorFormat::GetData<float>(input_data[k]);
            payload[k+1] += first_column_data[input_data[k].sel->get_index(i)];
          }
          payload[0]+=1;
        }
        col_idx++;
      }
    }
  }


  //std::cout<<"End NUMCAT"<<std::endl;

  //SUM CAT*CAT
  col_idx = 0;
  for(idx_t j=num_cols; j<cols; j++) {
    const int *first_column_data = duckdb::UnifiedVectorFormat::GetData<int>(input_data[j]);
    for (idx_t k = j; k < cols; k++) {
      const int *sec_column_data = duckdb::UnifiedVectorFormat::GetData<int>(input_data[k]);
      for (idx_t i = 0; i < count; i++) {
        auto state = states[sdata.sel->get_index(i)];
        auto &vals = state->quad_cat_cat[col_idx];
        auto entry = std::pair<int, int>(first_column_data[input_data[j].sel->get_index(i)], sec_column_data[input_data[k].sel->get_index(i)]);
        auto pos = vals.find(entry);
        if (pos == vals.end()) {
          vals[entry] = 1;
        }
        else {
          pos->second += 1;
        }
      }
      col_idx++;
    }
  }
  //std::cout<<"END SUMNOLIFT"<<std::endl;
}
