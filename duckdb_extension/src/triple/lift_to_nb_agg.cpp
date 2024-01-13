

#include <duckdb/function/scalar/nested_functions.hpp>
#include <duckdb/planner/expression/bound_function_expression.hpp>

#include <unordered_set>
#include <utils.h>

#include <triple/lift_to_nb_agg.h>

namespace Triple {
//actual implementation of this function
void to_nb_lift(duckdb::DataChunk &args, duckdb::ExpressionState &state, duckdb::Vector &result) {
  auto &result_children = duckdb::StructVector::GetEntries(result);
  idx_t size = args.size();//n. of rows to return

  duckdb::vector<duckdb::Vector> &in_data = args.data;
  int columns = in_data.size();

  size_t num_cols = 0;
  size_t cat_cols = 0;

  duckdb::UnifiedVectorFormat input_data[1024];//todo make this vary, columns

  for (duckdb::idx_t j=0;j<columns;j++){
    auto col_type = in_data[j].GetType();
    in_data[j].ToUnifiedFormat(size, input_data[j]);
    if (col_type == duckdb::LogicalType::FLOAT || col_type == duckdb::LogicalType::DOUBLE)
      num_cols++;
    else
      cat_cols++;
  }

  //set N
  result_children[0]->SetVectorType(duckdb::VectorType::FLAT_VECTOR);
  auto N_vec = duckdb::FlatVector::GetData<int32_t>(*result_children[0]);//first child (N)

  for (duckdb::idx_t i = 0; i < size; i++) {
    N_vec[i] = 1;
  }

  //set linear

  //get result structs for numerical
  duckdb::ListVector::Reserve(*result_children[1], num_cols*size);
  duckdb::ListVector::SetListSize(*result_children[1], num_cols*size);
  auto lin_vec_num = duckdb::FlatVector::GetData<duckdb::list_entry_t>(*result_children[1]);//sec child (lin. aggregates)
  auto lin_vec_num_data = duckdb::FlatVector::GetData<float>(duckdb::ListVector::GetEntry(*result_children[1]));

  //get result struct for categorical
  duckdb::ListVector::Reserve(*result_children[3], cat_cols*size);
  duckdb::ListVector::SetListSize(*result_children[3], cat_cols*size);

  auto lin_vec_cat = duckdb::FlatVector::GetData<duckdb::list_entry_t>(*result_children[3]);

  //Gets a reference to the underlying child-vector of a list
  //as this is a list of list, another GetEntry needs to be called on cat_relations_vector

  duckdb::list_entry_t *sublist_metadata = nullptr;
  int *cat_set_val_key = nullptr;
  float *cat_set_val_val = nullptr;

  if (cat_cols>0) {
    duckdb::Vector cat_relations_vector = duckdb::ListVector::GetEntry(*result_children[3]);
    duckdb::ListVector::Reserve(cat_relations_vector, size*cat_cols);
    duckdb::ListVector::SetListSize(cat_relations_vector, size*cat_cols);
    sublist_metadata = duckdb::ListVector::GetData(cat_relations_vector);
    duckdb::Vector cat_relations_vector_sub = duckdb::ListVector::GetEntry(cat_relations_vector);//this is a sequence of values (struct in our case, 2 vectors)
    duckdb::vector<duckdb::unique_ptr<duckdb::Vector>> &lin_struct_vector = duckdb::StructVector::GetEntries(cat_relations_vector_sub);
    cat_set_val_key = duckdb::FlatVector::GetData<int>(*((lin_struct_vector)[0]));
    cat_set_val_val = duckdb::FlatVector::GetData<float>(*((lin_struct_vector)[1]));
  }

  size_t curr_numerical = 0;
  size_t curr_categorical = 0;

  for (duckdb::idx_t j=0;j<columns;j++){
    auto col_type = in_data[j].GetType();
    if (col_type == duckdb::LogicalType::FLOAT || col_type ==duckdb::LogicalType::DOUBLE) {
      const float *column_data = duckdb::UnifiedVectorFormat::GetData<float>(input_data[j]);// input_data[j].GetData();//input column
      for (duckdb::idx_t i = 0; i < size; i++) {
        lin_vec_num_data[curr_numerical + (i * num_cols)] = column_data[input_data[j].sel->get_index(i)];
      }
      curr_numerical++;
    }
    else{//empty relations
      assert(cat_cols>0);
      const int *column_data = duckdb::UnifiedVectorFormat::GetData<int>(input_data[j]);// (int *)in_data[j].GetData();//input column
      for (duckdb::idx_t i = 0; i < size; i++) {
        cat_set_val_key[curr_categorical + (i * cat_cols)] = column_data[input_data[j].sel->get_index(i)];
        cat_set_val_val[curr_categorical + (i * cat_cols)] = 1;
        sublist_metadata[(i * cat_cols) + curr_categorical].length = 1;
        sublist_metadata[(i * cat_cols) + curr_categorical].offset = (i * cat_cols) + curr_categorical;
      }
      //duckdb::ListVector::PushBack(*result_children[3], duckdb::Value::LIST(cat_vals));
      curr_categorical++;
    }
  }


  //set N*N
  //NB needs only the diagonal;
  duckdb::ListVector::Reserve(*result_children[2], (num_cols) * size);
  duckdb::ListVector::SetListSize(*result_children[2], (num_cols) * size);
  auto quad_vec = duckdb::FlatVector::GetData<duckdb::list_entry_t>(*result_children[2]);//sec child (lin. aggregates)
  auto quad_vec_data = duckdb::FlatVector::GetData<float>(duckdb::ListVector::GetEntry(*result_children[2]));

  //numerical * numerical
  int col_idx = 0;
  for (duckdb::idx_t j=0;j<columns;j++){
    auto col_type = in_data[j].GetType();
    if (col_type == duckdb::LogicalType::FLOAT || col_type == duckdb::LogicalType::DOUBLE) {
      const float *column_data = duckdb::UnifiedVectorFormat::GetData<float>(input_data[j]);//.GetData();//input column //todo potential problems
      for (duckdb::idx_t i = 0; i < size; i++) {
        quad_vec_data[j + (i * num_cols)] = column_data[input_data[j].sel->get_index(i)] * column_data[input_data[j].sel->get_index(i)];
      }
    }
  }


  //std::cout<<"cat*cat: "<<duckdb::ListVector::GetListSize(*result_children[5])<<"\n";

  //std::cout<<"set views"<<std::endl;
  //set views
  for(int i=0;i<size;i++) {
    lin_vec_num[i].length = num_cols;
    lin_vec_num[i].offset = i * lin_vec_num[i].length;

    lin_vec_cat[i].length = cat_cols;
    lin_vec_cat[i].offset = i * lin_vec_cat[i].length;

    quad_vec[i].length = num_cols;
    quad_vec[i].offset = i * quad_vec[i].length;
  }
  //set categorical
}

//Returns the datatype used by this function
duckdb::unique_ptr<duckdb::FunctionData>
to_nb_lift_bind(duckdb::ClientContext &context, duckdb::ScalarFunction &function,
               duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> &arguments) {

  //set return type

  duckdb::child_list_t<duckdb::LogicalType> struct_children;
  struct_children.emplace_back("N", duckdb::LogicalType::INTEGER);
  struct_children.emplace_back("lin_num", duckdb::LogicalType::LIST(duckdb::LogicalType::FLOAT));
  struct_children.emplace_back("quad_num", duckdb::LogicalType::LIST(duckdb::LogicalType::FLOAT));

  //categorical structures
  duckdb::child_list_t<duckdb::LogicalType> lin_cat;
  lin_cat.emplace_back("key",duckdb:: LogicalType::INTEGER);
  lin_cat.emplace_back("value", duckdb::LogicalType::FLOAT);

  struct_children.emplace_back("lin_cat", duckdb::LogicalType::LIST(duckdb::LogicalType::LIST(duckdb::LogicalType::STRUCT(lin_cat))));

  auto struct_type = duckdb::LogicalType::STRUCT(struct_children);
  function.return_type = struct_type;
  //set arguments
  function.varargs = duckdb::LogicalType::ANY;
  return duckdb::make_uniq<duckdb::VariableReturnBindData>(function.return_type);
}
}
