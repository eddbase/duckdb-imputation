

#include <triple/lift.h>
#include <duckdb/function/scalar/nested_functions.hpp>
#include <duckdb/planner/expression/bound_function_expression.hpp>

#include <unordered_set>

#include <iostream>
#include <utils.h>


namespace Triple {
//actual implementation of this function
void CustomLift(duckdb::DataChunk &args, duckdb::ExpressionState &state, duckdb::Vector &result) {
  //std::cout << "StructPackFunction start " << std::endl;

  //std::cout<<"B"<<std::endl;

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
  //std::cout<<"Size categorical linear: "<<cat_cols*size<<" size "<<size<<" cat cols "<<cat_cols<<std::endl;

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


  //std::cout<<"cat linear: "<<duckdb::ListVector::GetListSize(*result_children[1])<<"\n";

  //set N*N
  //std::cout<<"N*N"<<std::endl;
  duckdb::ListVector::Reserve(*result_children[2], ((num_cols*(num_cols+1))/2) * size);
  duckdb::ListVector::SetListSize(*result_children[2], ((num_cols*(num_cols+1))/2) * size);
  auto quad_vec = duckdb::FlatVector::GetData<duckdb::list_entry_t>(*result_children[2]);//sec child (lin. aggregates)
  auto quad_vec_data = duckdb::FlatVector::GetData<float>(duckdb::ListVector::GetEntry(*result_children[2]));

  //numerical * numerical
  int col_idx = 0;
  for (duckdb::idx_t j=0;j<columns;j++){
    auto col_type = in_data[j].GetType();
    if (col_type == duckdb::LogicalType::FLOAT || col_type == duckdb::LogicalType::DOUBLE) {
      const float *column_data = duckdb::UnifiedVectorFormat::GetData<float>(input_data[j]);//.GetData();//input column //todo potential problems
      for (duckdb::idx_t k = j; k < columns; k++) {
        auto col_type = in_data[k].GetType();
        if (col_type == duckdb::LogicalType::FLOAT || col_type == duckdb::LogicalType::DOUBLE) {
          const float *sec_column_data = duckdb::UnifiedVectorFormat::GetData<float>(input_data[k]);//numerical * numerical
          for (duckdb::idx_t i = 0; i < size; i++) {
            quad_vec_data[col_idx + (i * num_cols * (num_cols + 1) / 2)] =
                column_data[input_data[j].sel->get_index(i)] * sec_column_data[input_data[k].sel->get_index(i)];
          }
          col_idx++;
        }
      }
    }
  }
  //std::cout<<"num quad: "<<duckdb::ListVector::GetListSize(*result_children[2])<<"\n";


  duckdb::ListVector::Reserve(*result_children[4], (num_cols * cat_cols) * size);
  duckdb::ListVector::SetListSize(*result_children[4], (num_cols * cat_cols) * size);
  auto cat_relations_vector_num_quad_list = duckdb::ListVector::GetData(*result_children[4]);

  if (cat_cols>0 && num_cols > 0) {
    duckdb::Vector &cat_relations_vector_num_quad = duckdb::ListVector::GetEntry(*result_children[4]);
    duckdb::ListVector::Reserve(cat_relations_vector_num_quad, num_cols * cat_cols * size);
    duckdb::ListVector::SetListSize(cat_relations_vector_num_quad, num_cols * cat_cols * size);
    sublist_metadata = duckdb::ListVector::GetData(cat_relations_vector_num_quad);
    duckdb::Vector cat_relations_vector_sub = duckdb::ListVector::GetEntry(cat_relations_vector_num_quad);//this is a sequence of values (struct in our case, 2 vectors)
    duckdb::vector<duckdb::unique_ptr<duckdb::Vector>> &lin_struct_vector = duckdb::StructVector::GetEntries(cat_relations_vector_sub);
    cat_set_val_key = duckdb::FlatVector::GetData<int>(*((lin_struct_vector)[0]));
    cat_set_val_val = duckdb::FlatVector::GetData<float>(*((lin_struct_vector)[1]));
  }

  //num * categorical
  col_idx = 0;
  for (duckdb::idx_t j=0;j<columns;j++){
    auto col_type = in_data[j].GetType();
    if (col_type == duckdb::LogicalType::FLOAT || col_type == duckdb::LogicalType::DOUBLE) {
      //numerical column
      const float *num_column_data = duckdb::UnifiedVectorFormat::GetData<float>(input_data[j]);//.GetData();//col1
      for (duckdb::idx_t k = 0; k < columns; k++) {
        auto col_type = in_data[k].GetType();
        if (col_type != duckdb::LogicalType::FLOAT && col_type != duckdb::LogicalType::DOUBLE) {//numerical * categorical
          const int *cat_column_data = duckdb::UnifiedVectorFormat::GetData<int>(input_data[k]);//.GetData();//categorical
          for (duckdb::idx_t i = 0; i < size; i++) {
            cat_set_val_key[col_idx + (i * (cat_cols * num_cols))] = cat_column_data[input_data[k].sel->get_index(i)];
            cat_set_val_val[col_idx + (i * (cat_cols * num_cols))] = num_column_data[input_data[j].sel->get_index(i)];
            sublist_metadata[col_idx + (i * (cat_cols * num_cols))].length = 1;
            sublist_metadata[col_idx + (i * (cat_cols * num_cols))].offset = col_idx + (i * (cat_cols * num_cols));
          }
          col_idx++;
        }
      }
    }
  }
  //std::cout<<"num*cat: "<<duckdb::ListVector::GetListSize(*result_children[4])<<"\n";

  //categorical * categorical

  duckdb::ListVector::Reserve(*result_children[5], ((cat_cols*(cat_cols+1))/2) * size);
  duckdb::ListVector::SetListSize(*result_children[5], ((cat_cols*(cat_cols+1))/2) * size);
  auto cat_relations_vector_cat_quad_list = duckdb::ListVector::GetData(*result_children[5]);
  int *cat_set_val_key_1 = nullptr;
  int *cat_set_val_key_2 = nullptr;

  if (cat_cols > 0) {
    duckdb::Vector &cat_relations_vector_cat_quad = duckdb::ListVector::GetEntry(*result_children[5]);
    duckdb::ListVector::Reserve(cat_relations_vector_cat_quad, ((cat_cols*(cat_cols+1))/2) * size);
    duckdb::ListVector::SetListSize(cat_relations_vector_cat_quad, ((cat_cols*(cat_cols+1))/2) * size);
    sublist_metadata = duckdb::ListVector::GetData(cat_relations_vector_cat_quad);
    duckdb::Vector cat_relations_vector_sub = duckdb::ListVector::GetEntry(cat_relations_vector_cat_quad);//this is a sequence of values (struct in our case, 2 vectors)
    duckdb::vector<duckdb::unique_ptr<duckdb::Vector>> &lin_struct_vector = duckdb::StructVector::GetEntries(cat_relations_vector_sub);
    cat_set_val_key_1 = duckdb::FlatVector::GetData<int>(*((lin_struct_vector)[0]));
    cat_set_val_key_2 = duckdb::FlatVector::GetData<int>(*((lin_struct_vector)[1]));
    cat_set_val_val = duckdb::FlatVector::GetData<float>(*((lin_struct_vector)[2]));
  }

  col_idx = 0;
  for (duckdb::idx_t j=0;j<columns;j++){
    auto col_type = in_data[j].GetType();
    if (col_type != duckdb::LogicalType::FLOAT && col_type != duckdb::LogicalType::DOUBLE) {
      const int *cat_1 = duckdb::UnifiedVectorFormat::GetData<int>(input_data[j]);//.GetData();//col1
      for (duckdb::idx_t k = j; k < columns; k++) {
        auto col_type = in_data[k].GetType();
        if (col_type != duckdb::LogicalType::FLOAT && col_type != duckdb::LogicalType::DOUBLE) {//categorical * categorical
          const int *cat_2 = duckdb::UnifiedVectorFormat::GetData<int> (input_data[k]); //.GetData();//categorical
          for (duckdb::idx_t i = 0; i < size; i++) {
            cat_set_val_key_1[col_idx + (i * ((cat_cols * (cat_cols + 1)) / 2))] = cat_1[input_data[j].sel->get_index(i)];
            cat_set_val_key_2[col_idx + (i * ((cat_cols * (cat_cols + 1)) / 2))] = cat_2[input_data[k].sel->get_index(i)];
            cat_set_val_val[col_idx + (i * ((cat_cols * (cat_cols + 1)) / 2))] = 1;
            sublist_metadata[col_idx + (i * ((cat_cols * (cat_cols + 1)) / 2))].length = 1;
            sublist_metadata[col_idx + (i * ((cat_cols * (cat_cols + 1)) / 2))].offset = col_idx + (i * ((cat_cols * (cat_cols + 1)) / 2));
          }
          col_idx++;
        }
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

    quad_vec[i].length = (num_cols*(num_cols+1))/2;
    quad_vec[i].offset = i * quad_vec[i].length;

    cat_relations_vector_num_quad_list[i].length = num_cols*cat_cols;
    cat_relations_vector_num_quad_list[i].offset = i * cat_relations_vector_num_quad_list[i].length;

    cat_relations_vector_cat_quad_list[i].length = cat_cols*(cat_cols+1)/2;
    cat_relations_vector_cat_quad_list[i].offset = i * cat_relations_vector_cat_quad_list[i].length;

  }
  //set categorical
}

//Returns the datatype used by this function
duckdb::unique_ptr<duckdb::FunctionData>
CustomLiftBind(duckdb::ClientContext &context, duckdb::ScalarFunction &function,
               duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> &arguments) {

  //set return type

  //std::cout<<"a"<<std::endl;

  duckdb::child_list_t<duckdb::LogicalType> struct_children;
  struct_children.emplace_back("N", duckdb::LogicalType::INTEGER);
  struct_children.emplace_back("lin_num", duckdb::LogicalType::LIST(duckdb::LogicalType::FLOAT));
  struct_children.emplace_back("quad_num", duckdb::LogicalType::LIST(duckdb::LogicalType::FLOAT));

  //categorical structures
  duckdb::child_list_t<duckdb::LogicalType> lin_cat;
  lin_cat.emplace_back("key",duckdb:: LogicalType::INTEGER);
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
  //lin_cat -> LIST(LIST(STRUCT(key1, key2, val))). E.g. [[{k1,k2,2},{k3,k4,5}],[]]...
  //quad_cat

  auto struct_type = duckdb::LogicalType::STRUCT(struct_children);
  function.return_type = struct_type;
  //set arguments
  function.varargs = duckdb::LogicalType::ANY;
  return duckdb::make_uniq<duckdb::VariableReturnBindData>(function.return_type);
}

//Generate statistics for this function. Given input type ststistics (mainly min and max for every attribute), returns the output statistics
duckdb::unique_ptr<duckdb::BaseStatistics>
CustomLiftStats(duckdb::ClientContext &context, duckdb::FunctionStatisticsInput &input) {
  auto &child_stats = input.child_stats;
  auto &expr = input.expr;
  auto struct_stats = duckdb::StructStats::CreateUnknown(expr.return_type);
  return struct_stats.ToUnique();
}
}
