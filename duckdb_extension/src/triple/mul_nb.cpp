

#include <triple/mul.h>
#include <duckdb/function/scalar/nested_functions.hpp>
#include <duckdb/planner/expression/bound_function_expression.hpp>
#include <duckdb/storage/statistics/list_stats.hpp>

#include <map>
#include <iostream>
#include <algorithm>
#include <utils.h>
#include <triple/mul_nb.h>

//DataChunk is a set of vectors




//actual implementation of this function
void Triple::multiply_nb(duckdb::DataChunk &args, duckdb::ExpressionState &state, duckdb::Vector &result) {
  //std::cout << "StructPackFunction start " << std::endl;
  auto &result_children = duckdb::StructVector::GetEntries(result);

  duckdb::idx_t size = args.size();//n. of rows to return

  //Flatten is required at the moment because join might not materialize a flatten vector
  //In the future we can remove it and use UnifiedVectorFormat instead
  RecursiveFlatten(args.data[0], size);
  RecursiveFlatten(args.data[1], size);

  auto &first_triple_children = duckdb::StructVector::GetEntries(args.data[0]);//vector of pointers to childrens
  auto &sec_triple_children = duckdb::StructVector::GetEntries(args.data[1]);

  //set N

  RecursiveFlatten(*first_triple_children[0], size);
  D_ASSERT((*first_triple_children[0]).GetVectorType() == duckdb::VectorType::FLAT_VECTOR);
  RecursiveFlatten(*sec_triple_children[0], size);
  D_ASSERT((*sec_triple_children[0]).GetVectorType() == duckdb::VectorType::FLAT_VECTOR);

  duckdb::UnifiedVectorFormat N_data[2];
  first_triple_children[0]->ToUnifiedFormat(size, N_data[0]);
  sec_triple_children[0]->ToUnifiedFormat(size, N_data[1]);

  auto input_data = duckdb::FlatVector::GetData<int32_t>(*result_children[0]);

  //set N
  for (duckdb::idx_t i = 0; i < size; i++) {
    input_data[i] = duckdb::UnifiedVectorFormat::GetData<int32_t>(N_data[0])[N_data[0].sel->get_index(i)] *
                    duckdb::UnifiedVectorFormat::GetData<int32_t>(N_data[1])[N_data[1].sel->get_index(i)];
  }

  //set linear aggregates
  RecursiveFlatten(*first_triple_children[1], size);
  D_ASSERT((*first_triple_children[1]).GetVectorType() == duckdb::VectorType::FLAT_VECTOR);
  RecursiveFlatten(*sec_triple_children[1], size);
  D_ASSERT((*sec_triple_children[1]).GetVectorType() == duckdb::VectorType::FLAT_VECTOR);

  auto num_attr_size_1 = duckdb::ListVector::GetListSize(*first_triple_children[1]) / size;
  auto num_attr_size_2 = duckdb::ListVector::GetListSize(*sec_triple_children[1]) / size;

  auto cat_attr_size_1 = duckdb::ListVector::GetListSize(*first_triple_children[3]);
  auto cat_attr_size_2 = duckdb::ListVector::GetListSize(*sec_triple_children[3]);
  //first_triple_children[3]->Print();
  if (cat_attr_size_1 > 0)
    cat_attr_size_1 = duckdb::ListVector::GetData(*first_triple_children[3])[0].length;
  //cat_attr_size_1 = duckdb::ListValue::GetChildren(duckdb::ListVector::GetEntry(*first_triple_children[3]).GetValue(0)).size();
  if (cat_attr_size_2 > 0)
    cat_attr_size_2 = duckdb::ListVector::GetData(*sec_triple_children[3])[0].length;

  duckdb::UnifiedVectorFormat lin_data[2];
  duckdb::ListVector::GetEntry(
      *first_triple_children[1]).ToUnifiedFormat(size, lin_data[0]);
  duckdb::ListVector::GetEntry(
      *sec_triple_children[1]).ToUnifiedFormat(size, lin_data[1]);

  auto lin_list_entries_1 = duckdb::UnifiedVectorFormat::GetData<float>(lin_data[0]);//entries are float
  auto lin_list_entries_2 = duckdb::UnifiedVectorFormat::GetData<float>(lin_data[1]);//entries are float


  (*result_children[1]).SetVectorType(duckdb::VectorType::FLAT_VECTOR);
  auto meta_lin_num = duckdb::ListVector::GetData(*result_children[1]);
  auto meta_lin_cat = duckdb::ListVector::GetData(*result_children[3]);

  duckdb::ListVector::Reserve(*result_children[1], (num_attr_size_1 + num_attr_size_2)*size);
  duckdb::ListVector::SetListSize(*result_children[1], (num_attr_size_1 + num_attr_size_2)*size);
  auto lin_res_num = duckdb::FlatVector::GetData<float>(duckdb::ListVector::GetEntry(*result_children[1]));

  //creates a single vector
  //set lin. num.
  for (duckdb::idx_t i = 0; i < size; i++) {
    //add first list for the tuple
    auto N_1 = duckdb::UnifiedVectorFormat::GetData<int32_t>(N_data[0])[N_data[0].sel->get_index(i)];
    auto N_2 = duckdb::UnifiedVectorFormat::GetData<int32_t>(N_data[1])[N_data[1].sel->get_index(i)];

    for (duckdb::idx_t j = 0; j < num_attr_size_1; j++)
      lin_res_num[(i*(num_attr_size_1 + num_attr_size_2)) + j] = lin_list_entries_1[lin_data[0].sel->get_index(j + (i * num_attr_size_1))] * N_2;

    for (duckdb::idx_t j = 0; j < num_attr_size_2; j++)
      lin_res_num[(i*(num_attr_size_1 + num_attr_size_2)) + j + num_attr_size_1] = lin_list_entries_2[lin_data[1].sel->get_index(j + (i * num_attr_size_2))] * N_1;
  }

  //lin. cat.
  //auto xx = *first_triple_children[3]
  duckdb::Vector v_cat_lin_1 = duckdb::ListVector::GetEntry(*first_triple_children[3]);
  duckdb::Vector v_cat_lin_2 = duckdb::ListVector::GetEntry(*sec_triple_children[3]);
  duckdb::ListVector::Reserve(*result_children[3], (cat_attr_size_1 + cat_attr_size_2) * size);//1241
  duckdb::ListVector::SetListSize(*result_children[3], (cat_attr_size_1 + cat_attr_size_2) * size);
  //std::cout<<"Set res lin cat list to "<< (cat_attr_size_1 + cat_attr_size_2) * size<<std::endl;
  //result_children[3]->SetVectorType(VectorType::FLAT_VECTOR);
  duckdb::Vector &v_cat_lin_res = duckdb::ListVector::GetEntry(*result_children[3]);
  duckdb::list_entry_t *sublist_metadata = nullptr;
  int *cat_set_val_key = nullptr;
  float *cat_set_val_val = nullptr;

  int *cat_attr_1_val_key = nullptr;
  float *cat_attr_1_val_val = nullptr;
  duckdb::list_entry_t *v_1_sublist_metadata = nullptr;
  const int *cat_attr_2_val_key = nullptr;
  const float *cat_attr_2_val_val = nullptr;
  duckdb::list_entry_t *v_2_sublist_metadata = nullptr;
  duckdb::UnifiedVectorFormat v_tmp[2];

  if (cat_attr_size_1 >0 || cat_attr_size_2 >0) {
    //init linear categorical
    size_t total_values = 0;
    if (cat_attr_size_1 > 0){
      auto sublist_meta = duckdb::ListVector::GetData(duckdb::ListVector::GetEntry(*first_triple_children[3]));
      auto main_list_len = duckdb::ListVector::GetListSize(*first_triple_children[3]);
      total_values += sublist_meta[main_list_len-1].offset + sublist_meta[main_list_len-1].length;
    }
    if (cat_attr_size_2 > 0){
      auto sublist_meta = duckdb::ListVector::GetData(duckdb::ListVector::GetEntry(*sec_triple_children[3]));
      auto main_list_len = duckdb::ListVector::GetListSize(*sec_triple_children[3]);
      total_values += sublist_meta[main_list_len-1].offset + sublist_meta[main_list_len-1].length;
      //std::cout<<sublist_meta[main_list_len-1].offset<<" "<<sublist_meta[main_list_len-1].length<<std::endl;
    }

    duckdb::Vector cat_relations_vector = duckdb::ListVector::GetEntry(*result_children[3]);
    duckdb::ListVector::Reserve(cat_relations_vector, total_values);
    duckdb::ListVector::SetListSize(cat_relations_vector, total_values);
    sublist_metadata = duckdb::ListVector::GetData(cat_relations_vector);
    duckdb::Vector cat_relations_vector_sub = duckdb::ListVector::GetEntry(cat_relations_vector);
    duckdb::vector<duckdb::unique_ptr<duckdb::Vector>> &lin_struct_vector = duckdb::StructVector::GetEntries(cat_relations_vector_sub);
    cat_set_val_key = duckdb::FlatVector::GetData<int>(*((lin_struct_vector)[0]));
    cat_set_val_val = duckdb::FlatVector::GetData<float>(*((lin_struct_vector)[1]));


    if (cat_attr_size_1 > 0) {
      duckdb::Vector v_cat_lin_1 = duckdb::ListVector::GetEntry(*first_triple_children[3]);
      v_1_sublist_metadata = duckdb::ListVector::GetData(v_cat_lin_1);
      duckdb::Vector v_cat_lin_1_values = duckdb::ListVector::GetEntry(v_cat_lin_1);
      duckdb::vector<duckdb::unique_ptr<duckdb::Vector>> &v_cat_lin_1_struct = duckdb::StructVector::GetEntries(
          v_cat_lin_1_values);
      cat_attr_1_val_key = duckdb::FlatVector::GetData<int>(*((v_cat_lin_1_struct)[0]));
      cat_attr_1_val_val = duckdb::FlatVector::GetData<float>(*((v_cat_lin_1_struct)[1]));
    }
    if (cat_attr_size_2 > 0) {
      duckdb::Vector v_cat_lin_1 = duckdb::ListVector::GetEntry(*sec_triple_children[3]);
      v_2_sublist_metadata = duckdb::ListVector::GetData(v_cat_lin_1);
      duckdb::Vector v_cat_lin_1_values = duckdb::ListVector::GetEntry(v_cat_lin_1);
      duckdb::vector<duckdb::unique_ptr<duckdb::Vector>> &v_cat_lin_1_struct = duckdb::StructVector::GetEntries(
          v_cat_lin_1_values);

      (v_cat_lin_1_struct[0])->ToUnifiedFormat(size, v_tmp[0]);
      (v_cat_lin_1_struct[1])->ToUnifiedFormat(size, v_tmp[1]);

      cat_attr_2_val_key = duckdb::UnifiedVectorFormat::GetData<int32_t>(v_tmp[0]);
      cat_attr_2_val_val = duckdb::UnifiedVectorFormat::GetData<float>(v_tmp[1]);

      //cat_attr_2_val_key = duckdb::FlatVector::GetData<int>(*((v_cat_lin_1_struct)[0]));
      //cat_attr_2_val_val = duckdb::FlatVector::GetData<float>(*((v_cat_lin_1_struct)[1]));
    }
  }

  size_t item_count = 0;
  size_t sublist_metadata_count = 0;
  size_t skipped = 0;
  //set linear categorical
  for (duckdb::idx_t i = 0; i < size; i++) {
    //add first linears
    auto N_1 = duckdb::UnifiedVectorFormat::GetData<int32_t>(N_data[0])[N_data[0].sel->get_index(i)];
    auto N_2 = duckdb::UnifiedVectorFormat::GetData<int32_t>(N_data[1])[N_data[1].sel->get_index(i)];
    //std::cout<<"N1 "<<N_1<<" N2 "<<N_2<<"\n";
    for (duckdb::idx_t column = 0; column < cat_attr_size_1; column++) {
      auto curr_metadata = v_1_sublist_metadata[column + (i*cat_attr_size_1)];
      for (duckdb::idx_t item = 0; item < curr_metadata.length; item++) {
        cat_set_val_key[item_count] = cat_attr_1_val_key[item + curr_metadata.offset];
        cat_set_val_val[item_count] = cat_attr_1_val_val[item + curr_metadata.offset] * N_2;
        item_count++;
      }
      sublist_metadata[sublist_metadata_count].length = curr_metadata.length;
      sublist_metadata[sublist_metadata_count].offset = skipped;
      skipped += curr_metadata.length;
      sublist_metadata_count++;
    }

    for (duckdb::idx_t column = 0; column < cat_attr_size_2; column++) {
      auto curr_metadata = v_2_sublist_metadata[column + (i*cat_attr_size_2)];
      for (duckdb::idx_t item = 0; item < curr_metadata.length; item++) {
        cat_set_val_key[item_count] = cat_attr_2_val_key[v_tmp[0].sel->get_index(item + curr_metadata.offset)];//[];
        //std::cout<<"Key is "<<cat_attr_2_val_key[v_tmp[0].sel->get_index(item + curr_metadata.offset)]<<" at offset "<<item + curr_metadata.offset<<std::endl;
        cat_set_val_val[item_count] = cat_attr_2_val_val[v_tmp[1].sel->get_index(item + curr_metadata.offset)]* N_1;//[item + curr_metadata.offset] * N_1;
        //std::cout<<"Value is "<<cat_attr_2_val_val[v_tmp[1].sel->get_index(item + curr_metadata.offset)]<<std::endl;
        item_count++;
      }
      sublist_metadata[sublist_metadata_count].length = curr_metadata.length;
      sublist_metadata[sublist_metadata_count].offset = skipped;
      skipped += curr_metadata.length;
      sublist_metadata_count++;
    }
  }

  //set linear metadata -> for each row to return the metadata (view of the full vector)
  for (duckdb::idx_t child_idx = 0; child_idx < size; child_idx++) {
    meta_lin_num[child_idx].length = num_attr_size_1 + num_attr_size_2;
    meta_lin_num[child_idx].offset =
        child_idx * meta_lin_num[child_idx].length;//ListVector::GetListSize(*result_children[1]);
    meta_lin_cat[child_idx].length = cat_attr_size_1 + cat_attr_size_2;
    meta_lin_cat[child_idx].offset =
        child_idx * meta_lin_cat[child_idx].length;//ListVector::GetListSize(*result_children[1]);
  }

  //quadratic numerical
  auto quad_lists_size_1 = num_attr_size_1;//duckdb::ListVector::GetListSize(*first_triple_children[2]) / size;
  auto quad_lists_size_2 = num_attr_size_2;//duckdb::ListVector::GetListSize(*sec_triple_children[2]) / size;

  duckdb::UnifiedVectorFormat quad_data[2];
  duckdb::ListVector::GetEntry(
      *first_triple_children[2]).ToUnifiedFormat(size, quad_data[0]);
  duckdb::ListVector::GetEntry(
      *sec_triple_children[2]).ToUnifiedFormat(size, quad_data[1]);

  auto quad_list_entries_1 = duckdb::UnifiedVectorFormat::GetData<float>(quad_data[0]);//entries are float
  auto quad_list_entries_2 = duckdb::UnifiedVectorFormat::GetData<float>(quad_data[1]);//entries are float

  (*result_children[2]).SetVectorType(duckdb::VectorType::FLAT_VECTOR);
  duckdb::ListVector::Reserve(*result_children[2], (((num_attr_size_1 + num_attr_size_2)*(num_attr_size_1 + num_attr_size_2 +1))/2)*size);
  duckdb::ListVector::SetListSize(*result_children[2], (((num_attr_size_1 + num_attr_size_2)*(num_attr_size_1 + num_attr_size_2 +1))/2)*size);
  auto meta_quad_num = duckdb::FlatVector::GetData<duckdb::list_entry_t>(*result_children[2]);

  //compute quad numerical
  //std::cout<<"set quad numerical\n";
  auto quad_res_num = duckdb::FlatVector::GetData<float>(duckdb::ListVector::GetEntry(*result_children[2]));

  //for each row
  auto set_index_num_quad = 0;
  for (duckdb::idx_t i = 0; i < size; i++) {
    auto N_1 = duckdb::UnifiedVectorFormat::GetData<int32_t>(N_data[0])[N_data[0].sel->get_index(i)];
    auto N_2 = duckdb::UnifiedVectorFormat::GetData<int32_t>(N_data[1])[N_data[1].sel->get_index(i)];
    //scale 1st quad. aggregate
    auto idx_lin_quad = 0;
    for (duckdb::idx_t j = 0; j < num_attr_size_1; j++) {
        quad_res_num[set_index_num_quad] = quad_list_entries_1[quad_data[0].sel->get_index(idx_lin_quad + (i * quad_lists_size_1))] * (N_2);
        idx_lin_quad++;
        set_index_num_quad++;
    }

    //scale 2nd quad. aggregate
    for (duckdb::idx_t j = 0; j < quad_lists_size_2; j++) {
      quad_res_num[set_index_num_quad] = quad_list_entries_2[quad_data[1].sel->get_index(j + (i * quad_lists_size_2))] * (N_1);
      set_index_num_quad++;
    }
  }
}

//Returns the datatype used by this function
duckdb::unique_ptr<duckdb::FunctionData>
Triple::multiply_nb_bind(duckdb::ClientContext &context, duckdb::ScalarFunction &function,
             duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> &arguments) {

  D_ASSERT(arguments.size() == 2);
  //function.return_type = arguments[0]->return_type;

  duckdb::child_list_t<duckdb::LogicalType> struct_children;
  struct_children.emplace_back("N", duckdb::LogicalType::INTEGER);
  struct_children.emplace_back("lin_num", duckdb::LogicalType::LIST(duckdb::LogicalType::FLOAT));
  struct_children.emplace_back("quad_num", duckdb::LogicalType::LIST(duckdb::LogicalType::FLOAT));

  //categorical structures
  duckdb::child_list_t<duckdb::LogicalType> lin_cat;
  lin_cat.emplace_back("key", duckdb::LogicalType::INTEGER);
  lin_cat.emplace_back("value", duckdb::LogicalType::FLOAT);

  struct_children.emplace_back("lin_cat", duckdb::LogicalType::LIST(duckdb::LogicalType::LIST(duckdb::LogicalType::STRUCT(lin_cat))));
  //lin_cat -> LIST(LIST(STRUCT(key1, key2, val))). E.g. [[{k1,k2,2},{k3,k4,5}],[]]...
  //quad_cat

  auto struct_type = duckdb::LogicalType::STRUCT(struct_children);
  function.return_type = struct_type;
  //function.varargs = LogicalType::ANY;
  return duckdb::make_uniq<duckdb::VariableReturnBindData>(function.return_type);
}
