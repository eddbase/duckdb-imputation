

#include <triple/mul.h>
#include <duckdb/function/scalar/nested_functions.hpp>
#include <duckdb/planner/expression/bound_function_expression.hpp>
#include <duckdb/storage/statistics/list_stats.hpp>

#include <map>
#include <iostream>
#include <algorithm>
#include <utils.h>
//DataChunk is a set of vectors




namespace Triple {
//actual implementation of this function
void MultiplyFunction(duckdb::DataChunk &args, duckdb::ExpressionState &state, duckdb::Vector &result) {
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

  //(*first_triple_children[3]).Flatten(size);
  //(*sec_triple_children[3]).Flatten(size);
  auto cat_attr_size_1 = duckdb::ListVector::GetListSize(*first_triple_children[3]);
  auto cat_attr_size_2 = duckdb::ListVector::GetListSize(*sec_triple_children[3]);
  //first_triple_children[3]->Print();
  if (cat_attr_size_1 > 0)
    cat_attr_size_1 = duckdb::ListVector::GetData(*first_triple_children[3])[0].length;
  //cat_attr_size_1 = duckdb::ListValue::GetChildren(duckdb::ListVector::GetEntry(*first_triple_children[3]).GetValue(0)).size();
  if (cat_attr_size_2 > 0)
    cat_attr_size_2 = duckdb::ListVector::GetData(*sec_triple_children[3])[0].length;
  //cat_attr_size_2 = duckdb::ListValue::GetChildren(duckdb::ListVector::GetEntry(*sec_triple_children[3]).GetValue(0)).size();

  //D_ASSERT((*first_triple_children[3]).GetVectorType() == duckdb::VectorType::FLAT_VECTOR);
  //D_ASSERT((*sec_triple_children[3]).GetVectorType() == duckdb::VectorType::FLAT_VECTOR);

  //auto cat_attr_size_1 = duckdb::ListVector::GetListSize(*first_triple_children[3]) / size;
  //auto cat_attr_size_2 = duckdb::ListVector::GetListSize(*sec_triple_children[3]) / size;

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
  for (duckdb::idx_t i = 0; i < size; i++) {
    //add first list for the tuple
    auto N_1 = duckdb::UnifiedVectorFormat::GetData<int32_t>(N_data[0])[N_data[0].sel->get_index(i)];
    auto N_2 = duckdb::UnifiedVectorFormat::GetData<int32_t>(N_data[1])[N_data[1].sel->get_index(i)];

    for (duckdb::idx_t j = 0; j < num_attr_size_1; j++)
      lin_res_num[(i*(num_attr_size_1 + num_attr_size_2)) + j] = lin_list_entries_1[lin_data[0].sel->get_index(j + (i * num_attr_size_1))] * N_2;

    for (duckdb::idx_t j = 0; j < num_attr_size_2; j++)
      lin_res_num[(i*(num_attr_size_1 + num_attr_size_2)) + j + num_attr_size_1] = lin_list_entries_2[lin_data[1].sel->get_index(j + (i * num_attr_size_2))] * N_1;
  }

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

  //set quadratic aggregates
  //std::cout<<"set quadratic aggregates\n";

  //RecursiveFlatten(*first_triple_children[2], size);
  //        D_ASSERT(first_triple_children[2]->GetVectorType() == duckdb::VectorType::FLAT_VECTOR);
  //RecursiveFlatten(*sec_triple_children[2], size);
  //        D_ASSERT(sec_triple_children[2]->GetVectorType() == duckdb::VectorType::FLAT_VECTOR);

  auto quad_lists_size_1 = num_attr_size_1 * (num_attr_size_1+1)/2;//duckdb::ListVector::GetListSize(*first_triple_children[2]) / size;
  auto quad_lists_size_2 = num_attr_size_2 * (num_attr_size_2+1)/2;//duckdb::ListVector::GetListSize(*sec_triple_children[2]) / size;

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
  auto meta_quad_num_cat = duckdb::FlatVector::GetData<duckdb::list_entry_t>(*result_children[4]);
  auto meta_quad_cat_cat = duckdb::FlatVector::GetData<duckdb::list_entry_t>(*result_children[5]);

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
      for (duckdb::idx_t k = j; k < num_attr_size_1; k++) {//a list * b (AA, BB, <prod. A and cols other table computed down>, AB)
        quad_res_num[set_index_num_quad] = quad_list_entries_1[quad_data[0].sel->get_index(idx_lin_quad + (i * quad_lists_size_1))] * (N_2);
        idx_lin_quad++;
        set_index_num_quad++;
      }

      //multiply lin. aggregates
      for (duckdb::idx_t k = 0; k < num_attr_size_2; k++) {
        quad_res_num[set_index_num_quad] = lin_list_entries_1[lin_data[0].sel->get_index(j + (i * num_attr_size_1))] *
                                           lin_list_entries_2[lin_data[1].sel->get_index(k + (i * num_attr_size_2))];
        set_index_num_quad++;
        //std::cout<<"Quad NUM: "<<lin_list_entries_1[j + (i * num_attr_size_1)] *
        //                         lin_list_entries_2[k + (i * num_attr_size_2)]<<"\n";
      }
    }

    //scale 2nd quad. aggregate
    for (duckdb::idx_t j = 0; j < quad_lists_size_2; j++) {
      quad_res_num[set_index_num_quad] = quad_list_entries_2[quad_data[1].sel->get_index(j + (i * quad_lists_size_2))] * (N_1);
      set_index_num_quad++;
    }
  }

  //set num*cat aggregates
  duckdb::Vector v_num_cat_quad_1 = duckdb::ListVector::GetEntry(
      *first_triple_children[4]);
  duckdb::Vector v_num_cat_quad_2 = duckdb::ListVector::GetEntry(
      *sec_triple_children[4]);

  auto size_num_cat_cols = ((num_attr_size_1 * cat_attr_size_1) + (num_attr_size_1 * cat_attr_size_2)
                            + (num_attr_size_2 * cat_attr_size_1) + (num_attr_size_2 * cat_attr_size_2));

  duckdb::ListVector::Reserve(*result_children[4], size_num_cat_cols * size);
  duckdb::ListVector::SetListSize(*result_children[4], size_num_cat_cols * size);

  //result_children[4]->SetVectorType(VectorType::FLAT_VECTOR);
  duckdb::Vector &v_num_cat_quad_res = duckdb::ListVector::GetEntry(*result_children[4]);

  //new size is (cont_A * cat_A) (curr. size) + (cont_A * cat_B) + (cont_B * cat_A) + (cont_B * cat_B)
  int *cat_attr_1_num_cat_key = nullptr;
  int *cat_attr_2_num_cat_key = nullptr;
  float *cat_attr_1_num_cat_val = nullptr;
  float *cat_attr_2_num_cat_val = nullptr;
  duckdb::list_entry_t *v_1_sublist_metadata_num_cat = nullptr;
  duckdb::list_entry_t *v_2_sublist_metadata_num_cat = nullptr;

  if (cat_attr_size_1 >0 || cat_attr_size_2 >0) {
    //init num*cat
    //new size is cont_a * cat_a + cont_b*cat_b (both already computed) + cont_A*cat_B (size lin. catB * num attr A) + cont_B*catA
    size_t total_values = 0;
    if (cat_attr_size_1 > 0){
      auto sublist_meta = duckdb::ListVector::GetData(duckdb::ListVector::GetEntry(*first_triple_children[4]));
      auto main_list_len = duckdb::ListVector::GetListSize(*first_triple_children[4]);//n of rows cont.A*cat.A
      total_values += sublist_meta[main_list_len-1].offset + sublist_meta[main_list_len-1].length;//n. elements
    }
    if(cat_attr_size_2 > 0) {
      auto sublist_meta = duckdb::ListVector::GetData(duckdb::ListVector::GetEntry(*sec_triple_children[3]));
      auto main_list_len = duckdb::ListVector::GetListSize(*sec_triple_children[3]);//n of rows cont.A*cat.B
      total_values += ((sublist_meta[main_list_len - 1].offset + sublist_meta[main_list_len - 1].length) *
                       num_attr_size_1);//n. elements
    }
    if(cat_attr_size_1 > 0) {
      auto sublist_meta = duckdb::ListVector::GetData(duckdb::ListVector::GetEntry(*first_triple_children[3]));
      auto main_list_len = duckdb::ListVector::GetListSize(*first_triple_children[3]);//n of rows cont.B*cat.A
      total_values += ((sublist_meta[main_list_len - 1].offset + sublist_meta[main_list_len - 1].length) *
                       num_attr_size_2);//n. elements
    }
    if (cat_attr_size_2 > 0){
      auto sublist_meta = duckdb::ListVector::GetData(duckdb::ListVector::GetEntry(*sec_triple_children[4]));
      auto main_list_len = duckdb::ListVector::GetListSize(*sec_triple_children[4]);
      total_values += sublist_meta[main_list_len-1].offset + sublist_meta[main_list_len-1].length;
    }
    duckdb::Vector cat_relations_vector = duckdb::ListVector::GetEntry(*result_children[4]);
    //std::cout<<"Setting num*cat to: "<<total_values<<std::endl;
    duckdb::ListVector::Reserve(cat_relations_vector, total_values);
    duckdb::ListVector::SetListSize(cat_relations_vector, total_values);
    sublist_metadata = duckdb::ListVector::GetData(cat_relations_vector);
    duckdb::Vector cat_relations_vector_sub = duckdb::ListVector::GetEntry(cat_relations_vector);
    duckdb::vector<duckdb::unique_ptr<duckdb::Vector>> &lin_struct_vector = duckdb::StructVector::GetEntries(cat_relations_vector_sub);
    cat_set_val_key = duckdb::FlatVector::GetData<int>(*((lin_struct_vector)[0]));
    cat_set_val_val = duckdb::FlatVector::GetData<float>(*((lin_struct_vector)[1]));

    if (cat_attr_size_1 > 0) {
      duckdb::Vector v_cat_lin_1 = duckdb::ListVector::GetEntry(*first_triple_children[4]);
      v_1_sublist_metadata_num_cat = duckdb::ListVector::GetData(v_cat_lin_1);
      duckdb::Vector v_cat_lin_1_values = duckdb::ListVector::GetEntry(v_cat_lin_1);
      duckdb::vector<duckdb::unique_ptr<duckdb::Vector>> &v_cat_lin_1_struct = duckdb::StructVector::GetEntries(
          v_cat_lin_1_values);
      cat_attr_1_num_cat_key = duckdb::FlatVector::GetData<int>(*((v_cat_lin_1_struct)[0]));
      cat_attr_1_num_cat_val = duckdb::FlatVector::GetData<float>(*((v_cat_lin_1_struct)[1]));
    }
    if (cat_attr_size_2 > 0) {
      duckdb::Vector v_cat_lin_1 = duckdb::ListVector::GetEntry(*sec_triple_children[4]);
      v_2_sublist_metadata_num_cat = duckdb::ListVector::GetData(v_cat_lin_1);
      duckdb::Vector v_cat_lin_1_values = duckdb::ListVector::GetEntry(v_cat_lin_1);
      duckdb::vector<duckdb::unique_ptr<duckdb::Vector>> &v_cat_lin_1_struct = duckdb::StructVector::GetEntries(
          v_cat_lin_1_values);
      cat_attr_2_num_cat_key = duckdb::FlatVector::GetData<int>(*((v_cat_lin_1_struct)[0]));
      cat_attr_2_num_cat_val = duckdb::FlatVector::GetData<float>(*((v_cat_lin_1_struct)[1]));
    }
  }


  //compute quad num_cat

  item_count = 0;
  sublist_metadata_count = 0;
  skipped = 0;

  for (duckdb::idx_t i = 0; i < size; i++) {
    auto N_1 = duckdb::UnifiedVectorFormat::GetData<int32_t>(N_data[0])[N_data[0].sel->get_index(i)];
    auto N_2 = duckdb::UnifiedVectorFormat::GetData<int32_t>(N_data[1])[N_data[1].sel->get_index(i)];

    //scale entries cont_A * cat_A
    auto row_offset = i*(num_attr_size_1 * cat_attr_size_1);
    for (size_t col1 = 0; col1 < num_attr_size_1; col1++){
      for (size_t col2 = 0; col2 < cat_attr_size_1; col2++){
        auto curr_metadata = v_1_sublist_metadata_num_cat[(col1 * cat_attr_size_1) + col2 + row_offset];
        for (duckdb::idx_t item = 0; item < curr_metadata.length; item++) {
          cat_set_val_key[item_count] = cat_attr_1_num_cat_key[item + curr_metadata.offset];
          cat_set_val_val[item_count] = cat_attr_1_num_cat_val[item + curr_metadata.offset] * N_2;
          item_count++;
        }
        sublist_metadata[sublist_metadata_count].length = curr_metadata.length;
        sublist_metadata[sublist_metadata_count].offset = skipped;
        skipped += curr_metadata.length;
        //std::cout<<"Setting metadata to: "<<skipped<<std::endl;
        sublist_metadata_count++;
      }

      //compute cont_A * cat_B
      for (size_t col2 = 0; col2 < cat_attr_size_2; col2++){
        auto curr_metadata = v_2_sublist_metadata[col2 + (i*cat_attr_size_2)];
        for(size_t cat_val = 0; cat_val < curr_metadata.length; cat_val++){
          cat_set_val_key[item_count] = cat_attr_2_val_key[cat_val + curr_metadata.offset];
          cat_set_val_val[item_count] = cat_attr_2_val_val[cat_val + curr_metadata.offset] * lin_list_entries_1[col1 + (i*num_attr_size_1)];
          item_count++;
        }
        sublist_metadata[sublist_metadata_count].length = curr_metadata.length;
        sublist_metadata[sublist_metadata_count].offset = skipped;
        skipped += curr_metadata.length;
        //std::cout<<"Setting metadata to: "<<skipped<<std::endl;
        sublist_metadata_count++;
      }
    }

    //compute cont_B * cat_A
    for (size_t col1 = 0; col1 < num_attr_size_2; col1++){
      for (size_t col2 = 0; col2 < cat_attr_size_1; col2++){
        auto curr_metadata = v_1_sublist_metadata[col2 + (i*cat_attr_size_1)];
        for(size_t cat_val = 0; cat_val < curr_metadata.length; cat_val++){
          cat_set_val_key[item_count] = cat_attr_1_val_key[cat_val + curr_metadata.offset];
          cat_set_val_val[item_count] = cat_attr_1_val_val[cat_val + curr_metadata.offset] * lin_list_entries_2[col1 + (i*num_attr_size_2)];
          item_count++;
        }
        sublist_metadata[sublist_metadata_count].length = curr_metadata.length;
        sublist_metadata[sublist_metadata_count].offset = skipped;
        skipped += curr_metadata.length;
        //std::cout<<"Setting metadata to: "<<skipped<<std::endl;
        sublist_metadata_count++;
      }

      //scale entries cont_B * cat_B
      row_offset = i*(num_attr_size_2 * cat_attr_size_2);
      for (size_t col2 = 0; col2 < cat_attr_size_2; col2++){
        auto curr_metadata = v_2_sublist_metadata_num_cat[(col1 * cat_attr_size_2) + col2 + row_offset];
        for(size_t cat_val = 0; cat_val < curr_metadata.length; cat_val++){
          cat_set_val_key[item_count] = cat_attr_2_num_cat_key[cat_val + curr_metadata.offset];
          cat_set_val_val[item_count] = cat_attr_2_num_cat_val[cat_val + curr_metadata.offset] * N_1;
          item_count++;
        }
        sublist_metadata[sublist_metadata_count].length = curr_metadata.length;
        sublist_metadata[sublist_metadata_count].offset = skipped;
        skipped += curr_metadata.length;
        //std::cout<<"Setting metadata to: "<<skipped<<std::endl;
        sublist_metadata_count++;
      }
    }
  }

  //set cat*cat aggregates

  //size: catcat_A + catcat_B +

  duckdb::Vector v_cat_cat_quad_1 = duckdb::ListVector::GetEntry(
      *first_triple_children[5]);
  duckdb::Vector v_cat_cat_quad_2 = duckdb::ListVector::GetEntry(
      *sec_triple_children[5]);
  //new size is old size A + old size B +
  auto size_cat_cat_cols = ((cat_attr_size_1 * (cat_attr_size_1+1)) /2) + ((cat_attr_size_2 * (cat_attr_size_2+1)) /2) + (cat_attr_size_1 * cat_attr_size_2);
  duckdb::ListVector::Reserve(*result_children[5], size_cat_cat_cols * size);
  duckdb::ListVector::SetListSize(*result_children[5], size_cat_cat_cols * size);

  result_children[5]->SetVectorType(duckdb::VectorType::FLAT_VECTOR);
  duckdb::Vector &v_cat_cat_quad_res = duckdb::ListVector::GetEntry(*result_children[5]);
  auto size_row_1 = (cat_attr_size_1 * (cat_attr_size_1+1)) /2;
  auto size_row_2 = (cat_attr_size_2 * (cat_attr_size_2+1)) /2;

  int* cat_set_val_key_1 = nullptr;
  int* cat_set_val_key_2 = nullptr;
  int *cat_attr_1_cat_cat_key_1 = nullptr;
  int *cat_attr_1_cat_cat_key_2 = nullptr;
  int *cat_attr_2_cat_cat_key_1 = nullptr;
  int *cat_attr_2_cat_cat_key_2 = nullptr;
  float *cat_attr_1_cat_cat_val = nullptr;
  float *cat_attr_2_cat_cat_val = nullptr;


  if (cat_attr_size_1 >0 || cat_attr_size_2 >0) {
    size_t total_values = 0;
    if (cat_attr_size_1 > 0){
      auto sublist_meta = duckdb::ListVector::GetData(duckdb::ListVector::GetEntry(*first_triple_children[5]));
      auto main_list_len = duckdb::ListVector::GetListSize(*first_triple_children[5]);//n of rows
      total_values += sublist_meta[main_list_len-1].offset + sublist_meta[main_list_len-1].length;//n. elements
    }
    if (cat_attr_size_2 > 0){
      auto sublist_meta = duckdb::ListVector::GetData(duckdb::ListVector::GetEntry(*sec_triple_children[5]));
      auto main_list_len = duckdb::ListVector::GetListSize(*sec_triple_children[5]);
      total_values += sublist_meta[main_list_len-1].offset + sublist_meta[main_list_len-1].length;
    }
    if(cat_attr_size_1 > 0 && cat_attr_size_2 > 0) {
      auto sublist_meta_1 = duckdb::ListVector::GetData(
          duckdb::ListVector::GetEntry(*first_triple_children[3]));
      auto main_list_len_1 = duckdb::ListVector::GetListSize(*first_triple_children[3]);
      auto sublist_meta_2 = duckdb::ListVector::GetData(
          duckdb::ListVector::GetEntry(*sec_triple_children[3]));
      auto main_list_len_2 = duckdb::ListVector::GetListSize(*sec_triple_children[3]);

      for (size_t k = 0; k < size; k++) {
        for (size_t i = 0; i < cat_attr_size_1; i++) {
          for (size_t j = 0; j < cat_attr_size_2; j++) {
            total_values += (sublist_meta_1[(k*cat_attr_size_1)+i].length *
                             sublist_meta_2[(k*cat_attr_size_2)+j].length);
          }
        }
      }
    }

    duckdb::Vector cat_relations_vector = duckdb::ListVector::GetEntry(*result_children[5]);
    duckdb::ListVector::Reserve(cat_relations_vector, total_values);
    duckdb::ListVector::SetListSize(cat_relations_vector, total_values);
    sublist_metadata = duckdb::ListVector::GetData(cat_relations_vector);
    duckdb::Vector cat_relations_vector_sub = duckdb::ListVector::GetEntry(cat_relations_vector);
    duckdb::vector<duckdb::unique_ptr<duckdb::Vector>> &lin_struct_vector = duckdb::StructVector::GetEntries(cat_relations_vector_sub);
    cat_set_val_key_1 = duckdb::FlatVector::GetData<int>(*((lin_struct_vector)[0]));
    cat_set_val_key_2 = duckdb::FlatVector::GetData<int>(*((lin_struct_vector)[1]));
    cat_set_val_val = duckdb::FlatVector::GetData<float>(*((lin_struct_vector)[2]));

    if (cat_attr_size_1 > 0) {
      duckdb::Vector v_cat_lin_1 = duckdb::ListVector::GetEntry(*first_triple_children[5]);
      v_1_sublist_metadata_num_cat = duckdb::ListVector::GetData(v_cat_lin_1);
      duckdb::Vector v_cat_lin_1_values = duckdb::ListVector::GetEntry(v_cat_lin_1);
      duckdb::vector<duckdb::unique_ptr<duckdb::Vector>> &v_cat_lin_1_struct = duckdb::StructVector::GetEntries(
          v_cat_lin_1_values);
      cat_attr_1_cat_cat_key_1 = duckdb::FlatVector::GetData<int>(*((v_cat_lin_1_struct)[0]));
      cat_attr_1_cat_cat_key_2 = duckdb::FlatVector::GetData<int>(*((v_cat_lin_1_struct)[1]));
      cat_attr_1_cat_cat_val = duckdb::FlatVector::GetData<float>(*((v_cat_lin_1_struct)[2]));
    }
    if (cat_attr_size_2 > 0) {
      duckdb::Vector v_cat_lin_1 = duckdb::ListVector::GetEntry(*sec_triple_children[5]);
      v_2_sublist_metadata_num_cat = duckdb::ListVector::GetData(v_cat_lin_1);
      duckdb::Vector v_cat_lin_1_values = duckdb::ListVector::GetEntry(v_cat_lin_1);
      duckdb::vector<duckdb::unique_ptr<duckdb::Vector>> &v_cat_lin_1_struct = duckdb::StructVector::GetEntries(
          v_cat_lin_1_values);
      cat_attr_2_cat_cat_key_1 = duckdb::FlatVector::GetData<int>(*((v_cat_lin_1_struct)[0]));
      cat_attr_2_cat_cat_key_2 = duckdb::FlatVector::GetData<int>(*((v_cat_lin_1_struct)[1]));
      cat_attr_2_cat_cat_val = duckdb::FlatVector::GetData<float>(*((v_cat_lin_1_struct)[2]));
    }
  }

  item_count = 0;
  sublist_metadata_count = 0;
  skipped = 0;

  for (duckdb::idx_t i = 0; i < size; i++) {
    auto N_1 = duckdb::UnifiedVectorFormat::GetData<int32_t>(N_data[0])[N_data[0].sel->get_index(i)];
    auto N_2 = duckdb::UnifiedVectorFormat::GetData<int32_t>(N_data[1])[N_data[1].sel->get_index(i)];

    // (cat_A * cat_A) * count_B (scale cat_A)
    size_t idx_cat_in = 0;
    for (size_t col1 = 0; col1 < cat_attr_size_1; col1++) {
      for (size_t col2 = col1; col2 < cat_attr_size_1; col2++) {
        auto curr_metadata = v_1_sublist_metadata_num_cat[idx_cat_in + (i * size_row_1)];
        for (size_t el = 0; el < curr_metadata.length; el++) {
          cat_set_val_key_1[item_count] = cat_attr_1_cat_cat_key_1[el + curr_metadata.offset];
          cat_set_val_key_2[item_count] = cat_attr_1_cat_cat_key_2[el + curr_metadata.offset];
          cat_set_val_val[item_count] = cat_attr_1_cat_cat_val[el + curr_metadata.offset] * N_2;
          item_count++;
        }
        sublist_metadata[sublist_metadata_count].length = curr_metadata.length;
        sublist_metadata[sublist_metadata_count].offset = skipped;
        skipped += curr_metadata.length;
        sublist_metadata_count++;
        idx_cat_in++;
      }

      //cat A * cat B
      auto curr_metadata_1 = v_1_sublist_metadata[col1 + (i*cat_attr_size_1)];
      for (size_t col2 = 0; col2 < cat_attr_size_2; col2++) {
        auto curr_metadata_2 = v_2_sublist_metadata[col2 + (i*cat_attr_size_2)];
        for (size_t j = 0; j < curr_metadata_1.length; j++){
          for (size_t k = 0; k < curr_metadata_2.length; k++){
            cat_set_val_key_1[item_count] = cat_attr_1_val_key[j + curr_metadata_1.offset];
            cat_set_val_key_2[item_count] = cat_attr_2_val_key[k + curr_metadata_2.offset];
            cat_set_val_val[item_count] = cat_attr_1_val_val[j + curr_metadata_1.offset] * cat_attr_2_val_val[k + curr_metadata_2.offset];
            item_count++;
          }
        }
        sublist_metadata[sublist_metadata_count].length = curr_metadata_1.length * curr_metadata_2.length;
        sublist_metadata[sublist_metadata_count].offset = skipped;
        skipped += (curr_metadata_1.length * curr_metadata_2.length);
        sublist_metadata_count++;
      }
    }

    // (cat_B * cat_B) * count_A (scale cat_B)
    for (size_t col = 0; col < size_row_2; col++) {
      auto curr_metadata = v_2_sublist_metadata_num_cat[col + (i*size_row_2)];
      for (size_t el = 0; el < curr_metadata.length; el++){
        cat_set_val_key_1[item_count] = cat_attr_2_cat_cat_key_1[el + curr_metadata.offset];
        cat_set_val_key_2[item_count] = cat_attr_2_cat_cat_key_2[el + curr_metadata.offset];
        cat_set_val_val[item_count] = cat_attr_2_cat_cat_val[el + curr_metadata.offset] * N_1;
        item_count++;
      }
      sublist_metadata[sublist_metadata_count].length = curr_metadata.length;
      sublist_metadata[sublist_metadata_count].offset = skipped;
      skipped += curr_metadata.length;
      sublist_metadata_count++;
      idx_cat_in++;
    }
  }

  //set for each row to return the metadata (view of the full vector)
  for (duckdb::idx_t row = 0; row < size; row++) {
    meta_quad_num[row].length = ((num_attr_size_1 * (num_attr_size_1 + 1))/2) + (num_attr_size_1 * num_attr_size_2) + ((num_attr_size_2 * (num_attr_size_2 + 1))/2);
    meta_quad_num[row].offset = row * meta_quad_num[row].length;

    meta_quad_num_cat[row].length = size_num_cat_cols;
    meta_quad_num_cat[row].offset = row * meta_quad_num_cat[row].length;

    meta_quad_cat_cat[row].length = size_cat_cat_cols;
    meta_quad_cat_cat[row].offset = row * meta_quad_cat_cat[row].length;
  }
}

//Returns the datatype used by this function
duckdb::unique_ptr<duckdb::FunctionData>
MultiplyBind(duckdb::ClientContext &context, duckdb::ScalarFunction &function,
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
  //function.varargs = LogicalType::ANY;
  return duckdb::make_uniq<duckdb::VariableReturnBindData>(function.return_type);
}

//Generate statistics for this function. Given input type ststistics (mainly min and max for every attribute), returns the output statistics
duckdb::unique_ptr<duckdb::BaseStatistics>
MultiplyStats(duckdb::ClientContext &context, duckdb::FunctionStatisticsInput &input) {
  auto &child_stats = input.child_stats;
  auto &expr = input.expr;
  auto struct_stats = duckdb::StructStats::CreateUnknown(expr.return_type);
  /*
  duckdb::StructStats::Copy(struct_stats, child_stats[0]);
  duckdb::BaseStatistics &ret_num_stats = duckdb::StructStats::GetChildStats(struct_stats, 0);

  //set statistics for N
  if (!duckdb::NumericStats::HasMinMax(ret_num_stats) || !duckdb::NumericStats::HasMinMax(duckdb::StructStats::GetChildStats(child_stats[1], 0)))
          return struct_stats.ToUnique();


  int32_t n_1_min = duckdb::NumericStats::Min(ret_num_stats).GetValue<int32_t>();
  int32_t n_1_max = duckdb::NumericStats::Max(ret_num_stats).GetValue<int32_t>();

  int32_t n_2_min = duckdb::NumericStats::GetMin<int32_t>(duckdb::StructStats::GetChildStats(child_stats[1], 0));
  int32_t n_2_max = duckdb::NumericStats::GetMax<int32_t>(duckdb::StructStats::GetChildStats(child_stats[1], 0));

  duckdb::NumericStats::SetMax(ret_num_stats, duckdb::Value(n_1_max * n_2_max));
  duckdb::NumericStats::SetMin(ret_num_stats, duckdb::Value(n_1_min * n_2_min));

  //set statistics for lin. aggregates
  duckdb::BaseStatistics& list_num_stat = duckdb::ListStats::GetChildStats(duckdb::StructStats::GetChildStats(struct_stats, 1));

  if (!duckdb::NumericStats::HasMinMax(list_num_stat) || !duckdb::NumericStats::HasMinMax(duckdb::ListStats::GetChildStats(duckdb::StructStats::GetChildStats(child_stats[1], 1))))
      return struct_stats.ToUnique();

  float lin_1_min = duckdb::NumericStats::GetMin<float>(list_num_stat);
  float lin_1_max = duckdb::NumericStats::GetMax<float>(list_num_stat);

  float lin_2_min = duckdb::NumericStats::GetMin<float>(duckdb::ListStats::GetChildStats(duckdb::StructStats::GetChildStats(child_stats[1], 1)));
  float lin_2_max = duckdb::NumericStats::GetMax<float>(duckdb::ListStats::GetChildStats(duckdb::StructStats::GetChildStats(child_stats[1], 1)));

  int32_t n_1_min_n = n_1_min;
  int32_t n_1_max_n = n_1_max;
  int32_t n_2_min_n = n_2_min;
  int32_t n_2_max_n = n_2_max;

  if (lin_1_max < 0){
      int tmp = n_2_max_n;
      n_2_max_n = n_2_min_n;
      n_2_min_n = tmp;
  }
  else if (lin_1_min < 0){//& max > 0
      n_2_min_n = n_2_max_n;
  }
  if (lin_2_max < 0){
      int tmp = n_1_max_n;
      n_1_max_n = n_1_min_n;
      n_1_min_n = tmp;
  }
  else if (lin_2_min < 0){//& max > 0
      n_1_min_n = n_1_max_n;
  }

  duckdb::NumericStats::SetMax(list_num_stat, duckdb::Value(std::max(n_2_max_n * lin_1_max, n_1_max_n* lin_2_max)));//n max and min are the same

  duckdb::NumericStats::SetMin(list_num_stat, duckdb::Value(std::min(n_2_min_n * lin_1_min, n_1_min_n * lin_2_min)));

  //set statistics for quadratic aggregate
  duckdb::BaseStatistics& list_quad_stat = duckdb::ListStats::GetChildStats(duckdb::StructStats::GetChildStats(struct_stats, 2));

  //max value in quadratic aggreagate is max of linear product, scale first list and scale second list
  if (!duckdb::NumericStats::HasMinMax(list_quad_stat) || duckdb::NumericStats::HasMinMax(duckdb::ListStats::GetChildStats(duckdb::StructStats::GetChildStats(child_stats[1], 2))))
      return struct_stats.ToUnique();

  n_1_min_n = n_1_min;
  n_1_max_n = n_1_max;
  n_2_min_n = n_2_min;
  n_2_max_n = n_2_max;

  if (duckdb::NumericStats::GetMax<float>(list_quad_stat) < 0){
      int tmp = n_2_max_n;
      n_2_max_n = n_2_min_n;
      n_2_min_n = tmp;
  }
  else if (duckdb::NumericStats::GetMin<float>(list_quad_stat) < 0){
      n_2_min_n = n_2_max_n;
  }
  if (duckdb::NumericStats::GetMax<float>(duckdb::ListStats::GetChildStats(duckdb::StructStats::GetChildStats(child_stats[1], 2))) < 0){
      int tmp = n_1_max_n;
      n_1_max_n = n_1_min_n;
      n_1_min_n = tmp;
  }
  else if (duckdb::NumericStats::GetMin<float>(duckdb::ListStats::GetChildStats(duckdb::StructStats::GetChildStats(child_stats[1], 2))) < 0){
      n_1_min_n = n_1_max_n;
  }


  duckdb::NumericStats::SetMax(list_quad_stat, duckdb::Value(std::max(std::max(std::max(std::max(lin_1_max * lin_2_max, lin_1_min * lin_2_min), lin_1_max*lin_2_min), lin_2_max*lin_1_min),
                                                                      std::max(n_2_max_n * duckdb::NumericStats::GetMax<float>(list_quad_stat),
                                                                     n_1_max_n * duckdb::NumericStats::GetMax<float>(duckdb::ListStats::GetChildStats(duckdb::StructStats::GetChildStats(child_stats[1], 2)))))));

  duckdb::NumericStats::SetMin(list_quad_stat, duckdb::Value(std::max(std::max(std::max(std::max(lin_1_max * lin_2_max, lin_1_min * lin_2_min), lin_1_max*lin_2_min), lin_2_max*lin_1_min),
                                                                      std::min(n_2_min_n * duckdb::NumericStats::GetMin<float>(list_quad_stat),
                                                                               n_1_min_n * duckdb::NumericStats::GetMin<float>(duckdb::ListStats::GetChildStats(duckdb::StructStats::GetChildStats(child_stats[1], 2)))))));

  std::cout << "statistics StructPackStats" << struct_stats.ToString() << std::endl;
   */
  return struct_stats.ToUnique();
}
}
