
#include "duckdb/function/scalar/nested_functions.hpp"
#include <triple/sum/sum_state.h>

#include <iostream>

/*
 * Combine states for triple and nb aggregates
 */
void
Triple::SumStateCombine(duckdb::Vector &state, duckdb::Vector &combined, duckdb::AggregateInputData &aggr_input_data,
                        idx_t count) {
  //std::cout<<"\nstart combine\n"<<std::endl;

  duckdb::UnifiedVectorFormat sdata;
  state.ToUnifiedFormat(count, sdata);
  auto states_ptr = (Triple::SumState **) sdata.data;

  //std::cout<<"SUMSTATECOMBINE"<<std::endl;

  auto combined_ptr = duckdb::FlatVector::GetData<Triple::SumState *>(combined);

  for (idx_t i = 0; i < count; i++) {
    auto state = states_ptr[sdata.sel->get_index(i)];
    combined_ptr[i]->count += state->count;
    if (combined_ptr[i]->lin_agg == nullptr && combined_ptr[i]->quad_num_cat == nullptr) {//init
      combined_ptr[i]->is_nb_aggregates = state->is_nb_aggregates;
      combined_ptr[i]->num_attributes = state->num_attributes;
      combined_ptr[i]->cat_attributes = state->cat_attributes;
      if(state->num_attributes > 0) {
        if (!state->is_nb_aggregates)
          combined_ptr[i]->lin_agg = new float[state->num_attributes +
                                               ((state->num_attributes * (state->num_attributes + 1)) / 2)];
        else
          combined_ptr[i]->lin_agg = new float[state->num_attributes + state->num_attributes];


        combined_ptr[i]->quadratic_agg = &(combined_ptr[i]->lin_agg[state->num_attributes]);


        for (idx_t k = 0; k < state->num_attributes; k++)
          combined_ptr[i]->lin_agg[k] = 0;

        if(!state->is_nb_aggregates) {
          for (idx_t k = 0; k < (state->num_attributes * (state->num_attributes + 1)) / 2; k++)
            combined_ptr[i]->quadratic_agg[k] = 0;
        }
        else{
          for (idx_t k = 0; k < (state->num_attributes); k++)
            combined_ptr[i]->quadratic_agg[k] = 0;
        }
      }

      if(state->cat_attributes > 0) {
        combined_ptr[i]->quad_num_cat = new std::map<int, std::vector<float>>[state->cat_attributes];//boost::container::flat_map<int, std::vector<float>>[state->cat_attributes];
        if(!state->is_nb_aggregates)
          combined_ptr[i]->quad_cat_cat = new std::map<std::pair<int, int>, float>[//boost::container::flat_map<std::pair<int, int>, float>[
              state->cat_attributes * (state->cat_attributes + 1) / 2];
      }
    }
    /*
    if(state->quad_cat_cat == nullptr)
      std::cout<<"quad_cat_cat null"<<std::endl;
    else
      std::cout<<"quad_cat_cat not null"<<std::endl;
    if(combined_ptr[i]->quad_cat_cat == nullptr)
      std::cout<<"combined null"<<std::endl;
    else
      std::cout<<"combined not null"<<std::endl;
      */

    //SUM NUMERICAL STATES
    for (int j = 0; j < combined_ptr[i]->num_attributes; j++) {
      combined_ptr[i]->lin_agg[j] += state->lin_agg[j];
    }
    if(!state->is_nb_aggregates)
      for (int j = 0; j < combined_ptr[i]->num_attributes*(combined_ptr[i]->num_attributes+1)/2; j++) {
        combined_ptr[i]->quadratic_agg[j] += state->quadratic_agg[j];
      }
    else
      for (int j = 0; j < combined_ptr[i]->num_attributes; j++) {
        combined_ptr[i]->quadratic_agg[j] += state->quadratic_agg[j];
      }

    //SUM CATEGORICAL STATES
    //num*categorical (and count)
    for (int j = 0; j < combined_ptr[i]->cat_attributes; j++) {
      auto &taget_map = combined_ptr[i]->quad_num_cat[j];//map of cat. col j
      for (auto const& state_vals : state->quad_num_cat[j]){//search in combine states map state values
        auto pos = taget_map.find(state_vals.first);
        if (pos == taget_map.end())
          taget_map[state_vals.first] = std::vector<float>(state_vals.second);
        else
          for(int k=0; k<state_vals.second.size(); k++)
            pos->second[k] += state_vals.second[k];
      }
    }

    //categorical*categorical
    if(!state->is_nb_aggregates){
      for (int j = 0; j < combined_ptr[i]->cat_attributes*(combined_ptr[i]->cat_attributes+1)/2; j++) {
        auto &taget_map = combined_ptr[i]->quad_cat_cat[j];
        for (auto const& state_vals : state->quad_cat_cat[j]){//search in combine states map state values
          auto pos = taget_map.find(state_vals.first);
          if (pos == taget_map.end())
            taget_map[state_vals.first] = state_vals.second;
          else
            pos->second += state_vals.second;
        }
      }
    }
  }
  //std::cout<<"\nCOMBINE END\n"<<std::endl;
}

void Triple::SumStateFinalize(duckdb::Vector &state_vector, duckdb::AggregateInputData &aggr_input_data, duckdb::Vector &result,
                              idx_t count,
                              idx_t offset) {
  //std::cout<<"Start fin"<<std::endl;
  assert(offset == 0);
  D_ASSERT(result.GetType().id() == duckdb::LogicalTypeId::STRUCT);

  duckdb::UnifiedVectorFormat sdata;
  state_vector.ToUnifiedFormat(count, sdata);
  auto states = (Triple::SumState **) sdata.data;
  auto &children = duckdb::StructVector::GetEntries(result);
  duckdb::Vector &c1 = *(children[0]);//N
  auto input_data = (int32_t *) duckdb::FlatVector::GetData(c1);
  if(count == 0)
    return ;

  for (idx_t i=0;i<count; i++) {
    const auto row_id = i + offset;
    input_data[row_id] = states[sdata.sel->get_index(i)]->count;
  }

  //Set List
  duckdb::Vector &c2 = *(children[1]);
  D_ASSERT(c2.GetType().id() == duckdb::LogicalTypeId::LIST);
  duckdb::Vector &c3 = *(children[2]);
  D_ASSERT(c3.GetType().id() == duckdb::LogicalTypeId::LIST);
  if(count > 0 && states[sdata.sel->get_index(0)]->num_attributes > 0) {
    duckdb::ListVector::Reserve(c2, states[sdata.sel->get_index(0)]->num_attributes * count);
    duckdb::ListVector::SetListSize(c2, states[sdata.sel->get_index(0)]->num_attributes * count);

    if(states[sdata.sel->get_index(0)]->is_nb_aggregates){
      duckdb::ListVector::Reserve(c3, states[sdata.sel->get_index(0)]->num_attributes * count);
      duckdb::ListVector::SetListSize(c3, states[sdata.sel->get_index(0)]->num_attributes * count);
    }
    else{
      duckdb::ListVector::Reserve(c3, ((states[sdata.sel->get_index(0)]->num_attributes * (states[sdata.sel->get_index(0)]->num_attributes +1))/2) * count);
      duckdb::ListVector::SetListSize(c3, ((states[sdata.sel->get_index(0)]->num_attributes * (states[sdata.sel->get_index(0)]->num_attributes +1))/2) * count);
    }
  }

  c2.SetVectorType(duckdb::VectorType::FLAT_VECTOR);
  auto result_data = duckdb::ListVector::GetData(c2);
  auto num_lin_res = duckdb::FlatVector::GetData<float>(duckdb::ListVector::GetEntry(c2));
  auto num_cat_res = duckdb::FlatVector::GetData<float>(duckdb::ListVector::GetEntry(c3));

  //copy lin agg
  for (idx_t i = 0; i < count; i++) {
    auto state = states[sdata.sel->get_index(i)];
    const auto row_id = i + offset;
    for (int j = 0; j < state->num_attributes; j++) {
      num_lin_res[j + (i*state->num_attributes)] = state->lin_agg[j];
    }
    result_data[row_id].length = state->num_attributes;
    result_data[row_id].offset = result_data[i].length*i;//ListVector::GetListSize(c2);
    //std::cout<<"linear len: "<<result_data[row_id].length<<"linear offs: "<<result_data[row_id].offset<<std::endl;
  }

  //set quadratic attributes
  c3.SetVectorType(duckdb::VectorType::FLAT_VECTOR);
  auto result_data2 = duckdb::ListVector::GetData(c3);

  if(states[sdata.sel->get_index(0)]->is_nb_aggregates) {
    for (idx_t i = 0; i < count; i++) {
      auto state = states[sdata.sel->get_index(i)];
      const auto row_id = i + offset;
      for (int j = 0; j < state->num_attributes; j++) {
        num_cat_res[j + (i*(state->num_attributes))] = state->quadratic_agg[j];
      }//Value::Numeric
      result_data2[row_id].length = state->num_attributes;
      result_data2[row_id].offset = i * result_data2[i].length;
      //std::cout<<"quad len: "<<result_data2[row_id].length<<"quad offs: "<<result_data2[row_id].offset<<std::endl;
    }
  }
  else{
    for (idx_t i = 0; i < count; i++) {
      auto state = states[sdata.sel->get_index(i)];
      const auto row_id = i + offset;
      for (int j = 0; j < state->num_attributes*(state->num_attributes+1)/2; j++) {
        num_cat_res[j + (i*(state->num_attributes*(state->num_attributes+1)/2))] = state->quadratic_agg[j];
      }//Value::Numeric
      result_data2[row_id].length = state->num_attributes*(state->num_attributes+1)/2;
      result_data2[row_id].offset = i * result_data2[i].length;
      //std::cout<<"quad len: "<<result_data2[row_id].length<<"quad offs: "<<result_data2[row_id].offset<<std::endl;
    }
  }

  //categorical sums

  duckdb::Vector &c4 = *(children[3]);
  D_ASSERT(c4.GetType().id() == duckdb::LogicalTypeId::LIST);
  c4.SetVectorType(duckdb::VectorType::FLAT_VECTOR);
  auto result_data3 = duckdb::ListVector::GetData(c4);

  //num*cat
  if(!states[sdata.sel->get_index(0)]->is_nb_aggregates) {
    duckdb::Vector &c5 = *(children[4]);
    D_ASSERT(c5.GetType().id() == duckdb::LogicalTypeId::LIST);
    c5.SetVectorType(duckdb::VectorType::FLAT_VECTOR);

    // cat*cat
    duckdb::Vector &c6 = *(children[5]);
    D_ASSERT(c6.GetType().id() == duckdb::LogicalTypeId::LIST);
    c6.SetVectorType(duckdb::VectorType::FLAT_VECTOR);
  }

  duckdb::list_entry_t *sublist_metadata_lin_cat = nullptr;
  duckdb::list_entry_t *sublist_metadata_num_cat = nullptr;
  duckdb::list_entry_t *sublist_metadata_cat_cat = nullptr;
  int *cat_set_val_key_lin_cat = nullptr;
  float *cat_set_val_val_lin_cat = nullptr;
  int *cat_set_val_key_num_cat = nullptr;
  float *cat_set_val_val_num_cat = nullptr;


  int cat_attributes = 0;
  int num_attributes = 0;

  if (count > 0 && states[sdata.sel->get_index(0)]->cat_attributes > 0) {
    //init list size
    cat_attributes = states[sdata.sel->get_index(0)]->cat_attributes;
    num_attributes = states[sdata.sel->get_index(0)]->num_attributes;
    int n_keys_cat_columns = 0;
    int n_items_cat_cat = 0;
    for (idx_t i = 0; i < count; i++) {
      auto state = states[sdata.sel->get_index(i)];
      for(idx_t j=0; j<cat_attributes; j++)
        n_keys_cat_columns += state->quad_num_cat[j].size();
    }

    {
      c4.SetVectorType(duckdb::VectorType::FLAT_VECTOR);
      duckdb::Vector cat_relations_vector = duckdb::ListVector::GetEntry(c4);
      duckdb::ListVector::Reserve(cat_relations_vector, n_keys_cat_columns);
      duckdb::ListVector::SetListSize(cat_relations_vector, n_keys_cat_columns);
      duckdb::ListVector::Reserve(c4, cat_attributes * count);
      duckdb::ListVector::SetListSize(c4, cat_attributes * count);
    }
    if(!states[sdata.sel->get_index(0)]->is_nb_aggregates) {

      for (idx_t i = 0; i < count; i++) {
        auto state = states[sdata.sel->get_index(i)];
        for(idx_t j=0; j<(cat_attributes*(cat_attributes+1))/2; j++)
          n_items_cat_cat += state->quad_cat_cat[j].size();
      }


      {
        duckdb::Vector &c5 = *(children[4]);
        c5.SetVectorType(duckdb::VectorType::FLAT_VECTOR);
        duckdb::Vector cat_relations_vector = duckdb::ListVector::GetEntry(c5);
        duckdb::ListVector::Reserve(cat_relations_vector, n_keys_cat_columns * num_attributes);
        duckdb::ListVector::SetListSize(cat_relations_vector, n_keys_cat_columns * num_attributes);
        duckdb::ListVector::Reserve(c5, cat_attributes * num_attributes * count);
        duckdb::ListVector::SetListSize(c5, cat_attributes * num_attributes * count);
      }
      {
        duckdb::Vector &c6 = *(children[5]);
        c6.SetVectorType(duckdb::VectorType::FLAT_VECTOR);
        duckdb::Vector cat_relations_vector = duckdb::ListVector::GetEntry(c6);
        duckdb::ListVector::Reserve(cat_relations_vector, n_items_cat_cat);
        duckdb::ListVector::SetListSize(cat_relations_vector, n_items_cat_cat);
        duckdb::ListVector::Reserve(c6, ((cat_attributes * (cat_attributes+1))/2) * count);
        duckdb::ListVector::SetListSize(c6, ((cat_attributes * (cat_attributes+1))/2) * count);
      }
    }
  }

  //define lin_cat
  if(cat_attributes > 0){
    duckdb::Vector cat_relations_vector = duckdb::ListVector::GetEntry(c4);
    sublist_metadata_lin_cat = duckdb::ListVector::GetData(cat_relations_vector);
    duckdb::Vector cat_relations_vector_sub = duckdb::ListVector::GetEntry(
        cat_relations_vector);//this is a sequence of values (struct in our case, 2 vectors)
    duckdb::vector<duckdb::unique_ptr<duckdb::Vector>> &lin_struct_vector = duckdb::StructVector::GetEntries(
        cat_relations_vector_sub);
    c4.SetVectorType(duckdb::VectorType::FLAT_VECTOR);
    ((lin_struct_vector)[0])->SetVectorType(duckdb::VectorType::FLAT_VECTOR);
    ((lin_struct_vector)[1])->SetVectorType(duckdb::VectorType::FLAT_VECTOR);
    cat_set_val_key_lin_cat = duckdb::FlatVector::GetData<int>(*((lin_struct_vector)[0]));
    cat_set_val_val_lin_cat = duckdb::FlatVector::GetData<float>(*((lin_struct_vector)[1]));

  }

  idx_t idx_element = 0;
  idx_t sublist_idx_lin_cat = 0;
  idx_t skipped_lin_cat = 0;


  if(states[sdata.sel->get_index(0)]->is_nb_aggregates){
    //set lin_cat and end if nb_aggregate
    for (idx_t i = 0; i < count; i++) {
      auto state = states[sdata.sel->get_index(i)];
      const auto row_id = i + offset;

      for (int j = 0; j < state->cat_attributes; j++) {
        const auto &ordered = state->quad_num_cat[j];//map of categorical column j
        for (auto const& state_val : ordered){//for each key of the cat. variable...
          auto &num_vals = state_val.second;
          //set lin_agg
          cat_set_val_key_lin_cat[idx_element] = state_val.first;
          cat_set_val_val_lin_cat[idx_element] = num_vals[0];
          idx_element++;
        }
        sublist_metadata_lin_cat[sublist_idx_lin_cat].length = ordered.size();
        sublist_metadata_lin_cat[sublist_idx_lin_cat].offset = skipped_lin_cat;
        skipped_lin_cat += ordered.size();
        sublist_idx_lin_cat++;
      }
      result_data3[row_id].length = state->cat_attributes;
      result_data3[row_id].offset = i * result_data3[row_id].length;
    }
    return;
  }

  //non nb aggregates
  //categorical num*cat


  duckdb::Vector &c5 = *(children[4]);
  auto result_data4 = duckdb::ListVector::GetData(c5);

  // cat*cat
  duckdb::Vector &c6 = *(children[5]);
  auto result_data5 = duckdb::ListVector::GetData(c6);

  if(cat_attributes > 0 && !states[sdata.sel->get_index(0)]->is_nb_aggregates) {
    duckdb::Vector &c5 = *(children[4]);
    duckdb::Vector cat_relations_vector = duckdb::ListVector::GetEntry(c5);
    sublist_metadata_num_cat = duckdb::ListVector::GetData(cat_relations_vector);
    duckdb::Vector cat_relations_vector_sub = duckdb::ListVector::GetEntry(
        cat_relations_vector);//this is a sequence of values (struct in our case, 2 vectors)
    duckdb::vector<duckdb::unique_ptr<duckdb::Vector>> &lin_struct_vector = duckdb::StructVector::GetEntries(
        cat_relations_vector_sub);
    ((lin_struct_vector)[0])->SetVectorType(duckdb::VectorType::FLAT_VECTOR);
    ((lin_struct_vector)[1])->SetVectorType(duckdb::VectorType::FLAT_VECTOR);
    cat_set_val_key_num_cat = duckdb::FlatVector::GetData<int>(*((lin_struct_vector)[0]));
    cat_set_val_val_num_cat = duckdb::FlatVector::GetData<float>(*((lin_struct_vector)[1]));
  }

  idx_element = 0;
  idx_t sublist_idx_num_cat = 0;
  sublist_idx_lin_cat = 0;
  idx_t skipped_num_cat = 0;
  skipped_lin_cat = 0;

  size_t row_offset = 0;
  for (idx_t i = 0; i < count; i++) {

    auto state = states[sdata.sel->get_index(i)];
    const auto row_id = i + offset;

    size_t cat_offset = 0;
    for (int j = 0; j < state->cat_attributes; j++)
      cat_offset += state->quad_num_cat[j].size();//all the keys in this row

    size_t cat_idx = 0;
    for (int j = 0; j < state->cat_attributes; j++) {
      const auto &ordered = state->quad_num_cat[j];//map of categorical column j
      for (auto const& state_val : ordered){//for each key of the cat. variable...
        auto &num_vals = state_val.second;

        //set lin_agg
        cat_set_val_key_lin_cat[idx_element] = state_val.first;
        cat_set_val_val_lin_cat[idx_element] = num_vals[0];
        idx_element++;

        //set num. columns
        for(int k=0; k<num_vals.size()-1; k++) {
          size_t index = cat_idx + (k * cat_offset) + row_offset;
          cat_set_val_key_num_cat[index] = state_val.first;
          cat_set_val_val_num_cat[index] = num_vals[k+1];
        }
        cat_idx++;
      }

      sublist_metadata_lin_cat[sublist_idx_lin_cat].length = ordered.size();
      sublist_metadata_lin_cat[sublist_idx_lin_cat].offset = skipped_lin_cat;
      skipped_lin_cat += ordered.size();
      sublist_idx_lin_cat++;
    }

    for (int k=0; k<state->num_attributes; k++) {
      for (int j = 0; j < state->cat_attributes; j++) {
        sublist_metadata_num_cat[sublist_idx_num_cat].length = state->quad_num_cat[j].size();
        sublist_metadata_num_cat[sublist_idx_num_cat].offset = skipped_num_cat;
        skipped_num_cat += state->quad_num_cat[j].size();
        sublist_idx_num_cat++;
      }
    }

    row_offset += (cat_offset * state->num_attributes);

    result_data3[row_id].length = state->cat_attributes;
    result_data3[row_id].offset = i * result_data3[row_id].length;

    result_data4[row_id].length = state->cat_attributes * state->num_attributes;
    result_data4[row_id].offset = i * result_data4[row_id].length;
  }

  //categorical cat*cat

  int *cat_set_val_key_1 = nullptr;
  int *cat_set_val_key_2 = nullptr;
  float *cat_set_val_val = nullptr;

  if(cat_attributes > 0) {
    duckdb::Vector cat_relations_vector = duckdb::ListVector::GetEntry(c6);
    sublist_metadata_cat_cat = duckdb::ListVector::GetData(cat_relations_vector);
    duckdb::Vector cat_relations_vector_sub = duckdb::ListVector::GetEntry(
        cat_relations_vector);//this is a sequence of values (struct in our case, 2 vectors)
    duckdb::vector<duckdb::unique_ptr<duckdb::Vector>> &lin_struct_vector = duckdb::StructVector::GetEntries(
        cat_relations_vector_sub);
    ((lin_struct_vector)[0])->SetVectorType(duckdb::VectorType::FLAT_VECTOR);
    ((lin_struct_vector)[1])->SetVectorType(duckdb::VectorType::FLAT_VECTOR);
    ((lin_struct_vector)[2])->SetVectorType(duckdb::VectorType::FLAT_VECTOR);
    cat_set_val_key_1 = duckdb::FlatVector::GetData<int>(*((lin_struct_vector)[0]));
    cat_set_val_key_2 = duckdb::FlatVector::GetData<int>(*((lin_struct_vector)[1]));
    cat_set_val_val = duckdb::FlatVector::GetData<float>(*((lin_struct_vector)[2]));
  }

  idx_element = 0;
  idx_t sublist_idx = 0;
  idx_t skipped = 0;

  for (idx_t i = 0; i < count; i++) {
    auto state = states[sdata.sel->get_index(i)];
    const auto row_id = i + offset;

    for (int j = 0; j < state->cat_attributes * (state->cat_attributes+1)/2; j++) {
      //std::map<std::pair<int, int>, float> ordered(state->quad_cat_cat[j].begin(), state->quad_cat_cat[j].end());
      const auto &ordered = state->quad_cat_cat[j];
      for (auto const& state_val : ordered){
        cat_set_val_key_1[idx_element] = state_val.first.first;
        cat_set_val_key_2[idx_element] = state_val.first.second;
        cat_set_val_val[idx_element] = state_val.second;
        idx_element++;
      }
      sublist_metadata_cat_cat[sublist_idx].length = ordered.size();
      sublist_metadata_cat_cat[sublist_idx].offset = skipped;
      skipped += ordered.size();
      sublist_idx++;
    }
    result_data5[row_id].length = state->cat_attributes * (state->cat_attributes+1) / 2;
    result_data5[row_id].offset = i * result_data5[row_id].length;
    //std::cout<<"catcat len: "<<result_data5[row_id].length<<"catcat offs: "<<result_data5[row_id].offset<<std::endl;
  }
  //std::cout<<"end fin"<<std::endl;

}
