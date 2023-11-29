
#include <triple/sum/sum_nb_state.h>
#include <duckdb/function/scalar/nested_functions.hpp>


void
Triple::SumNBStateCombine(duckdb::Vector &state, duckdb::Vector &combined, duckdb::AggregateInputData &aggr_input_data,
                        idx_t count) {
    //std::cout<<"\nstart combine\n"<<std::endl;

    duckdb::UnifiedVectorFormat sdata;
    state.ToUnifiedFormat(count, sdata);
    auto states_ptr = (Triple::SumNBState **) sdata.data;

    auto combined_ptr = duckdb::FlatVector::GetData<Triple::SumNBState *>(combined);

    for (idx_t i = 0; i < count; i++) {
        auto state = states_ptr[sdata.sel->get_index(i)];
        combined_ptr[i]->count += state->count;
        if (combined_ptr[i]->lin_agg == nullptr && combined_ptr[i]->quad_num_cat == nullptr) {//init

            combined_ptr[i]->num_attributes = state->num_attributes;
            combined_ptr[i]->cat_attributes = state->cat_attributes;
            if(state->num_attributes > 0) {
                combined_ptr[i]->lin_agg = new float[state->num_attributes*2];
                combined_ptr[i]->quadratic_agg = &(combined_ptr[i]->lin_agg[state->num_attributes]);

                for (idx_t k = 0; k < state->num_attributes; k++)
                    combined_ptr[i]->lin_agg[k] = 0;

                for (idx_t k = 0; k < state->num_attributes; k++)
                    combined_ptr[i]->quadratic_agg[k] = 0;
            }

            if(state->cat_attributes > 0) {
                combined_ptr[i]->quad_num_cat = new boost::container::flat_map<int, int>[state->cat_attributes];
            }
        }

        //SUM NUMERICAL STATES
        for (int j = 0; j < combined_ptr[i]->num_attributes; j++) {
            combined_ptr[i]->lin_agg[j] += state->lin_agg[j];
        }
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
                    taget_map[state_vals.first] = state_vals.second;
                else
                    taget_map[state_vals.first] += state_vals.second;
            }
        }
    }
}

void Triple::SumNBStateFinalize(duckdb::Vector &state_vector, duckdb::AggregateInputData &aggr_input_data, duckdb::Vector &result,
                              idx_t count,
                              idx_t offset) {
    assert(offset == 0);
            D_ASSERT(result.GetType().id() == duckdb::LogicalTypeId::STRUCT);

    duckdb::UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (Triple::SumNBState **) sdata.data;
    auto &children = duckdb::StructVector::GetEntries(result);
    duckdb::Vector &c1 = *(children[0]);//N
    auto input_data = (int32_t *) duckdb::FlatVector::GetData(c1);

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

        duckdb::ListVector::Reserve(c3, (states[sdata.sel->get_index(0)]->num_attributes) * count);
        duckdb::ListVector::SetListSize(c3, (states[sdata.sel->get_index(0)]->num_attributes ) * count);
    }

    c2.SetVectorType(duckdb::VectorType::FLAT_VECTOR);
    auto result_data = duckdb::ListVector::GetData(c2);
    auto num_lin_res = duckdb::FlatVector::GetData<float>(duckdb::ListVector::GetEntry(c2));
    auto num_cat_res = duckdb::FlatVector::GetData<float>(duckdb::ListVector::GetEntry(c3));

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
    for (idx_t i = 0; i < count; i++) {
        auto state = states[sdata.sel->get_index(i)];
        const auto row_id = i + offset;
        for (int j = 0; j < state->num_attributes; j++) {
            num_cat_res[j + (i*state->num_attributes)] = state->quadratic_agg[j];
        }//Value::Numeric
        result_data2[row_id].length = state->num_attributes;
        result_data2[row_id].offset = i * result_data2[i].length;
        //std::cout<<"quad len: "<<result_data2[row_id].length<<"quad offs: "<<result_data2[row_id].offset<<std::endl;
    }

    //categorical sums
    duckdb::Vector &c4 = *(children[3]);
            D_ASSERT(c4.GetType().id() == duckdb::LogicalTypeId::LIST);
    c4.SetVectorType(duckdb::VectorType::FLAT_VECTOR);
    auto result_data3 = duckdb::ListVector::GetData(c4);

    duckdb::list_entry_t *sublist_metadata_lin_cat = nullptr;
    int *cat_set_val_key_lin_cat = nullptr;
    float *cat_set_val_val_lin_cat = nullptr;
    int cat_attributes = 0;
    if (count > 0 && states[sdata.sel->get_index(0)]->cat_attributes > 0) {
        //init list size
        cat_attributes = states[sdata.sel->get_index(0)]->cat_attributes;
        int n_keys_cat_columns = 0;
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
    }

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
        cat_set_val_val_lin_cat = duckdb::FlatVector::GetData<float>(*((lin_struct_vector)[1]));//todo int?
    }
    //categorical num*cat

    idx_t idx_element = 0;
    idx_t sublist_idx_lin_cat = 0;
    idx_t skipped_lin_cat = 0;

    size_t row_offset = 0;
    for (idx_t i = 0; i < count; i++) {
        auto state = states[sdata.sel->get_index(i)];
        const auto row_id = i + offset;
        size_t cat_offset = 0;
        for (int j = 0; j < state->cat_attributes; j++)
            cat_offset += state->quad_num_cat[j].size();//all the keys in this row
        for (int j = 0; j < state->cat_attributes; j++) {
            const auto &ordered = state->quad_num_cat[j];//map of categorical column j
            for (auto const& state_val : ordered){//for each key of the cat. variable...
                //set lin_agg
                cat_set_val_key_lin_cat[idx_element] = state_val.first;
                cat_set_val_val_lin_cat[idx_element] = state_val.second;
                idx_element++;
            }
            sublist_metadata_lin_cat[sublist_idx_lin_cat].length = ordered.size();
            sublist_metadata_lin_cat[sublist_idx_lin_cat].offset = skipped_lin_cat;
            skipped_lin_cat += ordered.size();
            sublist_idx_lin_cat++;
        }
        row_offset += (cat_offset * state->num_attributes);
        result_data3[row_id].length = state->cat_attributes;
        result_data3[row_id].offset = i * result_data3[row_id].length;
    }

}
