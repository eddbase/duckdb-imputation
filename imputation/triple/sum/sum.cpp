

#include <triple/sum/sum.h>
#include <triple/From_duckdb.h>
#include <triple/sum/sum_state.h>
#include <duckdb/function/scalar/nested_functions.hpp>

#include <iostream>
#include <memory>
namespace Triple {

    duckdb::unique_ptr<duckdb::FunctionData>
    SumBind(duckdb::ClientContext &context, duckdb::AggregateFunction &function,
                     duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> &arguments) {

        D_ASSERT(arguments.size() == 1);
        D_ASSERT(function.arguments.size() == 1);

        //std::cout<<"SUM ARGS: "<<arguments[0]->ToString();
        //std::cout<<"SUM ARGS 2: "<<function.arguments[0].ToString();


        child_list_t<LogicalType> struct_children;
        struct_children.emplace_back("N", LogicalType::INTEGER);
        struct_children.emplace_back("lin_agg", LogicalType::LIST(LogicalType::FLOAT));
        struct_children.emplace_back("quad_agg", LogicalType::LIST(LogicalType::FLOAT));

        //categorical structures
        child_list_t<LogicalType> lin_cat;
        lin_cat.emplace_back("key", LogicalType::INTEGER);
        lin_cat.emplace_back("value", LogicalType::FLOAT);

        struct_children.emplace_back("lin_cat", LogicalType::LIST(LogicalType::LIST(LogicalType::STRUCT(lin_cat))));

        child_list_t<LogicalType> quad_num_cat;
        quad_num_cat.emplace_back("key", LogicalType::INTEGER);
        quad_num_cat.emplace_back("value", LogicalType::FLOAT);
        struct_children.emplace_back("quad_num_cat", LogicalType::LIST(LogicalType::LIST(LogicalType::STRUCT(quad_num_cat))));

        child_list_t<LogicalType> quad_cat_cat;
        quad_cat_cat.emplace_back("key1", LogicalType::INTEGER);
        quad_cat_cat.emplace_back("key2", LogicalType::INTEGER);
        quad_cat_cat.emplace_back("value", LogicalType::FLOAT);
        struct_children.emplace_back("quad_cat", LogicalType::LIST(LogicalType::LIST(LogicalType::STRUCT(quad_cat_cat))));

        auto struct_type = LogicalType::STRUCT(struct_children);
        function.return_type = struct_type;
        //set arguments
        return duckdb::make_uniq<duckdb::VariableReturnBindData>(function.return_type);
    }


//Updates the state with a new value. E.g, For SUM, this adds the value to the state
    void Sum(duckdb::Vector inputs[], duckdb::AggregateInputData &aggr_input_data, idx_t input_count,
                            duckdb::Vector &state_vector, idx_t count) {
                D_ASSERT(input_count == 1);
        //count: count of tuples to process
        //inputs: inputs to process

        //std::cout << "UPDATE START" << "\nINPUTS: " << inputs->ToString() << "\nINPUT COUNT : "
        //          << input_count << "\nSTATE VECTOR: " << state_vector.ToString() << "\nCOUNT: " << count<<std::endl;

        duckdb::UnifiedVectorFormat sdata;
        state_vector.ToUnifiedFormat(count, sdata);

        auto states = (SumState **) sdata.data;

        auto &input = inputs[0];
        duckdb::RecursiveFlatten(input, count);//not sure it's needed
        D_ASSERT(input.GetVectorType() == VectorType::FLAT_VECTOR);

        //SUM N
        auto &children = StructVector::GetEntries(input);
        Vector &c1 = *children[0];//this should be the vector of integers....

        UnifiedVectorFormat input_data;
        c1.ToUnifiedFormat(count, input_data);
        auto c1_u = UnifiedVectorFormat::GetData<int32_t>(input_data);

        //auto input_data = (int32_t *) FlatVector::GetData(c1);
        //auto state = states[sdata.sel->get_index(0)];//because of parallelism???

        for (int j = 0; j < count; j++) {
            auto state = states[sdata.sel->get_index(j)];
            state->count += c1_u[input_data.sel->get_index(j)];
        }
        //auto N =  input_data;

        //SUM LINEAR AGGREGATES:

        Vector &c2 = *children[1];//this should be the linear aggregate....
        UnifiedVectorFormat input_data_c2;
        duckdb::ListVector::GetEntry(c2).ToUnifiedFormat(count, input_data_c2);
        auto list_entries = UnifiedVectorFormat::GetData<float>(input_data_c2);

        auto num_attr_size = ListVector::GetListSize(c2);
        auto cat_attr_size = ListVector::GetListSize(*children[3]);

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
                    state->lin_agg = new float[num_attr + ((num_attr * (num_attr + 1)) / 2)];
                    state->quadratic_agg = &(state->lin_agg[num_attr]);
                    for(idx_t k=0; k<num_attr; k++)
                        state->lin_agg[k] = 0;

                    for(idx_t k=0; k<(num_attr*(num_attr+1))/2; k++)
                        state->quadratic_agg[k] = 0;
                }
                if(cat_attr > 0) {
                    state->quad_num_cat = new boost::container::flat_map<int, std::vector<float>>[cat_attr];

                    state->quad_cat_cat = new boost::container::flat_map<std::pair<int, int>, float>[
                    cat_attr * (cat_attr + 1) / 2];
                }
            }

            for(idx_t k=0; k<num_attr; k++) {
                state->lin_agg[k] += list_entries[input_data_c2.sel->get_index(k + (j * num_attr))];
            }
        }

        //SUM QUADRATIC AGGREGATES:
        Vector &c3 = *children[2];//this should be the linear aggregate....
        UnifiedVectorFormat input_data_c3;
        duckdb::ListVector::GetEntry(c3).ToUnifiedFormat(count, input_data_c3);
        auto list_entries_2 = UnifiedVectorFormat::GetData<float>(input_data_c3);

        int cols2 = ((num_attr * (num_attr + 1)) / 2);//quadratic aggregates columns

        for (idx_t j = 0; j < count; j++) {
            auto state = states[sdata.sel->get_index(j)];
            for(idx_t k=0; k<cols2; k++)
                state->quadratic_agg[k] += list_entries_2[input_data_c3.sel->get_index(k + (j*cols2))];
        }

        //sum cat lin. data

        //Vector v_cat_lin = duckdb::ListVector::GetEntry(*children[3]);

        Vector &c4 = ListVector::GetEntry(*children[3]);//this should be the linear aggregate....
        //UnifiedVectorFormat input_data_c4;
        //duckdb::ListVector::GetEntry(c4).ToUnifiedFormat(count, input_data_c4);
        //auto list_entries_2 = UnifiedVectorFormat::GetData<float>(input_data_c3);
        //UnifiedVectorFormat::Get

        int *cat_set_val_key_lin_cat = nullptr;
        float *cat_set_val_val_lin_cat = nullptr;
        int *cat_set_val_key_num_cat = nullptr;
        float *cat_set_val_val_num_cat = nullptr;
        list_entry_t *sublist_metadata_lin_cat = nullptr;
        list_entry_t *sublist_metadata_num_cat = nullptr;

        if(cat_attr > 0){
            Vector v_cat_lin_1 = duckdb::ListVector::GetEntry(*children[3]);
            sublist_metadata_lin_cat = duckdb::ListVector::GetData(v_cat_lin_1);
            Vector v_cat_lin_1_values = duckdb::ListVector::GetEntry(v_cat_lin_1);
            vector<unique_ptr<duckdb::Vector>> &v_cat_lin_1_struct = duckdb::StructVector::GetEntries(
                    v_cat_lin_1_values);
            //todo flatvector assumption
            cat_set_val_key_lin_cat = duckdb::FlatVector::GetData<int>(*((v_cat_lin_1_struct)[0]));
            cat_set_val_val_lin_cat = duckdb::FlatVector::GetData<float>(*((v_cat_lin_1_struct)[1]));

        }

        //set num*cat
        Vector &c5 = ListVector::GetEntry(*children[4]);

        if(cat_attr > 0){
            Vector v_cat_lin_1 = duckdb::ListVector::GetEntry(*children[4]);
            sublist_metadata_num_cat = duckdb::ListVector::GetData(v_cat_lin_1);
            Vector v_cat_lin_1_values = duckdb::ListVector::GetEntry(v_cat_lin_1);
            vector<unique_ptr<duckdb::Vector>> &v_cat_lin_1_struct = duckdb::StructVector::GetEntries(
                    v_cat_lin_1_values);
            //todo flatvector assumption
            cat_set_val_key_num_cat = duckdb::FlatVector::GetData<int>(*((v_cat_lin_1_struct)[0]));
            cat_set_val_val_num_cat = duckdb::FlatVector::GetData<float>(*((v_cat_lin_1_struct)[1]));
        }

        float sum_vals[num_attr+1];//sum of values for key

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
                    for(idx_t l=0; l<num_attr; l++){//same categorical, new numerical
                        auto curr_metadata = sublist_metadata_num_cat[(l*cat_attr) + k + (j*(cat_attr * num_attr))];
                        assert(cat_set_val_key_num_cat[key_idx+curr_metadata.offset] == key);
                        assert(curr_metadata.length == curr_metadata_lin_cat.length);
                        sum_vals[l+1] = cat_set_val_val_num_cat[key_idx+curr_metadata.offset];
                    }
                    auto pos = col_vals.find(key);
                    if (pos == col_vals.end()) {
                        col_vals[key] = std::vector(sum_vals, sum_vals+num_attr+1);
                    }
                    else {
                        for(idx_t l=0; l<num_attr+1; l++)
                            pos->second[l] += sum_vals[l];
                    }
                }
            }
        }


        Vector &c6 = ListVector::GetEntry(*children[5]);
        int *cat_set_val_key_1 = nullptr;
        int *cat_set_val_key_2 = nullptr;
        float *cat_set_val_val = nullptr;
        list_entry_t *sublist_metadata;
        if(cat_attr > 0){
            Vector v_cat_lin_1 = duckdb::ListVector::GetEntry(*children[5]);
            sublist_metadata = duckdb::ListVector::GetData(v_cat_lin_1);
            Vector v_cat_lin_1_values = duckdb::ListVector::GetEntry(v_cat_lin_1);
            vector<unique_ptr<duckdb::Vector>> &v_cat_lin_1_struct = duckdb::StructVector::GetEntries(
                    v_cat_lin_1_values);
            //todo flatvector assumption
            cat_set_val_key_1 = duckdb::FlatVector::GetData<int>(*((v_cat_lin_1_struct)[0]));
            cat_set_val_key_2 = duckdb::FlatVector::GetData<int>(*((v_cat_lin_1_struct)[1]));
            cat_set_val_val = duckdb::FlatVector::GetData<float>(*((v_cat_lin_1_struct)[2]));
        }

        for (idx_t j = 0; j < count; j++) {
            auto state = states[sdata.sel->get_index(j)];
            for(idx_t k=0; k<(cat_attr * (cat_attr+1))/2; k++){
                auto curr_metadata = sublist_metadata[k + (j*(cat_attr * (cat_attr+1))/2)];
                auto &col_vals = state->quad_cat_cat[k];
                for (size_t item = 0; item < curr_metadata.length; item++){
                    std::pair<int, int> key = std::pair<int, int>(cat_set_val_key_1[item + curr_metadata.offset], cat_set_val_key_2[item + curr_metadata.offset]);
                    auto pos = col_vals.find(key);
                    if (pos == col_vals.end())
                        col_vals[key] = cat_set_val_val[item + curr_metadata.offset];
                    else
                        pos->second += cat_set_val_val[item + curr_metadata.offset];
                }
            }
        }
    }

    vector<Value> sum_list_of_structs(const vector<Value> &v1, const vector<Value> &v2){
        std::map<int, float> content;
        vector<Value> col_cat_lin = {};
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
            child_list_t<Value> struct_values;
            struct_values.emplace_back("key", Value(value.first));
            struct_values.emplace_back("value", Value(value.second));
            col_cat_lin.push_back(duckdb::Value::STRUCT(struct_values));
        }
        return col_cat_lin;
    }

    vector<Value> sum_list_of_structs_key2(const vector<Value> &v1, const vector<Value> &v2){
        std::map<std::pair<int, int>, float> content;
        vector<Value> col_cat_lin = {};
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
            child_list_t<Value> struct_values;
            struct_values.emplace_back("key1", Value(value.first.first));
            struct_values.emplace_back("key2", Value(value.first.second));
            struct_values.emplace_back("value", Value(value.second));
            col_cat_lin.push_back(duckdb::Value::STRUCT(struct_values));
        }
        return col_cat_lin;
    }

    duckdb::Value sum_triple(const duckdb::Value &triple_1, const duckdb::Value &triple_2){

        auto first_triple_children = duckdb::StructValue::GetChildren(triple_1);//vector of pointers to childrens
        auto sec_triple_children = duckdb::StructValue::GetChildren(triple_2);

        auto N_1 = first_triple_children[0].GetValue<int>();;
        auto N_2 = sec_triple_children[0].GetValue<int>();

        child_list_t<Value> struct_values;
        struct_values.emplace_back("N", Value(N_1 + N_2));

        const vector<Value> &linear_1 = duckdb::ListValue::GetChildren(first_triple_children[1]);
        const vector<Value> &linear_2 = duckdb::ListValue::GetChildren(sec_triple_children[1]);
        vector<Value> lin = {};
        if (!linear_1.empty() && !linear_2.empty()) {
            for (idx_t i = 0; i < linear_1.size(); i++)
                lin.push_back(Value(linear_1[i].GetValue<float>() + linear_2[i].GetValue<float>()));
        }
        else if (!linear_1.empty()){
            for (idx_t i = 0; i < linear_1.size(); i++)
                lin.push_back(Value(linear_1[i].GetValue<float>()));
        }
        else if (!linear_2.empty()){
            for (idx_t i = 0; i < linear_2.size(); i++)
                lin.push_back(Value(linear_2[i].GetValue<float>()));
        }

        struct_values.emplace_back("lin_num", duckdb::Value::LIST(LogicalType::FLOAT, lin));

        const vector<Value> &quad_1 = duckdb::ListValue::GetChildren(first_triple_children[2]);
        const vector<Value> &quad_2 = duckdb::ListValue::GetChildren(sec_triple_children[2]);
        vector<Value> quad = {};

        if (!quad_1.empty() && !quad_2.empty()) {
            for(idx_t i=0;i<quad_1.size();i++)
                quad.push_back(Value(quad_1[i].GetValue<float>() + quad_2[i].GetValue<float>()));
        }
        else if (!quad_1.empty()){
            for (idx_t i = 0; i < quad_1.size(); i++)
                quad.push_back(Value(quad_1[i].GetValue<float>()));
        }
        else if (!quad_2.empty()){
            for (idx_t i = 0; i < quad_2.size(); i++)
                quad.push_back(Value(quad_2[i].GetValue<float>()));
        }

        struct_values.emplace_back("quad_num", duckdb::Value::LIST(LogicalType::FLOAT, quad));

        //categorical linear
        const vector<Value> &cat_linear_1 = duckdb::ListValue::GetChildren(first_triple_children[3]);
        const vector<Value> &cat_linear_2 = duckdb::ListValue::GetChildren(sec_triple_children[3]);
        vector<Value> cat_lin = {};
        if (!cat_linear_1.empty() && !cat_linear_2.empty()) {
            for (idx_t i = 0; i < cat_linear_1.size(); i++) {//for each cat. column copy into map
                const vector<Value> &pairs_cat_col_1 = duckdb::ListValue::GetChildren(cat_linear_1[i]);
                const vector<Value> &pairs_cat_col_2 = duckdb::ListValue::GetChildren(cat_linear_2[i]);
                cat_lin.push_back(duckdb::Value::LIST(sum_list_of_structs(pairs_cat_col_1, pairs_cat_col_2)));
            }
        }
        else if (!cat_linear_1.empty()){
            for (idx_t i = 0; i < cat_linear_1.size(); i++) {//for each cat. column copy into map
                const vector<Value> &pairs_cat_col_1 = duckdb::ListValue::GetChildren(cat_linear_1[i]);
                cat_lin.push_back(duckdb::Value::LIST(pairs_cat_col_1));
            }
        }
        else if (!cat_linear_2.empty()){
            for (idx_t i = 0; i < cat_linear_2.size(); i++) {//for each cat. column copy into map
                const vector<Value> &pairs_cat_col_1 = duckdb::ListValue::GetChildren(cat_linear_2[i]);
                cat_lin.push_back(duckdb::Value::LIST(pairs_cat_col_1));
            }
        }

        child_list_t<LogicalType> struct_values_l;
        struct_values_l.emplace_back("key", LogicalType::INTEGER);
        struct_values_l.emplace_back("value", LogicalType::FLOAT);

        struct_values.emplace_back("lin_cat", duckdb::Value::LIST(LogicalType::LIST(LogicalType::STRUCT(struct_values_l)),cat_lin));

        //num*cat

        const vector<Value> &num_cat_1 = duckdb::ListValue::GetChildren(first_triple_children[4]);
        const vector<Value> &num_cat_2 = duckdb::ListValue::GetChildren(sec_triple_children[4]);
        vector<Value> num_cat_res = {};

        if (!num_cat_1.empty() && !num_cat_2.empty()) {
            for(idx_t i=0;i<num_cat_1.size();i++) {//for each cat. column copy into map
                const vector<Value> &pairs_cat_col_1 = duckdb::ListValue::GetChildren(num_cat_1[i]);
                const vector<Value> &pairs_cat_col_2 = duckdb::ListValue::GetChildren(num_cat_2[i]);
                num_cat_res.push_back(duckdb::Value::LIST(sum_list_of_structs(pairs_cat_col_1, pairs_cat_col_2)));
            }
        }
        else if (!num_cat_1.empty()){
            for(idx_t i=0;i<num_cat_1.size();i++) {//for each cat. column copy into map
                const vector<Value> &pairs_cat_col_1 = duckdb::ListValue::GetChildren(num_cat_1[i]);
                num_cat_res.push_back(duckdb::Value::LIST(pairs_cat_col_1));
            }
        }
        else if (!num_cat_2.empty()){
            for(idx_t i=0;i<num_cat_2.size();i++) {//for each cat. column copy into map
                const vector<Value> &pairs_cat_col_2 = duckdb::ListValue::GetChildren(num_cat_2[i]);
                num_cat_res.push_back(duckdb::Value::LIST(pairs_cat_col_2));
            }
        }

        struct_values.emplace_back("quad_num_cat", duckdb::Value::LIST(LogicalType::LIST(LogicalType::STRUCT(struct_values_l)),num_cat_res));

        //cat*cat

        const vector<Value> &cat_cat_1 = duckdb::ListValue::GetChildren(first_triple_children[5]);
        const vector<Value> &cat_cat_2 = duckdb::ListValue::GetChildren(sec_triple_children[5]);
        vector<Value> cat_cat_res = {};

        if (!cat_cat_1.empty() && !cat_cat_2.empty()) {
            for(idx_t i=0;i<cat_cat_1.size();i++) {//for each cat. column copy into map
                const vector<Value> &pairs_cat_col_1 = duckdb::ListValue::GetChildren(cat_cat_1[i]);
                const vector<Value> &pairs_cat_col_2 = duckdb::ListValue::GetChildren(cat_cat_2[i]);
                cat_cat_res.push_back(duckdb::Value::LIST(sum_list_of_structs_key2(pairs_cat_col_1, pairs_cat_col_2)));
            }
        }
        else if (!cat_cat_1.empty()){
            for(idx_t i=0;i<cat_cat_1.size();i++) {//for each cat. column copy into map
                const vector<Value> &pairs_cat_col_1 = duckdb::ListValue::GetChildren(cat_cat_1[i]);
                cat_cat_res.push_back(duckdb::Value::LIST(pairs_cat_col_1));
            }
        }
        else if (!cat_cat_2.empty()){
            for(idx_t i=0;i<cat_cat_2.size();i++) {//for each cat. column copy into map
                const vector<Value> &pairs_cat_col_2 = duckdb::ListValue::GetChildren(cat_cat_2[i]);
                cat_cat_res.push_back(duckdb::Value::LIST(pairs_cat_col_2));
            }
        }

        child_list_t<LogicalType> struct_values_l2;
        struct_values_l2.emplace_back("key1", LogicalType::INTEGER);
        struct_values_l2.emplace_back("key2", LogicalType::INTEGER);
        struct_values_l2.emplace_back("value", LogicalType::FLOAT);

        struct_values.emplace_back("quad_cat", duckdb::Value::LIST(LogicalType::LIST(LogicalType::STRUCT(struct_values_l2)), cat_cat_res));

        auto ret = duckdb::Value::STRUCT(struct_values);
        return ret;
    }
}
