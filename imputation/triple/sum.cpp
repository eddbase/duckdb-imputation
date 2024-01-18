

#include <sum_sub.h>
#include <duckdb/function/scalar/nested_functions.hpp>

#include <iostream>
#include <memory>

using namespace duckdb;

namespace Triple {
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
