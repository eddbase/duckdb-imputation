

#include <triple/sub.h>
#include <duckdb/function/scalar/nested_functions.hpp>


#include <triple/mul.h>

#include <iostream>
#include <algorithm>
#include <triple/From_duckdb.h>

//DataChunk is a set of vectors

class duckdb::BoundFunctionExpression : public duckdb::Expression {
public:
    static constexpr const ExpressionClass TYPE = ExpressionClass::BOUND_FUNCTION;

public:
    BoundFunctionExpression(LogicalType return_type, ScalarFunction bound_function,
                            vector<unique_ptr<Expression>> arguments, unique_ptr<FunctionData> bind_info,
                            bool is_operator = false);

    //! The bound function expression
    ScalarFunction function;
    //! List of child-expressions of the function
    vector<unique_ptr<Expression>> children;
    //! The bound function data (if any)
    unique_ptr<FunctionData> bind_info;
    //! Whether or not the function is an operator, only used for rendering
    bool is_operator;

public:

    bool HasSideEffects() const override;
    bool IsFoldable() const override;
    string ToString() const override;
    bool PropagatesNullValues() const override;
    hash_t Hash() const override;
    bool Equals(const BaseExpression *other) const;

    unique_ptr<Expression> Copy() override;
    void Verify() const override;

    //void Serialize(FieldWriter &writer) const override;
    //static unique_ptr<Expression> Deserialize(ExpressionDeserializationState &state, FieldReader &reader);
};

namespace duckdb {
    struct ListStats {
        DUCKDB_API static void Construct(BaseStatistics &stats);

        DUCKDB_API static BaseStatistics CreateUnknown(LogicalType type);

        DUCKDB_API static BaseStatistics CreateEmpty(LogicalType type);

        DUCKDB_API static const BaseStatistics &GetChildStats(const BaseStatistics &stats);

        DUCKDB_API static BaseStatistics &GetChildStats(BaseStatistics &stats);

        DUCKDB_API static void SetChildStats(BaseStatistics &stats, unique_ptr<BaseStatistics> new_stats);

        //DUCKDB_API static void Serialize(const BaseStatistics &stats, FieldWriter &writer);

        //DUCKDB_API static BaseStatistics Deserialize(FieldReader &reader, LogicalType type);

        DUCKDB_API static string ToString(const BaseStatistics &stats);

        DUCKDB_API static void Merge(BaseStatistics &stats, const BaseStatistics &other);

        DUCKDB_API static void Copy(BaseStatistics &stats, const BaseStatistics &other);

        DUCKDB_API static void
        Verify(const BaseStatistics &stats, Vector &vector, const SelectionVector &sel, idx_t count);
    };
}

namespace Triple {
    //actual implementation of this function
    void TripleSubFunction(duckdb::DataChunk &args, duckdb::ExpressionState &state, duckdb::Vector &result) {
        //std::cout << "StructPackFunction start " << std::endl;
        auto &result_children = duckdb::StructVector::GetEntries(result);

        idx_t size = args.size();//n. of rows to return
        //std::cout<<"SIZE: "<<size;

        auto &first_triple_children = duckdb::StructVector::GetEntries(args.data[0]);//vector of pointers to childrens
        auto &sec_triple_children = duckdb::StructVector::GetEntries(args.data[1]);

        //set N

        RecursiveFlatten(*first_triple_children[0], size);
                D_ASSERT((*first_triple_children[0]).GetVectorType() == duckdb::VectorType::FLAT_VECTOR);
        RecursiveFlatten(*sec_triple_children[0], size);
                D_ASSERT((*sec_triple_children[0]).GetVectorType() == duckdb::VectorType::FLAT_VECTOR);


        auto N_1 = (int32_t *) duckdb::FlatVector::GetData(*first_triple_children[0]);
        auto N_2 = (int32_t *) duckdb::FlatVector::GetData(*sec_triple_children[0]);
        auto input_data = (int32_t *) duckdb::FlatVector::GetData(*result_children[0]);

        for (idx_t i = 0; i < size; i++) {
            //std::cout<<"New N: "<<N_1[i] - N_2[i]<<std::endl;
            input_data[i] = N_1[i] - N_2[i];
        }
        //set linear aggregates

        RecursiveFlatten(*first_triple_children[1], size);
                D_ASSERT((*first_triple_children[1]).GetVectorType() == duckdb::VectorType::FLAT_VECTOR);
        RecursiveFlatten(*sec_triple_children[1], size);
                D_ASSERT((*sec_triple_children[1]).GetVectorType() == duckdb::VectorType::FLAT_VECTOR);

        auto attr_size_1 = duckdb::ListVector::GetListSize(*first_triple_children[1]) / size;
        auto attr_size_2 = duckdb::ListVector::GetListSize(*sec_triple_children[1]) / size;
        D_ASSERT(attr_size_1 == attr_size_2);

        auto lin_list_entries_1 = (float *) duckdb::ListVector::GetEntry(
                *first_triple_children[1]).GetData();//entries are float
        auto lin_list_entries_2 = (float *) duckdb::ListVector::GetEntry(
                *sec_triple_children[1]).GetData();//entries are float


        (*result_children[1]).SetVectorType(duckdb::VectorType::FLAT_VECTOR);
        auto result_data = duckdb::FlatVector::GetData<duckdb::list_entry_t>(*result_children[1]);

        //creates a single vector
        for (idx_t i = 0; i < size; i++) {
            for (idx_t j = 0; j < attr_size_1; j++) {
                //std::cout<<"New LINEAR: "<<lin_list_entries_1[j + (i * attr_size_1)]<<std::endl;
                duckdb::ListVector::PushBack(*result_children[1], duckdb::Value(
                        lin_list_entries_1[j + (i * attr_size_1)] - lin_list_entries_2[j + (i * attr_size_1)]));
            }
        }

        //set for each row to return the metadata (view of the full vector)
        for (idx_t child_idx = 0; child_idx < size; child_idx++) {
            result_data[child_idx].length = attr_size_1;
            result_data[child_idx].offset =
                    child_idx * result_data[child_idx].length;//ListVector::GetListSize(*result_children[1]);
        }

        //set quadratic aggregates

        RecursiveFlatten(*first_triple_children[2], size);
                D_ASSERT((*first_triple_children[2]).GetVectorType() == duckdb::VectorType::FLAT_VECTOR);
        RecursiveFlatten(*sec_triple_children[2], size);
                D_ASSERT((*sec_triple_children[2]).GetVectorType() == duckdb::VectorType::FLAT_VECTOR);

        auto quad_lists_size_1 = duckdb::ListVector::GetListSize(*first_triple_children[2]) / size;
        auto quad_lists_size_2 = duckdb::ListVector::GetListSize(*sec_triple_children[2]) / size;

                D_ASSERT(quad_lists_size_1 == quad_lists_size_2);

        auto quad_list_entries_1 = (float *) duckdb::ListVector::GetEntry(
                *first_triple_children[2]).GetData();//entries are float
        auto quad_list_entries_2 = (float *) duckdb::ListVector::GetEntry(
                *sec_triple_children[2]).GetData();//entries are float

        (*result_children[2]).SetVectorType(duckdb::VectorType::FLAT_VECTOR);
        result_data = duckdb::FlatVector::GetData<duckdb::list_entry_t>(*result_children[2]);

        //for each row
        for (idx_t i = 0; i < size; i++) {
            for (idx_t j = 0; j < quad_lists_size_1; j++) {
                //std::cout<<"New QUADRATIC: "<<quad_list_entries_1[j + (i*quad_lists_size_1)] - quad_list_entries_2[j + (i*quad_lists_size_1)]<<std::endl;
                duckdb::ListVector::PushBack(*result_children[2], duckdb::Value(quad_list_entries_1[j + (i*quad_lists_size_1)] - quad_list_entries_2[j + (i*quad_lists_size_1)]));
            }
        }

        //set for each row to return the metadata (view of the full vector)
        for (idx_t child_idx = 0; child_idx < size; child_idx++) {
            result_data[child_idx].length = quad_lists_size_1;
            result_data[child_idx].offset = child_idx * result_data[child_idx].length;
        }

        //std::cout << "StructPackFunction end " << std::endl;
    }

    //Returns the datatype used by this function
    duckdb::unique_ptr<duckdb::FunctionData>
    TripleSubBind(duckdb::ClientContext &context, duckdb::ScalarFunction &function,
                 duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> &arguments) {

                D_ASSERT(arguments.size() == 2);


        child_list_t<LogicalType> struct_children;
        struct_children.emplace_back("N", LogicalType::INTEGER);
        struct_children.emplace_back("lin_agg", LogicalType::LIST(LogicalType::FLOAT));
        struct_children.emplace_back("quad_agg", LogicalType::LIST(LogicalType::FLOAT));

        auto struct_type = LogicalType::STRUCT(struct_children);
        function.return_type = struct_type;
        //set arguments
        function.varargs = struct_type;

        return duckdb::make_uniq<duckdb::VariableReturnBindData>(function.return_type);
    }

    //Generate statistics for this function. Given input type ststistics (mainly min and max for every attribute), returns the output statistics
    duckdb::unique_ptr<duckdb::BaseStatistics>
    TripleSubStats(duckdb::ClientContext &context, duckdb::FunctionStatisticsInput &input) {
        auto &child_stats = input.child_stats;
        auto &expr = input.expr;
        auto struct_stats = duckdb::StructStats::CreateUnknown(expr.return_type);
        return struct_stats.ToUnique();
    }


    vector<Value> sub_list_of_structs(const vector<Value> &v1, const vector<Value> &v2){
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
                std::cout<<"Error, key is not present in first triple\n";
            else
                pos->second -= children[1].GetValue<float>();
        }
        for (auto const& value : content){
            child_list_t<Value> struct_values;
            struct_values.emplace_back("key", Value(value.first));
            struct_values.emplace_back("value", Value(value.second));
            col_cat_lin.push_back(duckdb::Value::STRUCT(struct_values));
        }
        return col_cat_lin;
    }

    vector<Value> sub_list_of_structs_key2(const vector<Value> &v1, const vector<Value> &v2){
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
                std::cout<<"Error, key is not present in first triple\n";
            else
                pos->second -= children[2].GetValue<float>();
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

    duckdb::Value subtract_triple(duckdb::Value &triple_1, duckdb::Value &triple_2){

        auto first_triple_children = duckdb::StructValue::GetChildren(triple_1);//vector of pointers to childrens
        auto sec_triple_children = duckdb::StructValue::GetChildren(triple_2);

        auto N_1 = (int) first_triple_children[0].GetValue<int>();;
        auto N_2 = (int) sec_triple_children[0].GetValue<int>();

        child_list_t<Value> struct_values;
        struct_values.emplace_back("N", Value(N_1 - N_2));

        const vector<Value> &linear_1 = duckdb::ListValue::GetChildren(first_triple_children[1]);
        const vector<Value> &linear_2 = duckdb::ListValue::GetChildren(sec_triple_children[1]);
        //first_triple_children[1].Print();
        //sec_triple_children[1].Print();
        vector<Value> lin = {};

        if (!linear_1.empty() && !linear_2.empty()) {
            for(idx_t i=0;i<linear_1.size();i++)
                lin.push_back(Value(linear_1[i].GetValue<float>() - linear_2[i].GetValue<float>()));
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


        if (!quad_1.empty() && !quad_2.empty() > 0) {
            for(idx_t i=0;i<quad_1.size();i++)
                quad.push_back(Value(quad_1[i].GetValue<float>() - quad_2[i].GetValue<float>()));
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

        if (!cat_linear_1.empty() && !cat_linear_2.empty() > 0) {
            for(idx_t i=0;i<cat_linear_1.size();i++) {//for each cat. column copy into map
                const vector<Value> &pairs_cat_col_1 = duckdb::ListValue::GetChildren(cat_linear_1[i]);
                const vector<Value> &pairs_cat_col_2 = duckdb::ListValue::GetChildren(cat_linear_2[i]);
                cat_lin.push_back(duckdb::Value::LIST(sub_list_of_structs(pairs_cat_col_1, pairs_cat_col_2)));
            }
        }
        else if (!cat_linear_1.empty()){
            for (idx_t i = 0; i < cat_linear_1.size(); i++)
                cat_lin.push_back(Value(cat_linear_1[i].GetValue<float>()));
        }
        else if (!cat_linear_2.empty()){
            for (idx_t i = 0; i < cat_linear_2.size(); i++)
                cat_lin.push_back(Value(cat_linear_2[i].GetValue<float>()));
        }


        child_list_t<LogicalType> struct_values_l;
        struct_values_l.emplace_back("key", LogicalType::INTEGER);
        struct_values_l.emplace_back("value", LogicalType::FLOAT);

        struct_values.emplace_back("lin_cat", duckdb::Value::LIST(LogicalType::LIST(LogicalType::STRUCT(struct_values_l)), cat_lin));

        //num*cat

        const vector<Value> &num_cat_1 = duckdb::ListValue::GetChildren(first_triple_children[4]);
        const vector<Value> &num_cat_2 = duckdb::ListValue::GetChildren(sec_triple_children[4]);
        vector<Value> num_cat_res = {};

        if (!num_cat_1.empty() && !num_cat_2.empty() > 0) {
            for(idx_t i=0;i<num_cat_1.size();i++) {//for each cat. column copy into map
                const vector<Value> &pairs_cat_col_1 = duckdb::ListValue::GetChildren(num_cat_1[i]);
                const vector<Value> &pairs_cat_col_2 = duckdb::ListValue::GetChildren(num_cat_2[i]);
                num_cat_res.push_back(duckdb::Value::LIST(sub_list_of_structs(pairs_cat_col_1, pairs_cat_col_2)));
            }
        }
        else if (!num_cat_1.empty()){
            for (idx_t i = 0; i < num_cat_1.size(); i++) {//for each cat. column copy into map
                const vector<Value> &pairs_cat_col_1 = duckdb::ListValue::GetChildren(cat_linear_1[i]);
                num_cat_res.push_back(duckdb::Value::LIST(pairs_cat_col_1));
            }
        }
        else if (!num_cat_2.empty()){
            for (idx_t i = 0; i < num_cat_2.size(); i++) {//for each cat. column copy into map
                const vector<Value> &pairs_cat_col_1 = duckdb::ListValue::GetChildren(cat_linear_2[i]);
                num_cat_res.push_back(duckdb::Value::LIST(pairs_cat_col_1));
            }
        }

        struct_values.emplace_back("quad_num_cat", duckdb::Value::LIST(LogicalType::LIST(LogicalType::STRUCT(struct_values_l)), num_cat_res));

        //cat*cat

        const vector<Value> &cat_cat_1 = duckdb::ListValue::GetChildren(first_triple_children[5]);
        const vector<Value> &cat_cat_2 = duckdb::ListValue::GetChildren(sec_triple_children[5]);
        vector<Value> cat_cat_res = {};

        if (!cat_cat_1.empty() && !cat_cat_2.empty() > 0) {
            for(idx_t i=0;i<cat_cat_1.size();i++) {//for each cat. column copy into map
                const vector<Value> &pairs_cat_col_1 = duckdb::ListValue::GetChildren(cat_cat_1[i]);
                const vector<Value> &pairs_cat_col_2 = duckdb::ListValue::GetChildren(cat_cat_2[i]);
                cat_cat_res.push_back(duckdb::Value::LIST(sub_list_of_structs_key2(pairs_cat_col_1, pairs_cat_col_2)));
            }
        }
        else if (!cat_cat_1.empty()){
            for (idx_t i = 0; i < cat_cat_1.size(); i++) {//for each cat. column copy into map
                const vector<Value> &pairs_cat_col_1 = duckdb::ListValue::GetChildren(cat_linear_1[i]);
                cat_cat_res.push_back(duckdb::Value::LIST(pairs_cat_col_1));
            }
        }
        else if (!cat_cat_2.empty()){
            for (idx_t i = 0; i < cat_cat_2.size(); i++) {//for each cat. column copy into map
                const vector<Value> &pairs_cat_col_1 = duckdb::ListValue::GetChildren(cat_linear_2[i]);
                cat_cat_res.push_back(duckdb::Value::LIST(pairs_cat_col_1));
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
