

#include <triple/lift.h>
#include <triple/From_duckdb.h>
#include <duckdb/function/scalar/nested_functions.hpp>
#include <duckdb/planner/expression/bound_function_expression.hpp>

#include <unordered_set>

#include <iostream>


namespace Triple_nb {
    //actual implementation of this function
    void CustomLiftNb(duckdb::DataChunk &args, duckdb::ExpressionState &state, duckdb::Vector &result) {
        auto &result_children = duckdb::StructVector::GetEntries(result);
        idx_t size = args.size();//n. of rows to return

        vector<Vector> &in_data = args.data;
        int columns = in_data.size();

        size_t num_cols = 0;
        size_t cat_cols = 0;

        UnifiedVectorFormat input_data[columns];

        for (idx_t j=0;j<columns;j++){
            auto col_type = in_data[j].GetType();
            in_data[j].ToUnifiedFormat(size, input_data[j]);
            if (col_type == LogicalType::FLOAT || col_type == LogicalType::DOUBLE)
                num_cols++;
            else
                cat_cols++;
        }

        //set N
        result_children[0]->SetVectorType(VectorType::FLAT_VECTOR);
        auto N_vec = duckdb::FlatVector::GetData<int32_t>(*result_children[0]);//first child (N)

        for (idx_t i = 0; i < size; i++) {
            N_vec[i] = 1;
        }

        //set linear

        //get result structs for numerical
        duckdb::ListVector::Reserve(*result_children[1], num_cols*size);
        duckdb::ListVector::SetListSize(*result_children[1], num_cols*size);
        auto lin_vec_num = duckdb::FlatVector::GetData<duckdb::list_entry_t>(*result_children[1]);//sec child (lin. aggregates)
        auto lin_vec_num_data = duckdb::FlatVector::GetData<float>(duckdb::ListVector::GetEntry(*result_children[1]));

        duckdb::ListVector::Reserve(*result_children[3], cat_cols*size);
        duckdb::ListVector::SetListSize(*result_children[3], cat_cols*size);

        auto lin_vec_cat = duckdb::FlatVector::GetData<duckdb::list_entry_t>(*result_children[3]);

        //Gets a reference to the underlying child-vector of a list
        //as this is a list of list, another GetEntry needs to be called on cat_relations_vector

        list_entry_t *sublist_metadata = nullptr;
        int *cat_set_val_key = nullptr;
        float *cat_set_val_val = nullptr;

        if (cat_cols>0) {
            Vector cat_relations_vector = duckdb::ListVector::GetEntry(*result_children[3]);
            duckdb::ListVector::Reserve(cat_relations_vector, size*cat_cols);
            duckdb::ListVector::SetListSize(cat_relations_vector, size*cat_cols);
            sublist_metadata = duckdb::ListVector::GetData(cat_relations_vector);
            Vector cat_relations_vector_sub = duckdb::ListVector::GetEntry(cat_relations_vector);//this is a sequence of values (struct in our case, 2 vectors)
            vector<unique_ptr<duckdb::Vector>> &lin_struct_vector = duckdb::StructVector::GetEntries(cat_relations_vector_sub);
            cat_set_val_key = duckdb::FlatVector::GetData<int>(*((lin_struct_vector)[0]));
            cat_set_val_val = duckdb::FlatVector::GetData<float>(*((lin_struct_vector)[1]));
        }

        size_t curr_numerical = 0;
        size_t curr_categorical = 0;

        for (idx_t j=0;j<columns;j++){
            auto col_type = in_data[j].GetType();
            if (col_type == LogicalType::FLOAT || col_type == LogicalType::DOUBLE) {
                const float *column_data = UnifiedVectorFormat::GetData<float>(input_data[j]);// input_data[j].GetData();//input column
                for (idx_t i = 0; i < size; i++) {
                    lin_vec_num_data[curr_numerical + (i * num_cols)] = column_data[input_data[j].sel->get_index(i)];
                }
                curr_numerical++;
            }
            else{//empty relations
                assert(cat_cols>0);
                const int *column_data = UnifiedVectorFormat::GetData<int>(input_data[j]);// (int *)in_data[j].GetData();//input column
                for (idx_t i = 0; i < size; i++) {
                    cat_set_val_key[curr_categorical + (i * cat_cols)] = column_data[input_data[j].sel->get_index(i)];
                    cat_set_val_val[curr_categorical + (i * cat_cols)] = 1;
                    sublist_metadata[(i * cat_cols) + curr_categorical].length = 1;
                    sublist_metadata[(i * cat_cols) + curr_categorical].offset = (i * cat_cols) + curr_categorical;
                }
                curr_categorical++;
            }
        }


        //std::cout<<"cat linear: "<<duckdb::ListVector::GetListSize(*result_children[1])<<"\n";

        //set N*N
        //std::cout<<"N*N"<<std::endl;
        duckdb::ListVector::Reserve(*result_children[2], (num_cols * size));
        duckdb::ListVector::SetListSize(*result_children[2], (num_cols * size));
        auto quad_vec = duckdb::FlatVector::GetData<duckdb::list_entry_t>(*result_children[2]);//sec child (lin. aggregates)
        auto quad_vec_data = duckdb::FlatVector::GetData<float>(duckdb::ListVector::GetEntry(*result_children[2]));

        //numerical * numerical
        int col_idx = 0;
        for (idx_t j=0;j<columns;j++){
            auto col_type = in_data[j].GetType();
            if (col_type == LogicalType::FLOAT || col_type == LogicalType::DOUBLE) {
                const float *column_data = UnifiedVectorFormat::GetData<float>(input_data[j]);//.GetData();//input column
                for (idx_t i = 0; i < size; i++) {
                    quad_vec_data[j] = column_data[input_data[j].sel->get_index(i)]*column_data[input_data[j].sel->get_index(i)];
                }
            }
        }

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
    CustomLiftNbBind(duckdb::ClientContext &context, duckdb::ScalarFunction &function,
                   duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> &arguments) {

        //set return type

        child_list_t<LogicalType> struct_children;
        struct_children.emplace_back("N", LogicalType::INTEGER);
        struct_children.emplace_back("lin_num", LogicalType::LIST(LogicalType::FLOAT));
        struct_children.emplace_back("quad_num", LogicalType::LIST(LogicalType::FLOAT));

        //categorical structures
        child_list_t<LogicalType> lin_cat;
        lin_cat.emplace_back("key", LogicalType::INTEGER);
        lin_cat.emplace_back("value", LogicalType::FLOAT);

        struct_children.emplace_back("lin_cat", LogicalType::LIST(LogicalType::LIST(LogicalType::STRUCT(lin_cat))));
        //lin_cat -> LIST(LIST(STRUCT(key1, key2, val))). E.g. [[{k1,k2,2},{k3,k4,5}],[]]...
        //quad_cat

        auto struct_type = LogicalType::STRUCT(struct_children);
        function.return_type = struct_type;
        //set arguments
        function.varargs = LogicalType::ANY;
        return duckdb::make_uniq<duckdb::VariableReturnBindData>(function.return_type);
    }

    //Generate statistics for this function. Given input type ststistics (mainly min and max for every attribute), returns the output statistics
    duckdb::unique_ptr<duckdb::BaseStatistics>
    CustomLiftNbStats(duckdb::ClientContext &context, duckdb::FunctionStatisticsInput &input) {
        auto &child_stats = input.child_stats;
        auto &expr = input.expr;
        auto struct_stats = duckdb::StructStats::CreateUnknown(expr.return_type);
        return struct_stats.ToUnique();
    }
}
