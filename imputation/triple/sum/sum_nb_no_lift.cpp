

#include <triple/sum/sum_nb_no_lift.h>
#include <duckdb/function/scalar/nested_functions.hpp>
#include <triple/sum/sum_nb_state.h>
#include <triple/From_duckdb.h>
#include <map>
#include <boost/container/flat_map.hpp>

duckdb::unique_ptr<duckdb::FunctionData>
Triple::SumNBNoLiftBind(duckdb::ClientContext &context, duckdb::AggregateFunction &function,
                      duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> &arguments) {

    child_list_t<LogicalType> struct_children;
    struct_children.emplace_back("N", LogicalType::INTEGER);
    struct_children.emplace_back("lin_agg", LogicalType::LIST(LogicalType::FLOAT));
    struct_children.emplace_back("quad_agg", LogicalType::LIST(LogicalType::FLOAT));

    //categorical structures
    child_list_t<LogicalType> lin_cat;
    lin_cat.emplace_back("key", LogicalType::INTEGER);
    lin_cat.emplace_back("value", LogicalType::FLOAT);

    struct_children.emplace_back("lin_cat", LogicalType::LIST(LogicalType::LIST(LogicalType::STRUCT(lin_cat))));

    auto struct_type = LogicalType::STRUCT(struct_children);
    function.return_type = struct_type;
    //set arguments
    function.varargs = LogicalType::ANY;
    return duckdb::make_uniq<duckdb::VariableReturnBindData>(function.return_type);
}


//Updates the state with a new value. E.g, For SUM, this adds the value to the state. Input count: n. of columns
void Triple::SumNBNoLift(duckdb::Vector inputs[], duckdb::AggregateInputData &aggr_input_data, idx_t cols,
                       duckdb::Vector &state_vector, idx_t count) {
    //count: count of tuples to process
    //inputs: inputs to process

    duckdb::UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);

    auto states = (Triple::SumNBState **) sdata.data;

    size_t num_cols = 0;
    size_t cat_cols = 0;
    UnifiedVectorFormat input_data[cols];
    for (idx_t j=0;j<cols;j++){
        auto col_type = inputs[j].GetType();
        inputs[j].ToUnifiedFormat(count, input_data[j]);
        if (col_type == LogicalType::FLOAT || col_type == LogicalType::DOUBLE)
            num_cols++;
        else
            cat_cols++;
    }

    //SUM N

    for (int j = 0; j < count; j++) {
        auto state = states[sdata.sel->get_index(j)];
        state->count ++;
    }

    //SUM LINEAR AGGREGATES:
    //std::cout<<"Start SUMNOLIFT";

    for (idx_t j = 0; j < count; j++) {
        size_t curr_numerical = 0;
        size_t curr_cateogrical = 0;
        auto state = states[sdata.sel->get_index(j)];
        //initialize
        if (state->lin_agg == nullptr && state->quad_num_cat == nullptr){
            state->num_attributes = num_cols;
            state->cat_attributes = cat_cols;

            if(num_cols > 0) {
                state->lin_agg = new float[num_cols*2];
                state->quadratic_agg = &(state->lin_agg[num_cols]);
                for (idx_t k = 0; k < num_cols; k++)
                    state->lin_agg[k] = 0;
                for (idx_t k = 0; k < num_cols; k++)
                    state->quadratic_agg[k] = 0;
            }

            if(cat_cols > 0) {
                state->quad_num_cat = new boost::container::flat_map<int, int>[cat_cols];
                for(int kk=0; kk<cat_cols; kk++)
                    state->quad_num_cat[kk].reserve(4);
            }
        }


        for(idx_t k=0; k<num_cols; k++) {
            state->lin_agg[k] += UnifiedVectorFormat::GetData<float>(input_data[k])[input_data[k].sel->get_index(j)];
        }
    }


    //SUM QUADRATIC NUMERICAL AGGREGATES:
    int col_idx = 0;

    for(idx_t j=0; j<cols; j++) {
        auto col_type = inputs[j].GetType();
        if (col_type == LogicalType::FLOAT || col_type == LogicalType::DOUBLE) {//numericals
            const float *first_column_data = UnifiedVectorFormat::GetData<float>(input_data[j]);
            for (idx_t i = 0; i < count; i++) {
                auto state = states[sdata.sel->get_index(i)];
                state->quadratic_agg[col_idx] += (first_column_data[input_data[j].sel->get_index(i)] * first_column_data[input_data[j].sel->get_index(i)]);
            }
        }
    }

    //count unique keys in each categorical
    //and allocate appropriate float vector



    //SUM NUM*CAT

    //NUM*CAT
    if (cat_cols > 0) {
        for (idx_t i = 0; i < count; i++) {
            col_idx = 0;
            auto state = states[sdata.sel->get_index(i)];
            for (idx_t j = num_cols; j < cols; j++) {
                const int *sec_column_data = UnifiedVectorFormat::GetData<int>(input_data[j]);
                boost::container::flat_map<int, int> &vals = state->quad_num_cat[col_idx];//map of the column
                int key = sec_column_data[input_data[j].sel->get_index(i)];
                auto pos = vals.find(key);
                if (pos == vals.end()) {
                    vals[key] = 1;
                }
                else {
                    auto &payload = pos->second;
                    payload+=1;
                }
                col_idx++;
            }
        }
    }

}
