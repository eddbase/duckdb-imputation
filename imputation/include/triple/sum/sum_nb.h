

#ifndef DUCKDB_IMPUTATION_SUM_NB_H
#define DUCKDB_IMPUTATION_SUM_NB_H

#include <vector>
#include <duckdb.hpp>
#include <map>
#include <unordered_map>
#include <iostream>

namespace Triple {

    duckdb::unique_ptr<duckdb::FunctionData>
    SumNBBind(duckdb::ClientContext &context, duckdb::AggregateFunction &function,
            duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> &arguments);

    void SumNB(duckdb::Vector inputs[], duckdb::AggregateInputData &aggr_input_data, idx_t input_count,
             duckdb::Vector &state_vector, idx_t count);


    duckdb::Value sum_nb_triple(const duckdb::Value &triple_1, const duckdb::Value &triple_2);

}

#endif //DUCKDB_IMPUTATION_SUM_NB_H
