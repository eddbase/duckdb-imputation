//
// Created by Massimo Perini on 12/01/2024.
//

#ifndef DUCKDB_SUM_NB_AGG_H
#define DUCKDB_SUM_NB_AGG_H

#include <duckdb.hpp>

namespace Triple {

duckdb::unique_ptr<duckdb::FunctionData> sum_nb_agg_bind(
    duckdb::ClientContext &context, duckdb::AggregateFunction &function,
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> &arguments);

void sum_nb_agg(duckdb::Vector inputs[],
                duckdb::AggregateInputData &aggr_input_data, idx_t input_count,
                duckdb::Vector &state_vector, idx_t count);
}
#endif // DUCKDB_SUM_NB_AGG_H
