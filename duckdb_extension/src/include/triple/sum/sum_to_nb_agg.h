//
// Created by Massimo Perini on 12/01/2024.
//

#ifndef DUCKDB_SUM_TO_NB_AGG_H
#define DUCKDB_SUM_TO_NB_AGG_H

namespace Triple {
duckdb::unique_ptr<duckdb::FunctionData> sum_to_nb_agg_bind(
    duckdb::ClientContext &context, duckdb::AggregateFunction &function,
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> &arguments);

void sum_to_nb_agg(duckdb::Vector inputs[],
                   duckdb::AggregateInputData &aggr_input_data,
                   duckdb::idx_t cols, duckdb::Vector &state_vector,
                   duckdb::idx_t count);
}
#endif // DUCKDB_SUM_TO_NB_AGG_H
