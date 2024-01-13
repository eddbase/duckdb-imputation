//
// Created by Massimo Perini on 10/01/2024.
//

#ifndef DUCKDB_NAIVE_BAYES_H
#define DUCKDB_NAIVE_BAYES_H

#include <duckdb.hpp>

namespace ML {
void nb_train(duckdb::DataChunk &args, duckdb::ExpressionState &state,
               duckdb::Vector &result);

duckdb::unique_ptr<duckdb::FunctionData> nb_train_bind(
    duckdb::ClientContext &context, duckdb::ScalarFunction &function,
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> &arguments);

void nb_impute(duckdb::DataChunk &args, duckdb::ExpressionState &state,
                duckdb::Vector &result);

duckdb::unique_ptr<duckdb::FunctionData> nb_impute_bind(
    duckdb::ClientContext &context, duckdb::ScalarFunction &function,
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> &arguments);
}


#endif // DUCKDB_NAIVE_BAYES_H
