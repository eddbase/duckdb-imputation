//
// Created by Massimo Perini on 10/01/2024.
//

#ifndef DUCKDB_QDA_H
#define DUCKDB_QDA_H

#include <duckdb.hpp>

namespace ML {
void qda_train(duckdb::DataChunk &args, duckdb::ExpressionState &state,
               duckdb::Vector &result);

duckdb::unique_ptr<duckdb::FunctionData> qda_train_bind(
    duckdb::ClientContext &context, duckdb::ScalarFunction &function,
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> &arguments);

void qda_impute(duckdb::DataChunk &args, duckdb::ExpressionState &state,
                duckdb::Vector &result);

duckdb::unique_ptr<duckdb::FunctionData> qda_impute_bind(
    duckdb::ClientContext &context, duckdb::ScalarFunction &function,
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> &arguments);
}


#endif // DUCKDB_QDA_H
