//
// Created by Massimo Perini on 10/01/2024.
//

#ifndef DUCKDB_REGRESSION_H
#define DUCKDB_REGRESSION_H

#include "duckdb.hpp"

struct RegressionState : public duckdb::FunctionLocalState {
  explicit RegressionState(){
    random_seed_set = false;
    seed = 0;
  }
  bool random_seed_set;
  unsigned long seed;
};


namespace ML {

void ridge_linear_regression(duckdb::DataChunk &args,
                             duckdb::ExpressionState &state,
                             duckdb::Vector &result);

duckdb::unique_ptr<duckdb::FunctionData> ridge_linear_regression_bind(
    duckdb::ClientContext &context, duckdb::ScalarFunction &function,
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> &arguments);

duckdb::unique_ptr<duckdb::FunctionData> linreg_impute_bind(
    duckdb::ClientContext &context, duckdb::ScalarFunction &function,
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> &arguments);

void linreg_impute(duckdb::DataChunk &args, duckdb::ExpressionState &state,
                   duckdb::Vector &result);

duckdb::unique_ptr<duckdb::FunctionLocalState> regression_init_state(duckdb::ExpressionState &state, const duckdb::BoundFunctionExpression &expr,
                                                     duckdb::FunctionData *bind_data);

}

#endif // DUCKDB_REGRESSION_H
