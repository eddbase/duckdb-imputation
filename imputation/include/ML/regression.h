
#include <duckdb.hpp>

#ifndef DUCKDB_REGRESSION_H
#define DUCKDB_REGRESSION_H

namespace Triple{
    std::vector<double> ridge_linear_regression(const duckdb::Value &triple, size_t label, double step_size, double lambda, size_t max_num_iterations, bool compute_variance);
}

#endif //DUCKDB_REGRESSION_H
