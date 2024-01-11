

#ifndef DUCKDB_IMPUTATION_SUM_STATE_H
#define DUCKDB_IMPUTATION_SUM_STATE_H

#include <duckdb.hpp>
//#include <boost/unordered_map.hpp>
//#include <boost/container/flat_map.hpp>
//#include <boost/functional/hash.hpp>
#include <map>

namespace Triple {

struct SumState {
  int count;
  int num_attributes;
  int cat_attributes;

  //int num_keys_total_categories;

  float *lin_agg;
  float *quadratic_agg;

  //boost::container::flat_map<int, float> *lin_cat;
  std::map<int, std::vector<float>> *quad_num_cat;
  std::map<std::pair<int, int>, float> *quad_cat_cat;
};


struct StateFunction {
  template<class STATE>
  static void Initialize(STATE &state) {
    state.count = 0;
    state.num_attributes = 0;
    state.cat_attributes = 0;

    state.lin_agg = nullptr;
    state.quadratic_agg = nullptr;
    state.quad_num_cat = nullptr;
    state.quad_cat_cat = nullptr;
    //state->lin_agg = {};
    //state->quadratic_agg = {};
  }

  template<class STATE>
  static void Destroy(STATE &state, duckdb::AggregateInputData &aggr_input_data) {
    delete[] state.lin_agg;
    delete[] state.quad_cat_cat;
    delete[] state.quad_num_cat;
  }

  static bool IgnoreNull() {
    return false;
  }
};

void
SumStateCombine(duckdb::Vector &state, duckdb::Vector &combined, duckdb::AggregateInputData &aggr_input_data,
                idx_t count);

void
SumStateFinalize(duckdb::Vector &state_vector, duckdb::AggregateInputData &aggr_input_data, duckdb::Vector &result,
                 idx_t count,
                 idx_t offset);
}
#endif //DUCKDB_IMPUTATION_SUM_STATE_H
