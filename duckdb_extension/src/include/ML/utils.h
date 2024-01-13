

#ifndef TEST_UTILS_H
#define TEST_UTILS_H
#include<iostream>
#include <vector>
#include <map>
#include <duckdb.hpp>

struct cofactor{
  size_t N;
  size_t num_continuous_vars;
  size_t num_categorical_vars;
  std::vector<float> lin;
  std::vector<float> quad;

  std::vector<std::map<int, float>> lin_cat;
  std::vector<std::map<int, float>> num_cat;
  std::vector<std::map<std::pair<int, int>, float>> cat_cat;

};

size_t get_num_categories(const cofactor &cofactor, int label_categorical_sigma);
size_t sizeof_sigma_matrix(const cofactor &cofactor, int label_categorical_sigma);
void build_sigma_matrix(const cofactor &cofactor, size_t matrix_size, int label_categorical_sigma,
                        /* out */ double *sigma);
void extract_data(const duckdb::Vector &triple, cofactor *cofactor, size_t n_cofactor);
size_t find_in_array(uint64_t a, const uint64_t *array, size_t start, size_t end);
size_t n_cols_1hot_expansion(const cofactor *cofactors, size_t n_aggregates, uint32_t **cat_idxs, uint64_t **cat_unique_array, int drop_first);
void build_sigma_matrix(const cofactor &cofactor, size_t matrix_size, int label_categorical_sigma, uint64_t *cat_array, uint32_t *cat_vars_idxs, int drop_first,
                        /* out */ double *sigma);
void standardize_sigma(double *sigma, size_t num_params, double *means, double *std);


#endif //TEST_UTILS_H
