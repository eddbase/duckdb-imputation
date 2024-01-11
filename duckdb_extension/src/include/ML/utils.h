

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
void extract_data(const duckdb::Vector &triple, cofactor &cofactor);
size_t find_in_array(uint64_t a, const uint64_t *array, size_t start, size_t end);



#endif //TEST_UTILS_H
