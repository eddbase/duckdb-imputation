

#include <ML/lda.h>
#include <math.h>
#include <ML/utils.h>
#include <utils.h>
//#include <clapack.h>
#include <float.h>
#include <ML/lda.h>

#include <duckdb.hpp>
#include <duckdb/planner/expression/bound_function_expression.hpp>
#include <duckdb/function/scalar/nested_functions.hpp>
#include <cfloat>
#include <utils.h>

#include <iostream>

extern "C" void dgesdd_(char *JOBZ, int *m, int *n, double *A, int *lda, double *s, double *u, int *LDU, double *vt, int *l,
                        double *work, int *lwork, int *iwork, int *info);
extern "C" void dgelsd( int* m, int* n, int* nrhs, double* a, int* lda,
                       double* b, int* ldb, double* s, double* rcond, int* rank,
                       double* work, int* lwork, int* iwork, int* info );

extern "C" void dgemm (char *TRANSA, char* TRANSB, int *M, int* N, int *K, double *ALPHA,
                      double *A, int *LDA, double *B, int *LDB, double *BETA, double *C, int *LDC);


int compare( const void* a, const void* b)
{
  int int_a = * ( (int*) a );
  int int_b = * ( (int*) b );

  if ( int_a == int_b ) return 0;
  else if ( int_a < int_b ) return -1;
  else return 1;
}

// #include <postgres.h>
// #include <utils/memutils.h>
// #include <math.h>

//generate a vector of sum of attributes group by label
//returns count, sum of numerical attributes, sum of categorical attributes 1-hot encoded
void build_sum_vector(const cofactor &cofactor, size_t num_total_params, int label, int label_categorical_sigma,
                      /* out */ double *sum_vector)
{
  //allocates values in sum vector, each sum array is sorted in tuple order
  size_t num_categories = sizeof_sigma_matrix(cofactor, -1)  - cofactor.num_continuous_vars - 1;//num_total_params - cofactor->num_continuous_vars - 1;
  uint64_t *cat_array = new uint64_t[num_categories]; // array of categories, stores each key for each categorical variable
  uint32_t *cat_vars_idxs = new uint32_t[cofactor.num_categorical_vars + 1]; // track start each cat. variable
  cat_vars_idxs[0] = 0;

  size_t search_start = 0;        // within one category class
  size_t search_end = search_start;
  uint64_t * classes_order;

  for (size_t i = 0; i < cofactor.num_categorical_vars; i++) {
    std::map<int, float> relation_data = cofactor.lin_cat[i];
    //create sorted array
    for (auto &el:relation_data) {
      uint64_t value_to_insert = el.first;
      uint64_t tmp;
      for (size_t k = search_start; k < search_end; k++){
        if (value_to_insert < cat_array[k]){
          tmp = cat_array[k];
          cat_array[k] = value_to_insert;
          value_to_insert = tmp;
        }
      }
      cat_array[search_end] = value_to_insert;
      search_end++;
      //}
    }
    search_start = search_end;
    cat_vars_idxs[i + 1] = cat_vars_idxs[i] + relation_data.size();
  }

  //add count
  std::map<int, float> relation_data = cofactor.lin_cat[label];
  for (auto &el:relation_data) {
    size_t idx_key = find_in_array(el.first, cat_array, cat_vars_idxs[label], cat_vars_idxs[label+1])-cat_vars_idxs[label];
    assert(idx_key < relation_data.size());
    sum_vector[(idx_key*num_total_params)] = el.second;//count
  }


  //numerical features
  for (size_t numerical = 1; numerical < cofactor.num_continuous_vars + 1; numerical++) {
    for (size_t categorical = 0; categorical < cofactor.num_categorical_vars; categorical++) {
      //sum numerical group by label, copy value in sums vector
      if (categorical != label) {
        continue;
      }
      std::map<int, float> relation_data = cofactor.num_cat[((numerical-1)*cofactor.num_categorical_vars) + categorical];
      for (auto &item:relation_data) {
        size_t group_by_index = find_in_array(item.first, cat_array, cat_vars_idxs[label], cat_vars_idxs[label+1]) - cat_vars_idxs[label];
        sum_vector[(group_by_index * num_total_params) + numerical] = item.second;
      }
    }
  }
  //group by categorical*categorical
  size_t idx_in = 0;
  for (size_t curr_cat_var = 0; curr_cat_var < cofactor.num_categorical_vars; curr_cat_var++)
  {
    idx_in++;
    for (size_t other_cat_var = curr_cat_var+1; other_cat_var < cofactor.num_categorical_vars; other_cat_var++)//ignore self multiplication
    {
      size_t other_cat;
      int idx_other;
      int idx_current;
      std::map<std::pair<int, int>, float> relation_data = cofactor.cat_cat[idx_in];
      idx_in++;

      if (curr_cat_var == label) {
        other_cat = other_cat_var;
        idx_other = 1;
        idx_current = 0;
      }
      else if (other_cat_var == label){
        other_cat = curr_cat_var;
        idx_other = 0;
        idx_current = 1;
      }
      else {
        continue;
      }

      for (auto &item:relation_data)
      {
        std::vector<int> slots = {item.first.first, item.first.second};
        search_start = cat_vars_idxs[other_cat];
        search_end = cat_vars_idxs[other_cat + 1];
        size_t key_index_other_var = find_in_array(slots[idx_other], cat_array, search_start, search_end) - search_start;
        assert(key_index_other_var < search_end);

        key_index_other_var += cofactor.num_continuous_vars + 1 + search_start;
        if (search_start >= cat_vars_idxs[label])
          key_index_other_var -= (cat_vars_idxs[label + 1] - cat_vars_idxs[label]);

        size_t group_by_index = find_in_array((uint64_t) slots[idx_current], cat_array, cat_vars_idxs[label], cat_vars_idxs[label + 1]) - cat_vars_idxs[label];
        assert(group_by_index < search_end);


        sum_vector[(group_by_index*num_total_params)+key_index_other_var] = item.second;
      }
    }
  }
  delete[] cat_array;
  delete[] cat_vars_idxs;
}

duckdb::unique_ptr<duckdb::FunctionData>
lda_train_bind(duckdb::ClientContext &context, duckdb::ScalarFunction &function,
                duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> &arguments) {

  function.return_type = duckdb::LogicalType::LIST(duckdb::LogicalType::FLOAT);;
  return duckdb::make_uniq<duckdb::VariableReturnBindData>(function.return_type);
}

void lda_train(duckdb::DataChunk &args, duckdb::ExpressionState &state, duckdb::Vector &result)
{

  std::cout<<"a"<<std::endl;
  duckdb::idx_t size = args.size();
  duckdb::vector<duckdb::Vector> &in_data = args.data;
  duckdb::RecursiveFlatten(args.data[0], size);
  std::cerr<<"b"<<std::endl;
  int label = args.data[1].GetValue(0).GetValue<int>() -1;
  float shrinkage = args.data[2].GetValue(0).GetValue<float>();
  bool normalize = args.data[3].GetValue(0).GetValue<bool>();

  cofactor cofactor;
  std::cerr<<"a"<<std::endl;
  extract_data(args.data[0], cofactor);//duckdb::value passed as
  std::cerr<<"a"<<std::endl;
  size_t num_params = sizeof_sigma_matrix(cofactor, label);
  std::cerr<<"num params "<<num_params<<std::endl;
  double *sigma_matrix = new double [num_params * num_params]();


  //count distinct classes in var
  size_t num_categories = cofactor.lin_cat[label].size();;

  double *sum_vector = new double [num_params * num_categories]();
  double *mean_vector = new double [num_params * num_categories]();
  double *coef = new double [num_params * num_categories]();//from mean to coeff

  build_sigma_matrix(cofactor, num_params, label, sigma_matrix);
  build_sum_vector(cofactor, num_params, label, label, sum_vector);

  //build covariance matrix and mean vectors
  for (size_t i = 0; i < num_categories; i++) {
    for (size_t j = 0; j < num_params; j++) {
      for (size_t k = 0; k < num_params; k++) {
        sigma_matrix[(j*num_params)+k] -= ((double)(sum_vector[(i*num_params)+j] * sum_vector[(i*num_params)+k]) / (double) sum_vector[i*num_params]);//cofactor->count
      }
      coef[(i*num_params)+j] = sum_vector[(i*num_params)+j] / sum_vector[(i*num_params)];
      mean_vector[(j*num_categories)+i] = coef[(i*num_params)+j]; // if transposed (j*num_categories)+i
    }
  }

  //introduce shrinkage
  //double shrinkage = 0.4;
  double mu = 0;
  for (size_t j = 0; j < num_params; j++) {
    mu += sigma_matrix[(j*num_params)+j];
  }
  mu /= (float) num_params;

  for (size_t j = 0; j < num_params; j++) {
    for (size_t k = 0; k < num_params; k++) {
      sigma_matrix[(j*num_params)+k] *= (1-shrinkage);//apply shrinkage part 1
    }
  }

  for (size_t j = 0; j < num_params; j++) {
    sigma_matrix[(j*num_params)+j] += shrinkage * mu;
  }


  //Solve with LAPACK
  int err, lwork, rank;
  double rcond = -1.0;
  double wkopt;
  double* work;
  int *iwork = new int [((int)((3 * num_params * log2(num_params/2)) + (11*num_params)))];
  double *s = new double[num_params];
  int num_params_int = (int) num_params;
  int num_categories_int = (int) num_categories;

  lwork = -1;
  dgelsd( &num_params_int, &num_params_int, &num_categories_int, sigma_matrix, &num_params_int, coef, &num_params_int, s, &rcond, &rank, &wkopt, &lwork, iwork, &err);
  lwork = (int)wkopt;
  work = new double [lwork]();
  dgelsd( &num_params_int, &num_params_int, &num_categories_int, sigma_matrix, &num_params_int, coef, &num_params_int, s, &rcond, &rank, work, &lwork, iwork, &err);

  //elog(WARNING, "finished with err: %d", err);

  //compute intercept

  double alpha = 1;
  double beta = 0;
  double *res = new double [num_categories*num_categories];

  char task = 'N';
  dgemm(&task, &task, &num_categories_int, &num_categories_int, &num_params_int, &alpha, mean_vector, &num_categories_int, coef, &num_params_int, &beta, res, &num_categories_int);
  double *intercept = new double [num_categories];
  for (size_t j = 0; j < num_categories; j++) {
    intercept[j] = (res[(j*num_categories)+j] * (-0.5)) + log(sum_vector[j * num_params] / cofactor.N);
  }

  std::vector<float> d;

  d.push_back((float)num_params);
  d.push_back((float)num_categories);

  //return coefficients
  for (int i = 0; i < num_params * num_categories; i++) {
    d.push_back(((float)coef[i]));
  }

  //return intercept
  for (int i = 0; i < num_categories; i++) {
    d.push_back(((float)intercept[i]));
  }
  delete[] intercept;
  delete[] res;
  delete[] iwork;
  delete[] s;
  delete[] sum_vector;
  delete[] mean_vector;
  delete[] coef;
  delete[] sigma_matrix;
  
  std::cerr<<"write output "<<std::endl;

  //test
  result.SetVectorType(duckdb::VectorType::CONSTANT_VECTOR);
  duckdb::ListVector::Reserve(result, d.size());
  duckdb::ListVector::SetListSize(result, d.size());
  auto out_data = duckdb::ConstantVector::GetData<float>(duckdb::ListVector::GetEntry(result));
  auto metadata_out = duckdb::ListVector::GetData(result);
  //auto metadata_out = duckdb::ListVector::GetData(duckdb::ListVector::GetEntry(result));
  metadata_out[0].offset = 0;
  metadata_out[0].length = d.size();



  for(size_t i=0; i<d.size(); i++){
    out_data[i] = (float) (d[i]);
  }
}

extern "C" void dgemv (char *TRANSA, int *M, int* N, double *ALPHA,
                      double *A, int *LDA, double *X, int *INCX, double *BETA, double *Y, int *INCY);

void LDA_impute(duckdb::DataChunk &args, duckdb::ExpressionState &state, duckdb::Vector &result) {
  //params need to be the first attribute
  idx_t size = args.size();//n. of rows to return

  duckdb::vector<duckdb::Vector> &in_data = args.data;
  int columns = in_data.size();

  size_t num_cols = 0;
  size_t cat_cols = 0;

  duckdb::UnifiedVectorFormat input_data[1024];//todo vary length (columns)
  RecursiveFlatten(in_data[0], size);

  for (idx_t j=1;j<columns;j++){
    auto col_type = in_data[j].GetType();
    in_data[j].ToUnifiedFormat(size, input_data[j]);
    if (col_type == duckdb::LogicalType::FLOAT || col_type == duckdb::LogicalType::DOUBLE)
      num_cols++;
    else if (col_type == duckdb::LogicalType::INTEGER)
      cat_cols++;
  }
  duckdb::ListVector::GetEntry(in_data[0]).ToUnifiedFormat(size, input_data[0]);

  idx_t params_size = duckdb::ListVector::GetListSize(in_data[0]);
  float *params = new float[params_size];
  for(size_t i=0; i<params_size; i++){
    params[i] = duckdb::UnifiedVectorFormat::GetData<float>(input_data[0])[input_data[0].sel->get_index(i)];
  }
  int num_params = (int) (params[0]);
  int num_categories = (int) (params[1]);
  int size_one_hot = num_params - num_cols - 1;//PG_GETARG_INT64(3);

  double *coefficients = new double [num_params * num_categories];
  float *intercept = new float [num_categories];


  for(int i=0;i< num_categories;i++)
    for(int j=0;j< num_params;j++)
      coefficients[(j*num_categories)+i] = (double) (params[(i*num_params)+j+2]);

  for(int i=0;i<num_categories;i++)
    intercept[i] = (double) (params[i+(num_params * num_categories)+2]);

  //allocate features

  double *feats_c = new double [size_one_hot + num_cols + 1];
  double *res = new double [num_categories];

  for(int i=0; i<size; i++) {
    for (int j = 0; j < size_one_hot + num_cols + 1; j++) {
      feats_c[j] = 0;
    }

    feats_c[0] = 1;
    for (int j = 0; j < num_cols; j++) {//copy num values of tuple
      feats_c[1 + j] = (double)  duckdb::UnifiedVectorFormat::GetData<float>(input_data[j+1])[input_data[j+1].sel->get_index(i)];
    }

    for (int j = 0; j < cat_cols; j++) {//adds 1 to cat values (1-hot)
      feats_c[1 + num_cols + duckdb::UnifiedVectorFormat::GetData<int>(input_data[j+1+num_cols])[input_data[j+1+num_cols].sel->get_index(i)]] = 1;//cat_padding (1+n cont + value)
    }

    char task = 'N';
    double alpha = 1;
    int increment = 1;
    double beta = 0;

    dgemv(&task, &num_categories, &num_params, &alpha, coefficients, &num_categories, feats_c, &increment, &beta,
          res, &increment);

    double max = -DBL_MAX;
    int f_class = 0;

    for (int i = 0; i < num_categories; i++) {
      double val = res[i] + intercept[i];
      if (val > max) {
        max = val;
        f_class = i;
      }
    }
    result.SetValue(i, duckdb::Value(f_class));
    //f_class is the result
  }
  delete[] feats_c;
  delete[] res;
  delete[] coefficients;
  delete[] intercept;
  delete[] params;
}

//Returns the datatype used by this function
duckdb::unique_ptr<duckdb::FunctionData>
LDA_impute_bind(duckdb::ClientContext &context, duckdb::ScalarFunction &function,
                duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> &arguments) {

  auto struct_type = duckdb::LogicalType::INTEGER;
  function.return_type = struct_type;
  function.varargs = duckdb::LogicalType::ANY;
  return duckdb::make_uniq<duckdb::VariableReturnBindData>(function.return_type);
}

//Generate statistics for this function. Given input type ststistics (mainly min and max for every attribute), returns the output statistics
duckdb::unique_ptr<duckdb::BaseStatistics>
LDA_impute_stats(duckdb::ClientContext &context, duckdb::FunctionStatisticsInput &input) {
  auto &child_stats = input.child_stats;
  auto &expr = input.expr;
  auto struct_stats = duckdb::NumericStats::CreateUnknown(expr.return_type);
  return struct_stats.ToUnique();
}

