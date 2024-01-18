

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

static void print_matrix(size_t sz, const double *m)
{
    for (size_t i = 0; i < sz; i++)
    {
        for(size_t j = 0; j < sz; j++)
        {
            std::cout<< i <<", "<<j<<" -> "<<m[(i * sz) + j]<<std::endl;
        }
    }
}



// #include <postgres.h>
// #include <utils/memutils.h>
// #include <math.h>

//generate a vector of sum of attributes group by label
//returns count, sum of numerical attributes, sum of categorical attributes 1-hot encoded
void build_sum_vector(const cofactor &cofactor, size_t num_total_params, int label, uint64_t *cat_array, uint32_t *cat_vars_idxs, int drop_first,
                      /* out */ double *sum_vector)
{
  //allocates values in sum vector, each sum array is sorted in tuple order
  size_t search_start = 0;        // within one category class
  std::cout<<"count"<<std::endl;
  //add count
  std::map<int, float> relation_data = cofactor.lin_cat[label];
  for (auto &el:relation_data) {
    size_t idx_key = find_in_array(el.first, cat_array, cat_vars_idxs[label], cat_vars_idxs[label+1])-cat_vars_idxs[label];
    if (drop_first && idx_key == relation_data.size())//todo might be wrong but not used here
      continue;//skip
    assert(idx_key < relation_data.size());
      std::cout<<"element: "<<el.first<<" idx_key "<<idx_key<<std::endl;
      sum_vector[(idx_key*num_total_params)] = el.second;//count
  }

    std::cout<<"numerical"<<std::endl;

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
        if (drop_first && group_by_index == relation_data.size())//todo might be wrong but not used here
          continue;//skip
        sum_vector[(group_by_index * num_total_params) + numerical] = item.second;
      }
    }
  }
  //group by categorical*categorical
  std::cout<<"group by categorical*categorical"<<std::endl;
  //needs to retrieve products between label and other cat. vars
  size_t idx_in = 0;
  for (size_t curr_cat_var = 0; curr_cat_var < cofactor.num_categorical_vars; curr_cat_var++)
  {
    idx_in++;
    for (size_t other_cat_var = curr_cat_var+1; other_cat_var < cofactor.num_categorical_vars; other_cat_var++)//ignore self multiplication
    {
      size_t other_cat;
      int idx_other;
      int idx_current;
      assert (idx_in < cofactor.cat_cat.size());
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
        size_t search_end = cat_vars_idxs[other_cat + 1];
        size_t key_index_other_var = find_in_array(slots[idx_other], cat_array, search_start, search_end);//todo fix in postgres - search_start;
        if (drop_first && key_index_other_var == cat_vars_idxs[other_cat + 1])
          continue;//skip

        key_index_other_var += cofactor.num_continuous_vars + 1;
        size_t group_by_index = find_in_array((uint64_t) slots[idx_current], cat_array, cat_vars_idxs[label], cat_vars_idxs[label + 1]) - cat_vars_idxs[label];
        if (drop_first && group_by_index == cat_vars_idxs[label + 1] - cat_vars_idxs[label])
          continue;//skip

          std::cout<<"curr cat var: "<<curr_cat_var<<" other cat var: "<<other_cat_var<<" key label: "<<slots[idx_current]<<" key other var: "<<slots[idx_other]<<" storing in "<<(group_by_index*num_total_params)+key_index_other_var<<" value "<<item.second<<std::endl;

        sum_vector[(group_by_index*num_total_params)+key_index_other_var] = item.second;

      }
    }
  }
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
  std::cout<<"b"<<std::endl;
  int label = args.data[1].GetValue(0).GetValue<int>();
  float shrinkage = args.data[2].GetValue(0).GetValue<float>();
  bool normalize = args.data[3].GetValue(0).GetValue<bool>();

  cofactor cofactor;
  std::cout<<"a"<<std::endl;
  extract_data(args.data[0], &cofactor, 1);//duckdb::value passed as
  std::cout<<"a"<<std::endl;
  size_t num_params = sizeof_sigma_matrix(cofactor, label);
  std::cout<<"num params "<<num_params<<std::endl;
  double *sigma_matrix = new double [num_params * num_params]();
    std::cout<<"-- "<<num_params<<std::endl;


  //count distinct classes in var
  size_t num_categories = cofactor.lin_cat[label].size();;
    std::cout<<"---- "<<num_params<<std::endl;


  uint64_t *cat_array = NULL;//keeps all the categorical values inside the columns
  uint32_t *cat_vars_idxs = NULL;//keeps index of begin and end for each column in cat_array
    std::cout<<"n_cols_1hot_expansion "<<num_params<<std::endl;
  size_t tot_columns = n_cols_1hot_expansion(&cofactor, 1, &cat_vars_idxs, &cat_array, 0);//I need this function because I need cat_vars_idxs and cat_array
    std::cout<<"end "<<num_params<<std::endl;


  double *sum_vector = new double [num_params * num_categories]();
  double *mean_vector = new double [(num_params-1) * num_categories]();
  double *coef = new double [(num_params-1) * num_categories]();//from mean to coeff

    std::cout<<"build_sigma_matrix "<<num_params<<std::endl;

    build_sigma_matrix(cofactor, num_params, label, cat_array, cat_vars_idxs, 0, sigma_matrix);

    build_sum_vector(cofactor, num_params, label, cat_array, cat_vars_idxs, 0, sum_vector);

    double *means = NULL;
    double *std = NULL;

  if(normalize){
    means = new double [num_params]();
    std = new double [num_params]();
    standardize_sigma(sigma_matrix, num_params, means, std);
    //standardize also mean vec
    for (size_t i=0; i<num_categories;i++){
      for(size_t j=1; j<num_params; j++){
        //(value - N*mean)/std
        //N = sum_vec[0]
        sum_vector[(i*num_params)+j] = (sum_vector[(i*num_params)+j] - (means[j]*sum_vector[i*num_params])) / std[j];
      }
    }
  }
    std::cerr<<"a "<<num_params<<std::endl;

  //Removed constant terms: we just need cofactor (covariance)
  //shift cofactor, coef and mean
  for (size_t j = 1; j < num_params; j++) {
    for (size_t k = 1; k < num_params; k++) {
      sigma_matrix[((j-1) * (num_params-1)) + (k-1)] = sigma_matrix[(j * num_params) + k];
    }
  }
  num_params--;

    print_matrix(num_params, sigma_matrix);

    std::cerr<<"b "<<num_params<<std::endl;


    std::cout<<"sum_vector "<<num_params<<std::endl;

    for (size_t i = 0; i < num_categories; i++)
    {
        for(size_t j = 0; j < num_params+1; j++)
        {
            std::cout<< i <<", "<<j<<" -> "<<sum_vector[(i*(num_params+1)) + j]<<std::endl;
        }
    }


    //build covariance matrix and mean vectors
  for (size_t i = 0; i < num_categories; i++) {
    for (size_t j = 0; j < num_params; j++) {
      for (size_t k = 0; k < num_params; k++) {
          std::cout<<j<<" , "<<k<<" value: "<<sigma_matrix[(j*num_params)+k]<<" - "<<(double)(sum_vector[(i*(num_params+1))+(j+1)]) <<" * "<< sum_vector[(i*(num_params+1))+(k+1)] <<"/"<< (double) sum_vector[i*(num_params+1)]<<" = "<<((double)(sum_vector[(i*(num_params+1))+(j+1)] * sum_vector[(i*(num_params+1))+(k+1)]) / (double) sum_vector[i*(num_params+1)])<<std::endl;
        sigma_matrix[(j*num_params)+k] -= ((double)(sum_vector[(i*(num_params+1))+(j+1)] * sum_vector[(i*(num_params+1))+(k+1)]) / (double) sum_vector[i*(num_params+1)]);//cofactor->count
      }
      coef[(i*num_params)+j] = sum_vector[(i*(num_params+1))+(j+1)] / sum_vector[(i*(num_params+1))];
      mean_vector[(j*num_categories)+i] = coef[(i*num_params)+j]; // if transposed (j*num_categories)+i
    }
  }

    print_matrix(num_params, sigma_matrix);



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

  for (size_t j = 0; j < num_params; j++) {
    for (size_t k = 0; k < num_params; k++) {
      sigma_matrix[(j*num_params)+k] /= (double )(cofactor.N);//or / cofactor->count - num_categories
    }
  }

    std::cerr<<"c "<<num_params<<std::endl;

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

  //check errors:
  if (err > 0)
    std::cout<<"LDA failed to converge. Error: "<< err;
  else if (err < 0)
    std::cout<< "the i-th argument had an illegal value. Error: "<<err;

    std::cerr<<"compute intercept "<<num_params<<std::endl;

  //compute intercept

  double alpha = 1;
  double beta = 0;
  double *res = new double [num_categories*num_categories];

  char task = 'N';
  dgemm(&task, &task, &num_categories_int, &num_categories_int, &num_params_int, &alpha, mean_vector, &num_categories_int, coef, &num_params_int, &beta, res, &num_categories_int);
  double *intercept = new double [num_categories];
  for (size_t j = 0; j < num_categories; j++) {
    intercept[j] = (res[(j*num_categories)+j] * (-0.5)) + log(sum_vector[j * (num_params+1)] / (double) cofactor.N);
  }

  if(normalize){
    //apply normalization to coefficients
    //coefficients need to be devided by parameter variance
    //intercept is the same as there is no need for de-normalize output
    for(size_t i=0; i<num_categories; i++){
      for(size_t j=0; j<num_params; j++){//coeff does not include constant term and num_params already decreased
        coef[(i*num_params)+j] /= std[j+1];
      }
    }
  }

    std::cerr<<"kk "<<num_params<<std::endl;

  std::vector<float> d;

  d.push_back((float)num_categories);

  int size_idxs = cofactor.num_categorical_vars;
  if(cofactor.num_categorical_vars == 1)//. If 1 cat. vars then it's 0 (only label), otherwise idxs size +1 compensates no label
      size_idxs = 0;

    d.push_back((float)size_idxs);//todo fix this in postgres

  //d.push_back((float)cofactor.num_categorical_vars -1);//todo fix this in postgresql

  if (num_params - cofactor.num_continuous_vars > 0) {//there are categorical variables excluding label
    //store categorical value indices of cat. columns (without label)
    int remove = 0;//todo fix this in postgres
    for (size_t i = 0; i < cofactor.num_categorical_vars + 1; i++) {
      if (i == label) {
          remove = cat_vars_idxs[label+1] - cat_vars_idxs[label];
        continue;
      }
      d.push_back(cat_vars_idxs[i]-remove);//remove label idxs
    }
    //now store categorical values without label
    for (size_t i = 0; i < cat_vars_idxs[label]; i++) {
      d.push_back(cat_array[i]);
    }
    for (size_t i = cat_vars_idxs[label + 1]; i < cat_vars_idxs[cofactor.num_categorical_vars]; i++) {
      d.push_back(cat_array[i]);
    }
  }

  //add categorical labels
  for (size_t i = cat_vars_idxs[label]; i < cat_vars_idxs[label + 1]; i++) {
    d.push_back(cat_array[i]);
  }


  //return coefficients
  for (int i = 0; i < num_params * num_categories; i++) {
    d.push_back(((float)coef[i]));
  }

  //return intercept
  for (int i = 0; i < num_categories; i++) {
    d.push_back(((float)intercept[i]));
  }

  if(normalize){
    for (int i = 0; i < num_params; i++) {
      d.push_back(means[i+1]);
    }
  }

  delete[] intercept;
  delete[] res;
  delete[] iwork;
  delete[] s;
  delete[] sum_vector;
  delete[] mean_vector;
  delete[] coef;
  delete[] sigma_matrix;
  delete[] cat_array;
  delete[] cat_vars_idxs;

  if(normalize){
    delete[] means;
    delete[] std;
  }


  result.SetVectorType(duckdb::VectorType::CONSTANT_VECTOR);
  duckdb::ListVector::Reserve(result, d.size());
  duckdb::ListVector::SetListSize(result, d.size());
  auto out_data = duckdb::ConstantVector::GetData<float>(duckdb::ListVector::GetEntry(result));
  auto metadata_out = duckdb::ListVector::GetData(result);
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
  bool normalize = args.data[1].GetValue(0).GetValue<bool>();

  for (idx_t j=2;j<columns;j++){
    auto col_type = in_data[j].GetType();
    in_data[j].ToUnifiedFormat(size, input_data[j-1]);
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
    std::cout<<"Params: "<<params[i]<<std::endl;
  }
  int num_categories = (int) (params[0]);
  int size_cat_vars_idxs = (int) (params[1]);
  size_t curr_param_offset = 2;

  //

  int num_params = num_cols;
  uint64_t *cat_vars_idxs;
  uint64_t *cat_vars;
  if (size_cat_vars_idxs > 0) {
    cat_vars_idxs = new uint64_t [size_cat_vars_idxs];
    for(size_t i=0; i<size_cat_vars_idxs; i++){//build cat_var_idxs
      cat_vars_idxs[i] = params[i+curr_param_offset];//re-build index vector (begin:end of each cat. column)
      std::cout<<"cat_var_idxs "<<cat_vars_idxs[i]<<" index "<<i+curr_param_offset<<std::endl;
    }
    curr_param_offset += size_cat_vars_idxs;
    num_params = num_cols + cat_vars_idxs[size_cat_vars_idxs-1];
    cat_vars = new uint64_t [cat_vars_idxs[size_cat_vars_idxs-1]];//build cat_var
    for(size_t i=0; i<cat_vars_idxs[size_cat_vars_idxs-1]; i++){
      cat_vars[i] = params[i+curr_param_offset];
        std::cout<<"cat_vars "<<cat_vars[i]<<" index "<<i+curr_param_offset<<std::endl;
    }
    curr_param_offset += cat_vars_idxs[size_cat_vars_idxs-1];
  }

  int *target_labels = new int [num_categories];
  for(size_t i=0; i<num_categories; i++) {
    target_labels[i] = params[i + curr_param_offset];//build label classes
      std::cout<<"target_labels "<<target_labels[i]<<" index "<<i + curr_param_offset<<std::endl;
  }

  curr_param_offset += num_categories;

  double *coefficients = new double [num_params * num_categories];
  float *intercept = new float [num_categories];


  for(int i=0;i< num_categories;i++)
    for(int j=0;j< num_params;j++) {
        coefficients[(j * num_categories) + i] = (double) (params[(i * num_params) + j + curr_param_offset]);
        std::cout<<"coeff: "<<coefficients[(j * num_categories) + i]<<std::endl;
    }

  curr_param_offset += (num_params * num_categories);

  //build intercept
  for(int i=0;i<num_categories;i++) {
      intercept[i] = (double) params[i + curr_param_offset];
      std::cout<<"intercept: "<<intercept[i]<<std::endl;

  }
  curr_param_offset += num_categories;


  //allocate features

  double *feats_c = new double [num_params];
  double *res = new double [num_categories];


  for(int i=0; i<size; i++) {

      for(int j=0; j<num_params; j++){
          feats_c[j] = 0;
      }

    for(int j=0;j<num_cols;j++) {
      feats_c[j] = (double)  duckdb::UnifiedVectorFormat::GetData<float>(input_data[j+1])[input_data[j+1].sel->get_index(i)];
      std::cout<<"feats num "<<feats_c[j]<<std::endl;
    }

    //allocate categorical features

    for(int j=0;j<cat_cols;j++) {
      int in_class = (int)  duckdb::UnifiedVectorFormat::GetData<int>(input_data[j+num_cols+1])[input_data[j+num_cols+1].sel->get_index(i)];//todo fix this in ddb
      size_t index = find_in_array(in_class, cat_vars, cat_vars_idxs[j], cat_vars_idxs[j+1]);//use cat.vars to find index of cat. value
        std::cout<<"search "<<in_class<<"position "<<index<<std::endl;
        assert (index < cat_vars_idxs[j+1]);//1-hot used here, without removing values from 1-hot
      feats_c[num_cols + index] = 1;//build 1-hot feat vector
    }

    //todo fix
    if(normalize){
      //center input vector
        std::cout<<"start"<<std::endl;
        for(int i=0;i<num_cols;i++) {
          std::cout<<"index "<<curr_param_offset + i<<"val: "<<params[curr_param_offset + i]<<std::endl;
        feats_c[i] -= (double) (params[curr_param_offset + i]);
      }
        std::cout<<"aaa"<<std::endl;
        //also center other classes
        if (cat_cols > 0) {
            for (int i = 0; i < cat_vars_idxs[size_cat_vars_idxs - 1]; i++) {
                std::cout << "index " << curr_param_offset + num_cols + i << "val: "
                          << params[curr_param_offset + num_cols + i] << std::endl;
                feats_c[num_cols + i] -= (double) (params[curr_param_offset + num_cols + i]);
            }
        }
    }

      for(int j=0; j<num_params; j++){
          std::cout<<"feats: "<<feats_c[j]<<std::endl;
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
      std::cout<<"pred. val: "<<val<<std::endl;
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
  delete[] target_labels;

  if (size_cat_vars_idxs > 0) {
    delete[] cat_vars_idxs;
    delete[] cat_vars;
  }

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

