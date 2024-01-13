//
// Created by Massimo Perini on 10/01/2024.
//

#include <ML/qda.h>
#include <utils.h>
#include <ML/utils.h>

#include <cfloat>

#include <duckdb/function/scalar/nested_functions.hpp>

extern "C" void dgemm (char *TRANSA, char* TRANSB, int *M, int* N, int *K, double *ALPHA,
                  double *A, int *LDA, double *B, int *LDB, double *BETA, double *C, int *LDC);

extern "C" void dgemv (char *TRANSA, int *M, int* N, double *ALPHA,
                  double *A, int *LDA, double *X, int *INCX, double *BETA, double *Y, int *INCY);

extern "C" void dgesvd( char* jobu, char* jobvt, int* m, int* n, double* a,
                   int* lda, double* s, double* u, int* ldu, double* vt, int* ldvt,
                   double* work, int* lwork, int* info );

extern "C" void dscal( int* N, double* DA, double *DX, int* INCX );

void ML::qda_train(duckdb::DataChunk &args, duckdb::ExpressionState &state,
               duckdb::Vector &result){

  duckdb::idx_t size = args.size();
  duckdb::vector<duckdb::Vector> &in_data = args.data;

  duckdb::Vector &in_cofactors = args.data[0];
  duckdb::Vector &in_labels = args.data[1];

  duckdb::RecursiveFlatten(in_cofactors, size);//these are lists
  duckdb::RecursiveFlatten(in_labels, size);//these are lists
  //also labels in list
  //extract labels values

  bool normalize = args.data[2].GetValue(0).GetValue<bool>();

  int n_aggregates = duckdb::ListVector::GetListSize(in_cofactors);

  struct cofactor *cofactor = new struct cofactor[n_aggregates];
  extract_data(duckdb::ListVector::GetEntry(in_cofactors), cofactor, n_aggregates); // duckdb::value passed as
  int drop_first = 1;//PG_GETARG_INT64(2);

  //parse cofactor aggregates

  const auto &labels = duckdb::ConstantVector::GetData<int>(duckdb::ListVector::GetEntry(in_cofactors));

  uint32_t *cat_idxs = NULL;
  uint64_t *cat_array = NULL;
  size_t num_params = n_cols_1hot_expansion(cofactor, n_aggregates, &cat_idxs, &cat_array, drop_first);//enable drop first

  int m = num_params-1;//remove constant term computed previously
  double *sing_values =new double [m];
  double *u = new double[m*m];
  double *vt = new double[m*m];
  double* inva = new double [m*m];
  double* mean_vector = new double [m];
  double *lin_result = new double[m];

  size_t res_size = (1+1+ (((m*m)+m+1)*n_aggregates)) + n_aggregates;//need space for n_labels, size idxs, labels, quadratic, linear and constant term for each class
  if (cofactor[0].num_categorical_vars > 0){
    res_size += cofactor[0].num_categorical_vars+1 + cat_idxs[cofactor[0].num_categorical_vars]; //if there are categoricals, need also to store its idxs and unique values
  }
  if(normalize){
    res_size += m;
  }

  result.SetVectorType(duckdb::VectorType::CONSTANT_VECTOR);

  duckdb::ListVector::Reserve(result, res_size);
  duckdb::ListVector::SetListSize(result, res_size);
  auto out_data = duckdb::ConstantVector::GetData<float>(duckdb::ListVector::GetEntry(result));
  auto metadata_out = duckdb::ListVector::GetData(result);
  //auto metadata_out = duckdb::ListVector::GetData(duckdb::ListVector::GetEntry(result));
  metadata_out[0].offset = 0;
  metadata_out[0].length = res_size;

  out_data[0] = (float) n_aggregates;
  size_t param_out_index = 2;

  //save categorical (unique) values and begin:end indices in output
  if (cofactor[0].num_categorical_vars > 0) {
    out_data[1] = cofactor[0].num_categorical_vars + 1;
    for(size_t i=0; i<cofactor[0].num_categorical_vars+1; i++)
      out_data[i+2] = cat_idxs[i];

    param_out_index = cofactor[0].num_categorical_vars+3;
    for(size_t i=0; i<cat_idxs[cofactor[0].num_categorical_vars]; i++)
      out_data[param_out_index+i] = cat_array[i];
    param_out_index += cat_idxs[cofactor[0].num_categorical_vars];
  }
  else
    out_data[1] = 0;

  //copy labels
  for(size_t i=0; i<n_aggregates; i++) {
    out_data[param_out_index + i] =  (float) labels[i];
  }

  param_out_index += n_aggregates;

  double tot_tuples = 0;
  for(size_t i=0; i<n_aggregates; i++){
    tot_tuples += cofactor[i].N;
  }

  double **sigma_matrices = new double* [n_aggregates];
  //double **sum_vectors = (double **)palloc0(sizeof(double **) * n_aggregates);

  for(size_t i=0; i<n_aggregates; i++) {
    sigma_matrices[i] = new double [num_params * num_params];
    build_sigma_matrix(cofactor[i], num_params, -1, cat_array, cat_idxs, drop_first, sigma_matrices[i]);
  }

  //handle normalization
  double *means = NULL;
  double *std = NULL;
  if (normalize){
    means = new double [num_params];
    std = new double [num_params];
    for(size_t i=0; i<n_aggregates; i++) {
      for (size_t j = 0; j < num_params; j++) {
        means[j] += sigma_matrices[i][j];
        std[j] += sigma_matrices[i][(j*num_params)+j];
      }
    }
    for (size_t j = 0; j < num_params; j++) {
      means[j] /= tot_tuples;
      std[j] = sqrt((std[j]/tot_tuples) - pow(means[j], 2));//sqrt((sigma[(i*num_params)+i]/sigma[0]) - pow(sigma[i]/sigma[0], 2));
    }
    //std[0] = 1;

    //normalize
    for(size_t mm=0; mm<n_aggregates; mm++) {
      double *sigma = sigma_matrices[mm];
      //double *sum_vector = sum_vectors[mm];

      //elog(WARNING, "Printing sigma 0... mm= %lu", mm);
      //print_matrix(num_params, sigma);

      for (size_t i = 1; i < num_params; i++) {
        for (size_t j = 1; j < num_params; j++) {
          sigma[(i * num_params) + j] =
              (sigma[(i * num_params) + j] - (means[i] * sigma[j]) - (means[j] * sigma[i]) +
               (sigma[0] * means[j] * means[i])) / (std[i] * std[j]);
        }
      }

      //fix first row and col
      for (size_t i = 1; i < num_params; i++) {
        sigma[i] = (sigma[i] - (means[i] * sigma[0])) / std[i];
        sigma[(i * num_params)] = (sigma[(i * num_params)]- (means[i] * sigma[0])) / std[i];
      }
    } //end for each aggregate
  }

  for(size_t ii=0; ii<n_aggregates; ii++) {

    double *sigma_matrix = sigma_matrices[ii];
    //double *sum_vector = sum_vectors[ii];

    //print_matrix(num_params, sigma_matrix);

    //generate covariance matrix (for each class)
    double count_tuples = sigma_matrix[0];
    for(size_t i=0; i<m; i++)
      mean_vector[i] = sigma_matrix[i+1];//this is sum vector at the moment

    for (size_t j = 1; j < num_params; j++) {
      for (size_t k = 1; k < num_params; k++) {
        sigma_matrix[((j-1) * (num_params-1)) + (k-1)] = (sigma_matrix[(j * num_params) + k] - ((float)(mean_vector[j-1] * mean_vector[k-1]) / (float) count_tuples));
      }
    }

    for(size_t i=0; i<m; i++)
      mean_vector[i] /= count_tuples;

    //normalize with / count
    //regularization if enabled should be added before
    for (size_t j = 0; j < m; j++) {
      for (size_t k = 0; k < m; k++) {
        sigma_matrix[(j*m)+k] /= count_tuples;//or / cofactor->count - num_categories
      }
    }

    //print_matrix(num_params-1, sigma_matrix);

    //invert the matrix with SVD. We can also use LU decomposition, might be faster but less stable
    int lwork = -1;
    double wkopt;
    int info;
    dgesvd( "S", "S", &m, &m, sigma_matrix, &m, sing_values, u, &m, vt, &m, &wkopt, &lwork, &info );
    double *work = new double [wkopt];
    lwork = wkopt;
    /* Compute SVD */
    dgesvd( "S", "S", &m, &m, sigma_matrix, &m, sing_values, u, &m, vt, &m, work, &lwork, &info );
    delete[] work;
    /* Check for convergence */
    if( info > 0 ) {
      std::cerr<< "The algorithm computing SVD failed to converge.";
    }
    //we computed SVD, now calculate u=Σ-1U, where Σ is diagonal, so compute by a loop of calling BLAS ?scal function for computing product of a vector by a scalar
    //u is stored column-wise, and vt is row-wise here, thus the formula should be like u=UΣ-1
    double rcond = (1.0e-15);//Cutoff for small singular values. Singular values less than or equal to rcond * largest_singular_value are set to zero
    //consider using syevd for SVD. For positive semi-definite matrices (like covariance) SVD and eigenvalue decomp. are the same

    int incx = 1;
    for(int i=0; i<m; i++)
    {
      double ss;
      if(sing_values[i] > 1.0e-9)
        ss=1.0/sing_values[i];//1/sing. value is the inverse of diagonal matrix
      else
        ss=sing_values[i];
      dscal(&m, &ss, &u[i*m], &incx);
    }
    double determinant = 1;
    for(int i=0; i<m; i++)
      determinant *= sing_values[i];

    //elog(WARNING, "determinant %lf", determinant);
    //calculate A+=(V*)TuT, use MKL ?GEMM function to calculate matrix multiplication
    double alpha=1.0, beta=0.0;
    dgemm( "T", "T", &m, &m, &m, &alpha, vt, &m, u, &m, &beta, inva, &m);
    //elog(WARNING, "g");

    if(normalize){
      std[0] = 1;
      //make sure the inverse is / by square of std
      for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
          out_data[param_out_index + (i * m) + j] = (((float) -1 * inva[(i * m) + j] / 2) / (std[i+1]*std[j+1]));
        }
      }
    }
    else {
      //I have the inverse, save it in output
      for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
          out_data[param_out_index + (i * m) + j] = ((float) -1 * inva[(i * m) + j] / 2);
        }
      }
    }
    //elog(WARNING, "h");
    param_out_index += (m*m);
    //compute product with mean
    char task = 'N';
    int increment = 1;

    dgemv(&task, &m, &m, &alpha, inva, &m, mean_vector, &increment, &beta, lin_result, &increment);
    if (normalize){//need to scale down params
      for (int j = 0; j < m; j++) {
        out_data[param_out_index + j] = ((float) lin_result[j]/std[j+1]);
      }
    }
    else {
      for (int j = 0; j < m; j++) {
        //elog(WARNING, "mean %lf", mean_vector[j]);
        out_data[param_out_index + j] = ((float) lin_result[j]);
      }
    }
    //elog(WARNING, "i");

    param_out_index += m;
    int row=1;
    double intercept = 0;
    dgemv(&task, &row, &m, &alpha, mean_vector, &row, lin_result, &increment, &beta, &intercept, &increment);
    /*elog(WARNING, "intercept 1 %lf", (intercept));
    for(int i=0; i<m; i++)
        elog(WARNING, " %lf * %lf", mean_vector[i], lin_result[i]);*/
    intercept = ((-1)*intercept/(double)2) - (log(determinant)/(double)2) + log(count_tuples/(double)tot_tuples);
    //elog(WARNING, "intercept 1 %lf", (log(determinant)/(double)2));
    //elog(WARNING, "intercept 1 %lf", log(sum_vector[0]/(double)tot_tuples));
    out_data[param_out_index] = ((float ) intercept);
    param_out_index++;
    //elog(WARNING, "l");
  }

  if(normalize){
    //save means
    for (int j = 0; j < m; j++) {
      out_data[param_out_index+j] = (means[j+1]);
    }
    param_out_index += m;
  }

  assert(res_size == param_out_index);

  delete[] sing_values;
  delete[] u;
  delete[] vt;
  delete[] inva;
  delete[] mean_vector;
  delete[] lin_result;

  delete[] cofactor;
  delete[] cat_idxs;
  delete[] cat_array;
  if(normalize){
    delete[] means;
    delete[] std;
  }

  for(size_t i=0; i<n_aggregates; i++) {
    delete[] sigma_matrices[i];
  }
  delete[] sigma_matrices;

}

duckdb::unique_ptr<duckdb::FunctionData> ML::qda_train_bind(
    duckdb::ClientContext &context, duckdb::ScalarFunction &function,
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> &arguments){

  function.return_type = duckdb::LogicalType::LIST(duckdb::LogicalType::FLOAT);;
  return duckdb::make_uniq<duckdb::VariableReturnBindData>(function.return_type);
}

void ML::qda_impute(duckdb::DataChunk &args, duckdb::ExpressionState &state,
                duckdb::Vector &result){

  idx_t size = args.size();//n. of rows to return
  duckdb::vector<duckdb::Vector> &in_data = args.data;
  int columns = in_data.size();
  size_t num_cols = 0;
  size_t cat_cols = 0;

  duckdb::UnifiedVectorFormat input_data[1024];//todo vary length (columns)
  RecursiveFlatten(in_data[0], size);//means_covariance in postgresql
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
  //extract train data

  idx_t params_size = duckdb::ListVector::GetListSize(in_data[0]);
  float *params = new float[params_size];
  for(size_t i=0; i<params_size; i++){//copy params
    params[i] = duckdb::UnifiedVectorFormat::GetData<float>(input_data[0])[input_data[0].sel->get_index(i)];
  }
  int n_classes = (int) (params[0]);
  int size_idxs = (int) (params[1]);
  int one_hot_size = 0;

  uint64_t *cat_vars_idxs;
  uint64_t *cat_vars;
  //elog(WARNING, "c");
  size_t start_params_idx=2;

  //extract categorical values and indices
  if (size_idxs > 0) {
    cat_vars_idxs = new uint64_t [size_idxs];//max. size
    for (size_t i = 0; i < size_idxs; i++)
      cat_vars_idxs[i] = params[i + 2];

    one_hot_size = cat_vars_idxs[size_idxs - 1];
    cat_vars = new uint64_t [one_hot_size];//max. size
    for (size_t i = 0; i < cat_vars_idxs[size_idxs - 1]; i++)
      cat_vars[i] = params[i + 2 + size_idxs];

    start_params_idx=2+cat_vars_idxs[size_idxs-1]+size_idxs;
  }

  //skip classes output
  size_t start_out_classes = start_params_idx;
  start_params_idx += n_classes;

  size_t n_params = one_hot_size + num_cols;
  double *features = new double [n_params];
  double *quad_matrix = new double [n_params*n_params];
  double *lin_matrix = new double [n_params*n_params];
  double *res_matmul = new double [n_params];

  int m = n_params;

  //finished extracting values

  //for each tuple to predict...
  for (size_t j=0; j<size; j++){//like LDA, this is unnecessary expensive. A better implementation would
    //write all features of all rows in an array and solve the linear algebra operations once

    for (int i = 0; i < n_params; i++) {
      features[j] = 0;
    }

    //copy features
    for (int i = 0; i < num_cols; i++) {
      features[i] = (double)  duckdb::UnifiedVectorFormat::GetData<float>(input_data[i+2])[input_data[i+2].sel->get_index(j)];
    }

    for(int i=0; i<cat_cols; i++){//categorical feats (builds 1-hot encoded vector)
      int in_class = duckdb::UnifiedVectorFormat::GetData<int>(input_data[i+2+num_cols])[input_data[i+2+num_cols].sel->get_index(j)];//cat_padding (1+n cont + value)
      size_t index = find_in_array(in_class, cat_vars, cat_vars_idxs[i], cat_vars_idxs[i+1]);
      if (index < cat_vars_idxs[i+1])
        features[index+num_cols] = 1;
    }

    if(normalize) {
      size_t offset_params = start_params_idx + (((n_params*n_params) + n_params + 1)*n_classes);
      for (int i = 0; i < num_cols; i++) {
        features[i] -= (params[offset_params + i]);
      }

      if(size_idxs > 0) {//if categorical
        for (int i = 0; i < cat_vars_idxs[size_idxs - 1]; i++) {//categorical feats (builds 1-hot encoded vector)
          features[i + num_cols] -= (params[offset_params + num_cols + i]);
        }
      }
    }


    size_t best_class = 0;
    double max_prob = -DBL_MAX;
    for(size_t i=0; i<n_classes; i++){
      //copy qda params
      for(size_t j=0; j<n_params*n_params; j++){
        quad_matrix[j] = params[j + start_params_idx];
      }
      start_params_idx += (n_params*n_params);
      for(size_t j=0; j<n_params; j++){
        lin_matrix[j] = params[j + start_params_idx];
      }
      start_params_idx += (n_params);
      double intercept = params[start_params_idx];

      start_params_idx++;

      //compute probability of given class with matrix multiplication

      char task = 'N';
      int increment = 1;
      double alpha=1.0, beta=0.0;

      //quad_matrix * features
      dgemv(&task, &m, &m, &alpha, quad_matrix, &m, features, &increment, &beta, res_matmul, &increment);
      int row=1;
      double res_prob_1 = 0;
      //(quad_matrix * features)*features
      dgemv(&task, &row, &m, &alpha, res_matmul, &row, features, &increment, &beta, &res_prob_1, &increment);

      //lin_matrix * features
      double res_prob_2 = 0;
      dgemv(&task, &row, &m, &alpha, lin_matrix, &row, features, &increment, &beta, &res_prob_2, &increment);

      double total_prob = intercept + res_prob_1 + res_prob_2;
      //elog(WARNING, "prob: %lf intercept %lf prob 1 %lf prob 2 %lf", total_prob, intercept, res_prob_1, res_prob_2);

      if (total_prob > max_prob){
        max_prob = total_prob;
        best_class = i;
      }
    }

    int actual_class = (int) params[start_out_classes + best_class];
    result.SetValue(j, duckdb::Value(actual_class));

  }

  if (size_idxs > 0) {
    delete[] cat_vars_idxs;
    delete[] cat_vars;
  }

  delete[] features;
  delete[] quad_matrix;
  delete[] lin_matrix;
  delete[] res_matmul;
}

duckdb::unique_ptr<duckdb::FunctionData> ML::qda_impute_bind(
    duckdb::ClientContext &context, duckdb::ScalarFunction &function,
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> &arguments){

  //expects params, boolean if normalized, columns
  auto struct_type = duckdb::LogicalType::INTEGER;
  function.return_type = struct_type;
  function.varargs = duckdb::LogicalType::ANY;
  return duckdb::make_uniq<duckdb::VariableReturnBindData>(function.return_type);
}
