

#include <ML/regression.h>
#include <ML/utils.h>

#include <math.h>
#include<iostream>

#include <duckdb/function/scalar/nested_functions.hpp>
#include <iostream>
#include <algorithm>
#include <utils.h>
#include <vector>

extern "C" void dgemv (char *TRANSA, int *M, int* N, double *ALPHA,
                      double *A, int *LDA, double *X, int *INCX, double *BETA, double *Y, int *INCY);

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

void compute_gradient(size_t num_params, size_t label_idx,
                      const double* sigma, const std::vector<double> &params,
                      /* out */ std::vector<double> &grad)
{
  if (sigma[0] == 0.0) return;

  /* Compute Sigma * Theta */
  for (size_t i = 0; i < num_params; i++)
  {
    grad[i] = 0.0;
    for (size_t j = 0; j < num_params; j++)
    {
      grad[i] += sigma[(i * num_params) + j] * params[j];
    }
    grad[i] /= sigma[0]; // count
  }
  grad[label_idx] = 0.0;
}

double compute_error(size_t num_params, const double *sigma,
                     const std::vector<double> &params, const double lambda)
{
  if (sigma[0] == 0.0) return 0.0;

  double error = 0.0;

  /* Compute 1/N * Theta^T * Sigma * Theta */
  for (size_t i = 0; i < num_params; i++)
  {
    double tmp = 0.0;
    for (size_t j = 0; j < num_params; j++)
    {
      tmp += sigma[(i * num_params) + j] * params[j];
    }
    error += params[i] * tmp;
  }
  error /= sigma[0]; // count

  /* Add the regulariser to the error */
  double param_norm = 0.0;
  for (size_t i = 1; i < num_params; i++)
  {
    param_norm += params[i] * params[i];
  }
  param_norm -= 1; // param_norm -= params[LABEL_IDX] * params[LABEL_IDX];
  error += lambda * param_norm;

  return error / 2;
}

inline double compute_step_size(double step_size, int num_params,
                                const std::vector<double> &params, const std::vector<double> &prev_params,
                                const std::vector<double> &grad, const std::vector<double> &prev_grad)
{
  double DSS = 0.0, GSS = 0.0, DGS = 0.0;

  for (int i = 0; i < num_params; i++)
  {
    double paramDiff = params[i] - prev_params[i];
    double gradDiff = grad[i] - prev_grad[i];

    DSS += paramDiff * paramDiff;
    GSS += gradDiff * gradDiff;
    DGS += paramDiff * gradDiff;
  }

  if (DGS == 0.0 || GSS == 0.0)
    return step_size;

  double Ts = DSS / DGS;
  double Tm = DGS / GSS;

  if (Tm < 0.0 || Ts < 0.0)
    return step_size;

  return (Tm / Ts > 0.5) ? Tm : Ts - 0.5 * Tm;
}


void ML::ridge_linear_regression(duckdb::DataChunk &args, duckdb::ExpressionState &state, duckdb::Vector &result)
{

  duckdb::idx_t size = args.size();
  duckdb::vector<duckdb::Vector> &in_data = args.data;
  duckdb::RecursiveFlatten(args.data[0], size);
  int label = args.data[1].GetValue(0).GetValue<int>();
  float step_size = args.data[2].GetValue(0).GetValue<float>();
  float lambda = args.data[3].GetValue(0).GetValue<float>();
  int max_num_iterations = args.data[4].GetValue(0).GetValue<int>();
  bool compute_variance = args.data[5].GetValue(0).GetValue<bool>();
  bool normalize = args.data[6].GetValue(0).GetValue<bool>();

  cofactor cofactor;
  extract_data(args.data[0], &cofactor, 1);//duckdb::value passed as

  for(size_t i=0; i<cofactor.lin.size(); i++)
    std::cout<<"LINEAR AGG: "<<cofactor.lin[i]<<std::endl;

  if (cofactor.num_continuous_vars < label) {
    std::cout<<"label ID >= number of continuous attributes";
    //return {};
  }

  uint64_t *cat_array = NULL;
  uint32_t *cat_vars_idxs = NULL;
  size_t num_params = n_cols_1hot_expansion(&cofactor, 1, &cat_vars_idxs, &cat_array, 0);//tot columns include label as well

  std::vector <double> grad(num_params, 0);
  std::vector <double> prev_grad(num_params, 0);
  std::vector <double> learned_coeff(num_params, 0);
  std::vector <double> prev_learned_coeff(num_params, 0);
  double *sigma = new double[num_params * num_params]();
  std::vector <double> update(num_params, 0);

  build_sigma_matrix(cofactor, num_params, -1, cat_array, cat_vars_idxs, 0, sigma);
  print_matrix(num_params, sigma);

  double *means = NULL;
  double *std = NULL;
  if (normalize){
    means = new double [num_params];
    std = new double [num_params];
    standardize_sigma(sigma, num_params, means, std);
  }

  //print_matrix(num_params, sigma);


  for (size_t i = 0; i < num_params; i++){
    learned_coeff[i] = 0; // ((double) (rand() % 800 + 1) - 400) / 100;
  }

  label += 1;     // index 0 corresponds to intercept
  prev_learned_coeff[label] = -1;
  learned_coeff[label] = -1;

  compute_gradient(num_params, label, sigma, learned_coeff, grad);

  double gradient_norm = grad[0] * grad[0]; // bias
  for (size_t i = 1; i < num_params; i++)
  {
    double upd = grad[i] + lambda * learned_coeff[i];
    gradient_norm += upd * upd;
  }
  gradient_norm -= lambda * lambda; // label correction
  double first_gradient_norm = sqrt(gradient_norm);

  double prev_error = compute_error(num_params, sigma, learned_coeff, lambda);

  size_t num_iterations = 1;
  do
  {
    // Update parameters and compute gradient norm
    update[0] = grad[0];
    gradient_norm = update[0] * update[0];
    prev_learned_coeff[0] = learned_coeff[0];
    prev_grad[0] = grad[0];
    learned_coeff[0] = learned_coeff[0] - step_size * update[0];
    double dparam_norm = update[0] * update[0];
    //std::cout<<"Learned coeff 0: "<<learned_coeff[0]<<"\n";
    for (size_t i = 1; i < num_params; i++)
    {
      update[i] = grad[i] + lambda * learned_coeff[i];
      gradient_norm += update[i] * update[i];
      prev_learned_coeff[i] = learned_coeff[i];
      prev_grad[i] = grad[i];
      learned_coeff[i] = learned_coeff[i] - step_size * update[i];
      dparam_norm += update[i] * update[i];
      //std::cout<<"Learned coeff "<<i<<" : "<<learned_coeff[i]<<"\n";
    }
    learned_coeff[label] = -1;
    gradient_norm -= lambda * lambda; // label correction
    dparam_norm = step_size * sqrt(dparam_norm);

    double error = compute_error(num_params, sigma, learned_coeff, lambda);

    /* Backtracking Line Search: Decrease step_size until condition is satisfied */
    size_t backtracking_steps = 0;
    while (error > prev_error - (step_size / 2) * gradient_norm && backtracking_steps < 500)
    {
      step_size /= 2; // Update parameters based on the new step_size.

      dparam_norm = 0.0;
      for (size_t i = 0; i < num_params; i++)
      {
        double newp = prev_learned_coeff[i] - step_size * update[i];
        double dp = learned_coeff[i] - newp;
        learned_coeff[i] = newp;
        dparam_norm += dp * dp;
      }
      dparam_norm = sqrt(dparam_norm);
      learned_coeff[label] = -1;
      error = compute_error(num_params, sigma, learned_coeff, lambda);
      backtracking_steps++;
    }

    /* Normalized residual stopping condition */
    gradient_norm = sqrt(gradient_norm);
    if (dparam_norm < 1e-20 ||
        gradient_norm / (first_gradient_norm + 0.001) < 1e-8)
    {
      break;
    }
    compute_gradient(num_params, label, sigma, learned_coeff, grad);

    step_size = compute_step_size(step_size, num_params, learned_coeff, prev_learned_coeff, grad, prev_grad);
    prev_error = error;
    std::cout<<"Error "<<error<<std::endl;
    num_iterations++;
  } while (num_iterations < max_num_iterations);
  double variance = 0;
  if(compute_variance){
    //compute variance for stochastic linear regression

    double *learned_coeff_tmp = new double[num_params];
    for(int i=0; i<num_params; i++)
      learned_coeff_tmp[i] = learned_coeff[i];

    char task = 'N';
    double alpha = 1;
    int increment = 1;
    double beta = 0;
    int int_num_params = num_params;
    double *res = new double[num_params];
    //sigma * (X^T * X)
    learned_coeff_tmp[label] = -1;
    dgemv(&task, &int_num_params, &int_num_params, &alpha, sigma, &int_num_params, learned_coeff_tmp, &increment, &beta, res, &increment);
    int size_row = 1;
    //(sigma * (X^T * X))*sigma^T
    dgemv(&task, &size_row, &int_num_params, &alpha, res, &size_row, learned_coeff_tmp, &increment, &beta, &variance, &increment);
    variance /= (double) cofactor.N;
    delete[] res;
    delete[] learned_coeff_tmp;
    learned_coeff.push_back(variance);
  }

  if (normalize){//rescale coeff. because of standardized dataset
    for (size_t i=1; i<num_params; i++)
      learned_coeff[i] = (learned_coeff[i] / std[i]) * std[label];

    learned_coeff[0] = (learned_coeff[0]* std[label]) + means[label];
  }

  delete[] sigma;
  result.SetVectorType(duckdb::VectorType::CONSTANT_VECTOR);

  //std::cout<<"Num params 1: "<<num_params<<std::endl;

  size_t learned_params_size = num_params;
  if (cofactor.num_categorical_vars > 0)//there are categorical variables, store unique vars and idxs
    num_params += cat_vars_idxs[cofactor.num_categorical_vars] + cofactor.num_categorical_vars+1;//# unique categoricals + idxs
  //std::cout<<"Num params 2: "<<num_params<<std::endl;

  if(normalize){
    num_params += (learned_params_size - 2);//need to store mean for each column (except label and constant term)
  }
  //num_params --;//no label element, variance not counted. However also need to store in [0] n. cat. vars
  //std::cout<<"Num params 3: "<<num_params<<std::endl;


  if(!compute_variance) {
    duckdb::ListVector::Reserve(result, num_params);
    duckdb::ListVector::SetListSize(result, num_params);
  }
  else {
    duckdb::ListVector::Reserve(result, num_params +1);
    duckdb::ListVector::SetListSize(result, num_params +1);
  }

  auto out_data = duckdb::ConstantVector::GetData<float>(duckdb::ListVector::GetEntry(result));
  auto metadata_out = duckdb::ListVector::GetData(result);
  //auto metadata_out = duckdb::ListVector::GetData(duckdb::ListVector::GetEntry(result));
  metadata_out[0].offset = 0;
  if(!compute_variance)
    metadata_out[0].length = num_params;
  else
    metadata_out[0].length = num_params+1;

  //std::cout<<"Num params: "<<num_params<<std::endl;

  //copy values



  out_data[0] = ((float)cofactor.num_categorical_vars);//size categorical columns

  int idx_output = 1;
  if (cofactor.num_categorical_vars > 0) {//there are categorical variables
    //store categorical value indices of cat. columns (without label)
    for (size_t i = 0; i < cofactor.num_categorical_vars + 1; i++) {
      out_data[idx_output] = ((float) cat_vars_idxs[i]);
      idx_output++;
    }
    for (size_t i = 0; i < cat_vars_idxs[cofactor.num_categorical_vars]; i++) {
      out_data[idx_output] = ((float) cat_array[i]);
      idx_output++;
    }
  }

  for (size_t i = 0; i < label; i++){//the first element is the constant term, so label element is label +1
    out_data[idx_output] = (learned_coeff[i]);
    //std::cout<<"coeff "<<learned_coeff[i]<<std::endl;
    idx_output++;
  }
  for (size_t i = label+1; i < learned_params_size; i++){
    out_data[idx_output] = (learned_coeff[i]);
    //std::cout<<"coeff "<<learned_coeff[i]<<std::endl;
    idx_output++;
  }

  if (normalize) {
    for (size_t i = 1; i < label; i++) {
      out_data[idx_output] = (means[i]);
      idx_output++;
    }
    for (size_t i = label+1; i < learned_params_size; i++) {
      out_data[idx_output] = (means[i]);
      idx_output++;
    }
  }

  if(compute_variance){
    out_data[idx_output] = (sqrt(variance));//returns std instead of variance
    idx_output++;
  }
}


duckdb::unique_ptr<duckdb::FunctionData>
ML::ridge_linear_regression_bind(duckdb::ClientContext &context, duckdb::ScalarFunction &function,
               duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> &arguments) {

  function.return_type = duckdb::LogicalType::LIST(duckdb::LogicalType::FLOAT);;
  return duckdb::make_uniq<duckdb::VariableReturnBindData>(function.return_type);
}


duckdb::unique_ptr<duckdb::FunctionData>
ML::linreg_impute_bind(duckdb::ClientContext &context, duckdb::ScalarFunction &function,
                   duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> &arguments){

  auto struct_type = duckdb::LogicalType::FLOAT;
  function.return_type = struct_type;
  function.varargs = duckdb::LogicalType::ANY;
  return duckdb::make_uniq<duckdb::VariableReturnBindData>(function.return_type);

}

unique_ptr<FunctionLocalState> ML::regression_init_state(ExpressionState &state, const BoundFunctionExpression &expr,
                                                   FunctionData *bind_data) {
  auto &info = bind_data->Cast<RegressionState>();
  if(!info.random_seed_set){
    FILE *istream = fopen("/dev/urandom", "rb");
    assert(istream);
    unsigned long seed = 0;
    for (unsigned i = 0; i < sizeof seed; i++) {
      seed *= (UCHAR_MAX + 1);
      int ch = fgetc(istream);
      assert(ch != EOF);
      seed += (unsigned)ch;
    }
    fclose(istream);
    srandom(seed);
    info.random_seed_set = true;
  }
  return make_uniq<RegressionState>(info);
}

void ML::linreg_impute(duckdb::DataChunk &args, duckdb::ExpressionState &state, duckdb::Vector &result_array){

  auto &r_state = ExecuteFunctionState::GetFunctionState(state)->Cast<RegressionState>();

  duckdb::idx_t size = args.size();
  duckdb::vector<duckdb::Vector> &in_data = args.data;
  duckdb::RecursiveFlatten(args.data[0], size);
  bool noise = args.data[1].GetValue(0).GetValue<bool>();
  bool normalize = args.data[2].GetValue(0).GetValue<bool>();
  int columns = in_data.size();
  int num_cols = 0;
  int cat_cols = 0;
  duckdb::UnifiedVectorFormat input_data[1024];//todo vary length (columns)

  for (idx_t j=3;j<columns;j++){
    auto col_type = in_data[j].GetType();
    in_data[j].ToUnifiedFormat(size, input_data[j-2]);
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


  int n_cat_columns = (int) params[0];
  int max_cat_vars_idx = 0;
  size_t start_params = 1 + n_cat_columns;//skip n. cat. columns and idx_cat_vals is n.cols

  if (n_cat_columns > 0) {
    max_cat_vars_idx = (int)params[start_params];
    start_params += max_cat_vars_idx +1;
  }

  for(size_t row=0; row<size; row++) {

    double result = params[start_params]; // init with intercept
    std::cout<<"start param: "<<result;

    if (normalize) {
      for (size_t i = 0; i < num_cols; i++) { // build num. pred.
        result +=
            ((double)(params[i + start_params + 1]) *
             ((double)(duckdb::UnifiedVectorFormat::GetData<float>(input_data[i+1])[input_data[i+1].sel->get_index(row)]) -
              (params[1 + num_cols + max_cat_vars_idx + start_params + i])));
      }
    } else {
      for (size_t i = 0; i < num_cols; i++) { // build num. pred.
        result += ((double)(params[i + start_params + 1]) *
                   (double)duckdb::UnifiedVectorFormat::GetData<float>(input_data[i+1])[input_data[i+1].sel->get_index(row)]); // re-build index vector (begin:end of each cat. column)
      }
    }

    for (size_t i = 0; i < cat_cols; i++) { // build cat. pred.
      int curr_class = duckdb::UnifiedVectorFormat::GetData<int>(input_data[num_cols + 1 + i])[input_data[num_cols + 1 + i].sel->get_index(row)];
      // search for class in cat array
      int start = (int)(params[1 + i]);
      int end = (int)(params[2 + i]);

      size_t index = start;
      while (index < end) {
        if ((int)(params[index + 2 + n_cat_columns]) == curr_class)
          break;
        index++;
      }
        std::cout<<"search class "<<curr_class<<" found at "<<index<<" cat col index "<<i<<" out of "<<cat_cols<<std::endl;
        if (normalize) {
        for (size_t j = start; j < index; j++) {
          result +=
              (double)((params[j + start_params + num_cols + 1]) *
                       (0 - (params[1 + (2 * num_cols) + max_cat_vars_idx +
                                    start_params + j])));
        }
        result += (double)((params[index + start_params + num_cols + 1]) *
                           (1 - (params[1 + (2 * num_cols) + max_cat_vars_idx +
                                        start_params + index])));
        for (size_t j = index + 1; j < end; j++) {
          result +=
              (double)((params[j + start_params + num_cols + 1]) *
                       (0 - (params[1 + (2 * num_cols) + max_cat_vars_idx +
                                    start_params + j])));
        }
      }
        else {
        result += (double)(params[index + start_params + num_cols +
                                  1]); // skip continous vars, class idx and unique class (+1 because of constant term)
                                  std::cout<<"categorical param value: "<<index + start_params + num_cols + 1<<std::endl;
      }
    }

    if (noise) {

      double u1, u2;
      do {
        u1 = random() / (double)(RAND_MAX + 1.0);
      } while (u1 == 0);
      u2 = random() / (double)(RAND_MAX + 1.0);
      // compute z0 and z1
      double mag =
          (params[params_size - 1]) * sqrt(-2.0 * log(u1)) * cos(2 * PI * u2);
      result += mag;
    }
    // save result
    result_array.SetValue(row, duckdb::Value(result));
  }

}
