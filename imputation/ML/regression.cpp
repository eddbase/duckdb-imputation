

#include <ML/regression.h>
#include <ML/utils.h>

#include <math.h>
#include<iostream>
extern "C" void dgemv (char *TRANSA, int *M, int* N, double *ALPHA,
                   double *A, int *LDA, double *X, int *INCX, double *BETA, double *Y, int *INCY);


void print_matrix(size_t sz, const std::vector<double> &m)
{
    for (size_t i = 0; i < sz; i++)
    {
        for(size_t j = 0; j < sz; j++)
        {
            std::cout<<i<<", "<<j<<" -> "<< m[(i * sz) + j];
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


std::vector<double> Triple::ridge_linear_regression(const duckdb::Value &triple, size_t label, double step_size, double lambda, size_t max_num_iterations, bool compute_variance)
{
    cofactor cofactor;
    extract_data(triple, cofactor);

    if (cofactor.num_continuous_vars <= label) {
        std::cout<<"label ID >= number of continuous attributes";
        return {};
    }

    size_t num_params = sizeof_sigma_matrix(cofactor, -1);

    std::vector <double> grad(num_params, 0);
    std::vector <double> prev_grad(num_params, 0);
    std::vector <double> learned_coeff(num_params, 0);
    std::vector <double> prev_learned_coeff(num_params, 0);
    double *sigma = new double[num_params * num_params]();
    std::vector <double> update(num_params, 0);

    build_sigma_matrix(cofactor, num_params, -1, sigma);


    for (size_t i = 0; i < num_params; i++)
    {
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
        //std::cout<<"Error "<<error<<"\n";
        num_iterations++;
    } while (num_iterations < 1000 || num_iterations < max_num_iterations);
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


    //std::cout<< "num_iterations = "<< num_iterations;
    // export params to pgpsql
    delete[] sigma;

    return learned_coeff;
}


