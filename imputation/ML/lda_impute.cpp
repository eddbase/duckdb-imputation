
#include <ML/lda_impute.h>

#include <duckdb.hpp>
#include <duckdb/planner/expression/bound_function_expression.hpp>
#include <duckdb/function/scalar/nested_functions.hpp>
#include <cfloat>
#include <triple/From_duckdb.h>

#include <iostream>

extern "C" void dgemv (char *TRANSA, int *M, int* N, double *ALPHA,
                   double *A, int *LDA, double *X, int *INCX, double *BETA, double *Y, int *INCY);

void LDA_impute(duckdb::DataChunk &args, duckdb::ExpressionState &state, duckdb::Vector &result) {
    //params need to be the first attribute
    idx_t size = args.size();//n. of rows to return

    duckdb::vector<duckdb::Vector> &in_data = args.data;
    int columns = in_data.size();

    size_t num_cols = 0;
    size_t cat_cols = 0;

    duckdb::UnifiedVectorFormat input_data[columns];
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

