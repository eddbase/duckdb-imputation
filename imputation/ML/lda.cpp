

#include <ML/lda.h>
#include <math.h>
#include <ML/utils.h>

//#include <clapack.h>
#include <float.h>

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

duckdb::Value lda_train(const duckdb::Value &triple, size_t label, double shrinkage)
{
    cofactor cofactor;
    extract_data(triple, cofactor);
    size_t num_params = sizeof_sigma_matrix(cofactor, label);
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

    duckdb::vector<duckdb::Value> d = {};

    d.push_back(duckdb::Value((float)num_params));
    d.push_back(duckdb::Value((float)num_categories));

    //return coefficients
    for (int i = 0; i < num_params * num_categories; i++) {
        d.push_back(duckdb::Value((float)coef[i]));
    }

    //return intercept
    for (int i = 0; i < num_categories; i++) {
        d.push_back(duckdb::Value((float)intercept[i]));
    }
    delete[] intercept;
    delete[] res;
    delete[] iwork;
    delete[] s;
    delete[] sum_vector;
    delete[] mean_vector;
    delete[] coef;
    delete[] sigma_matrix;

    return duckdb::Value::LIST(d);
}
