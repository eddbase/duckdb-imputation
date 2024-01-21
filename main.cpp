#include <iostream>

#include <duckdb.hpp>

#include <imputation_baseline.h>
#include <imputation_low.h>
#include <imputation_high.h>

#include <random>

#include <partition.h>
#include <vector>

std::vector<int> extract_sample(unsigned int samples_to_extract, unsigned int range, unsigned int seed = std::random_device{}()){
    std::vector<int> tid_sampled(range);
    for(int i=0; i<range; i++){
        tid_sampled[i] = i;
    }
    unsigned int curr_size = range;


    std::mt19937 gen(seed);
    for(size_t i=0; i<(int)(samples_to_extract); i++){
        std::uniform_int_distribution<> distr(0, curr_size-1); // define the range
        int random_idx = distr(gen);
        int val = tid_sampled[random_idx];
        tid_sampled[random_idx] = tid_sampled[curr_size];
        tid_sampled[curr_size] = val;
        curr_size--;
    }

    //return last elements
    std::vector<int> result(samples_to_extract);
    for(int i=0; i<samples_to_extract; i++)
        result[i] = tid_sampled[curr_size+i];
    return result;

}

int main(){
    duckdb::DBConfig c;
    //c.SetOption("allow_unsigned_extensions", duckdb::Value(true));
    c.options.allow_unsigned_extensions = true;
    duckdb::DuckDB db(":memory:", &c);
    duckdb::Connection con(db);

    //Load library
    con.Query("INSTALL '../duckdb_extension/build/release/extension/duckdb_imputation/duckdb_imputation.duckdb_extension'")->Print();
    con.Query("load duckdb_imputation")->Print();
    //load dataset

    con.Query("CREATE TABLE iris_orig(sepal_length FLOAT, sepal_width FLOAT, petal_length FLOAT, petal_width FLOAT, target INTEGER);");
    con.Query("INSERT INTO iris_orig SELECT * FROM read_csv('../iris.csv', header = true, AUTO_DETECT=TRUE);")->Print();

    con.Query("CREATE SEQUENCE id_sequence START 1");
    con.Query("ALTER TABLE iris_orig ADD COLUMN id INT DEFAULT nextval('id_sequence')");

    con.Query("SELECT * FROM iris_orig;")->Print();
    con.Query("CREATE TABLE iris_imputed AS SELECT * FROM iris_orig;")->Print();

    int size = con.Query("SELECT count(*) FROM iris_orig;")->GetValue<int>(0,0);

    //random generator
    std::vector<std::string> columns = {"sepal_length", "petal_length", "target"};

    for(size_t i=0; i<columns.size(); i++) {
        std::vector<int> tuples_null = extract_sample((int)(0.2*size), size, i);
        std::ostringstream oss;
        std::copy(begin(tuples_null), end(tuples_null), std::ostream_iterator<int>(oss, ","));
        con.Query("UPDATE iris_imputed SET "+columns[i]+" = null where id in ("+oss.str()+");")->Print();
    }

    con.Query("SELECT * from iris_imputed")->Print();
    run_MICE_baseline(con, {"sepal_length", "sepal_width", "petal_length", "petal_width"}, {"target"}, {"sepal_length", "petal_length"}, {"target"}, "iris_imputed", 5);
    con.Query("SELECT * from iris_imputed_complete")->Print();



    //load iris

}