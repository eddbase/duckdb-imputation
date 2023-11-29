

#include <ML/utils.h>

void extract_data(const duckdb::Value &triple, cofactor &cofactor){
    //extract data

    auto first_triple_children = duckdb::StructValue::GetChildren(triple);//vector of pointers to childrens
    cofactor.N = (int) first_triple_children[0].GetValue<int>();

    duckdb::child_list_t<duckdb::Value> struct_values;
    const duckdb::vector<duckdb::Value> &linear = duckdb::ListValue::GetChildren(first_triple_children[1]);
    cofactor.lin.reserve(linear.size());
    cofactor.num_continuous_vars = linear.size();

    const duckdb::vector<duckdb::Value> &categorical = duckdb::ListValue::GetChildren(first_triple_children[3]);
    cofactor.lin_cat.reserve(categorical.size());
    cofactor.num_categorical_vars = categorical.size();
    //lin numerical
    for(idx_t i=0;i<linear.size();i++)
        cofactor.lin.push_back(linear[i].GetValue<float>());

    //copy lin. categorical
    //std::cout<<triple.ToString()<<std::endl;
    for(idx_t i=0;i<categorical.size();i++) {
        cofactor.lin_cat.emplace_back();
        const duckdb::vector<duckdb::Value> &items_in_col = duckdb::ListValue::GetChildren(categorical[i]);
        for(size_t j=0; j<items_in_col.size(); j++){
            //expand struct
            const duckdb::vector<duckdb::Value> &curr_pair = duckdb::StructValue::GetChildren(items_in_col[j]);
            cofactor.lin_cat[i][curr_pair[0].GetValue<int>()] = curr_pair[1].GetValue<float>();
        }
    }

    //quad numerical

    const duckdb::vector<duckdb::Value> &quad = duckdb::ListValue::GetChildren(first_triple_children[2]);

    cofactor.quad.reserve(quad.size());
    for(idx_t i=0;i<quad.size();i++)
        cofactor.quad.push_back(quad[i].GetValue<float>());

    //num*cat
    const duckdb::vector<duckdb::Value> &num_cat_in = duckdb::ListValue::GetChildren(first_triple_children[4]);
    cofactor.num_cat.reserve(categorical.size() * linear.size());
    for(idx_t num=0;num<linear.size();num++) {
        for (idx_t cat = 0; cat < categorical.size(); cat++) {
            size_t idx = (num*categorical.size()) + cat;
            cofactor.num_cat.emplace_back();
            const duckdb::vector<duckdb::Value> &items_in_col = duckdb::ListValue::GetChildren(num_cat_in[idx]);
            for (size_t j = 0; j < items_in_col.size(); j++) {
                //expand struct
                const duckdb::vector<duckdb::Value> &curr_pair = duckdb::StructValue::GetChildren(items_in_col[j]);
                cofactor.num_cat[idx][curr_pair[0].GetValue<int>()] = curr_pair[1].GetValue<float>();
            }
        }
    }

    //cat*cat
    const duckdb::vector<duckdb::Value> &cat_cat_in = duckdb::ListValue::GetChildren(first_triple_children[5]);
    cofactor.cat_cat.reserve((categorical.size() * (categorical.size()+1))/2);
    size_t idx = 0;
    for(idx_t cat_1=0; cat_1<categorical.size(); cat_1++) {
        for (idx_t cat_2 = cat_1; cat_2 < categorical.size(); cat_2++) {
            cofactor.cat_cat.emplace_back();
            const duckdb::vector<duckdb::Value> &items_in_col = duckdb::ListValue::GetChildren(cat_cat_in[idx]);
            for (size_t j = 0; j < items_in_col.size(); j++) {
                const duckdb::vector<duckdb::Value> &curr_pair = duckdb::StructValue::GetChildren(items_in_col[j]);
                cofactor.cat_cat[idx][std::pair<int, int>(curr_pair[0].GetValue<int>(), curr_pair[1].GetValue<int>())] = curr_pair[2].GetValue<float>();
            }
            idx++;
        }
    }
}

size_t find_in_array(uint64_t a, const uint64_t *array, size_t start, size_t end)
{
    size_t index = start;
    while (index < end)
    {
        if (array[index] == a)
            break;
        index++;
    }
    return index;
}

void build_sigma_matrix(const cofactor &cofactor, size_t matrix_size, int label_categorical_sigma,
        /* out */ double *sigma)
{
    // start numerical data:
    // numerical_params = cofactor->num_continuous_vars + 1

    // count
    sigma[0] = cofactor.N;

    // sum1
    const std::vector<float> &sum1_scalar_array = cofactor.lin;
    for (size_t i = 0; i < cofactor.num_continuous_vars; i++){
        sigma[i + 1] = sum1_scalar_array[i];
        sigma[(i + 1) * matrix_size] = sum1_scalar_array[i];
    }

    //sum2 full matrix (from half)
    const std::vector<float> &sum2_scalar_array = cofactor.quad;
    for (size_t row = 0; row < cofactor.num_continuous_vars; row++){
        for (size_t col = 0; col < cofactor.num_continuous_vars; col++){
            if (row > col)
                sigma[((row + 1) * matrix_size) + (col + 1)] = sum2_scalar_array[(col * cofactor.num_continuous_vars) - (((col) * (col + 1)) / 2) + row];
            else
                sigma[((row + 1) * matrix_size) + (col + 1)] = sum2_scalar_array[(row * cofactor.num_continuous_vars) - (((row) * (row + 1)) / 2) + col];
        }
    }
    // (numerical_params) * (numerical_params) allocated

    // start relational data:

    size_t num_categories = matrix_size - cofactor.num_continuous_vars - 1;
    uint64_t *cat_array = new uint64_t [num_categories]; // array of categories
    uint32_t *cat_vars_idxs = new uint32_t[cofactor.num_categorical_vars + 1]; // track start each cat. variable


    //linear categorical
    size_t search_start = 0;        // within one category class
    size_t search_end = search_start;

    cat_vars_idxs[0] = 0;

    for (size_t i = 0; i < cofactor.num_categorical_vars; i++) {
        std::map<int, float> relation_data = cofactor.lin_cat[i];
        if (label_categorical_sigma >= 0 && ((size_t)label_categorical_sigma) == i ){
            cat_vars_idxs[i + 1] = cat_vars_idxs[i];
            continue;
        }
        //create sorted array
        for (auto &element:relation_data) {
            size_t key_index = find_in_array(element.first, cat_array, search_start, search_end);
            if (key_index == search_end){
                uint64_t value_to_insert = element.first;
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
            }
        }
        search_start = search_end;
        cat_vars_idxs[i + 1] = cat_vars_idxs[i] + relation_data.size();
    }


    // count * categorical (group by A, group by B, ...)
    for (size_t i = 0; i < cofactor.num_categorical_vars; i++)
    {
        std::map<int, float> relation_data = cofactor.lin_cat[i];
        if (label_categorical_sigma >= 0 && ((size_t)label_categorical_sigma) == i )
        {
            //skip this variable
            continue;
        }

        for (auto &item:relation_data)
        {
            // search key index
            size_t key_index = find_in_array(item.first, cat_array, cat_vars_idxs[i], cat_vars_idxs[i + 1]);
            assert(key_index < search_end);
            /*if (key_index == search_end)    // not found
            {
                cat_array[search_end] = r->tuples[j].key;
                search_end++;
            }*/

            // add to sigma matrix
            key_index += cofactor.num_continuous_vars + 1;
            sigma[key_index] = item.second;
            sigma[key_index * matrix_size] = item.second;
            sigma[(key_index * matrix_size) + key_index] = item.second;
        }
        search_start = search_end;
        //cat_vars_idxs[i + 1] = cat_vars_idxs[i] + r->num_tuples;
    }

    // categorical * numerical
    for (size_t numerical = 1; numerical < cofactor.num_continuous_vars + 1; numerical++)
    {
        for (size_t categorical = 0; categorical < cofactor.num_categorical_vars; categorical++)
        {
            std::map<int, float> relation_data = cofactor.num_cat[(numerical-1)*cofactor.num_categorical_vars + categorical];
            if (label_categorical_sigma >= 0 && ((size_t)label_categorical_sigma) == categorical )
            {
                //skip this variable
                continue;
            }

            for (auto &item:relation_data)
            {
                //search in the right categorical var
                search_start = cat_vars_idxs[categorical];
                search_end = cat_vars_idxs[categorical + 1];

                size_t key_index = find_in_array(item.first, cat_array, search_start, search_end);
                assert(key_index < search_end);

                // add to sigma matrix
                key_index += cofactor.num_continuous_vars + 1;
                sigma[(key_index * matrix_size) + numerical] = item.second;
                sigma[(numerical * matrix_size) + key_index] = item.second;
            }
        }
    }

    // categorical * categorical
    size_t curr_element = 0;
    for (size_t curr_cat_var = 0; curr_cat_var < cofactor.num_categorical_vars; curr_cat_var++)
    {
        for (size_t other_cat_var = curr_cat_var; other_cat_var < cofactor.num_categorical_vars; other_cat_var++)
        {
            std::map<std::pair<int, int>, float> relation_data = cofactor.cat_cat[curr_element];
            curr_element++;

            if (label_categorical_sigma >= 0 && (((size_t)label_categorical_sigma) == curr_cat_var || ((size_t)label_categorical_sigma) == other_cat_var))
            {
                //skip this variable
                continue;
            }

            for (auto &item:relation_data)
            {
                search_start = cat_vars_idxs[curr_cat_var];
                search_end = cat_vars_idxs[curr_cat_var + 1];

                size_t key_index_curr_var = find_in_array(item.first.first, cat_array, search_start, search_end);
                assert(key_index_curr_var < search_end);

                search_start = cat_vars_idxs[other_cat_var];
                search_end = cat_vars_idxs[other_cat_var + 1];

                size_t key_index_other_var = find_in_array(item.first.second, cat_array, search_start, search_end);
                assert(key_index_other_var < search_end);

                // add to sigma matrix
                key_index_curr_var += cofactor.num_continuous_vars + 1;
                key_index_other_var += cofactor.num_continuous_vars + 1;
                sigma[(key_index_curr_var * matrix_size) + key_index_other_var] = item.second;
                sigma[(key_index_other_var * matrix_size) + key_index_curr_var] = item.second;
            }
        }
    }


    delete [] cat_array;
    delete [] cat_vars_idxs;
}

size_t get_num_categories(const cofactor &cofactor, int label_categorical_sigma)
{
    size_t num_categories = 0;

    for (size_t i = 0; i < cofactor.num_categorical_vars; i++)
    {
        std::map<int, float> relation_data = cofactor.lin_cat[i];
        if (label_categorical_sigma >= 0 && ((size_t)label_categorical_sigma) == i)
        {
            //skip this variable
            continue;
        }

        num_categories += relation_data.size();
    }
    return num_categories;
}

size_t sizeof_sigma_matrix(const cofactor &cofactor, int label_categorical_sigma)
{
    // count :: numerical :: 1-hot_categories
    return 1 + cofactor.num_continuous_vars + get_num_categories(cofactor, label_categorical_sigma);
}
