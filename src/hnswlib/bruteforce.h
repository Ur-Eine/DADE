#pragma once
#include "hnswlib.h"
#include "dade.h"
#include <unordered_map>
#include <fstream>
#include <mutex>
#include <algorithm>

using namespace std;

namespace hnswlib {
    template<typename dist_t>
    class BruteforceSearch : public AlgorithmInterface<dist_t> {
    public:
        BruteforceSearch(SpaceInterface <dist_t> *s) {

        }
        BruteforceSearch(SpaceInterface<dist_t> *s, const std::string &location) {
            loadIndex(location, s);
        }

        BruteforceSearch(SpaceInterface <dist_t> *s, size_t maxElements) {
            max_elements_ = maxElements;
            data_size_ = s->get_data_size();
            fstdistfunc_ = s->get_dist_func();
            dist_func_param_ = s->get_dist_func_param();
            size_per_element_ = data_size_ + sizeof(labeltype);
            data_ = (char *) malloc(maxElements * size_per_element_);
            if (data_ == nullptr)
                std::runtime_error("Not enough memory: BruteforceSearch failed to allocate data");
            cur_element_count = 0;
        }

        ~BruteforceSearch() {
            free(data_);
        }

        char *data_;
        size_t max_elements_;
        size_t cur_element_count;
        size_t size_per_element_;

        size_t data_size_;
        DISTFUNC <dist_t> fstdistfunc_;
        void *dist_func_param_;
        std::mutex index_lock;

        std::unordered_map<labeltype,size_t > dict_external_to_internal;

        void addPoint(const void *datapoint, labeltype label) {

            int idx;
            {
                std::unique_lock<std::mutex> lock(index_lock);



                auto search=dict_external_to_internal.find(label);
                if (search != dict_external_to_internal.end()) {
                    idx=search->second;
                }
                else{
                    if (cur_element_count >= max_elements_) {
                        throw std::runtime_error("The number of elements exceeds the specified limit\n");
                    }
                    idx=cur_element_count;
                    dict_external_to_internal[label] = idx;
                    cur_element_count++;
                }
            }
            memcpy(data_ + size_per_element_ * idx + data_size_, &label, sizeof(labeltype));
            memcpy(data_ + size_per_element_ * idx, datapoint, data_size_);




        };

        void removePoint(labeltype cur_external) {
            size_t cur_c=dict_external_to_internal[cur_external];

            dict_external_to_internal.erase(cur_external);

            labeltype label=*((labeltype*)(data_ + size_per_element_ * (cur_element_count-1) + data_size_));
            dict_external_to_internal[label]=cur_c;
            memcpy(data_ + size_per_element_ * cur_c,
                   data_ + size_per_element_ * (cur_element_count-1),
                   data_size_+sizeof(labeltype));
            cur_element_count--;

        }


        std::priority_queue<std::pair<dist_t, labeltype >>
        searchKnn(void *query_data, size_t k, int adaptive=0) const {
            std::priority_queue<std::pair<dist_t, labeltype >> topResults;
            if (cur_element_count == 0) return topResults;
            for (int i = 0; i < k; i++) {
                dist_t dist = fstdistfunc_(query_data, data_ + size_per_element_ * i, dist_func_param_);
                topResults.push(std::pair<dist_t, labeltype>(dist, *((labeltype *) (data_ + size_per_element_ * i +
                                                                                    data_size_))));
                dade::tot_full_dist++;
            }
            dist_t lastdist = topResults.top().first;
            for (int i = k; i < cur_element_count; i++) {
                dist_t dist;
                if (adaptive == 0){
                    dist = fstdistfunc_(query_data, data_ + size_per_element_ * i, dist_func_param_);
                    dade::tot_full_dist++;

                    if (dist <= lastdist) {
                        topResults.push(std::pair<dist_t, labeltype>(dist, *((labeltype *) (data_ + size_per_element_ * i +
                                                                                            data_size_))));
                        if (topResults.size() > k)
                            topResults.pop();
                        lastdist = topResults.top().first;
                    }
                }else{
                    dist = dade::dist_comp(lastdist, data_ + size_per_element_ * i, query_data, 0, 0);

                    if (dist >= 0) {
                        topResults.push(std::pair<dist_t, labeltype>(dist, *((labeltype *) (data_ + size_per_element_ * i +
                                                                                            data_size_))));
                        if (topResults.size() > k)
                            topResults.pop();
                        lastdist = topResults.top().first;
                    }
                }
            }
            return topResults;
        };

        void saveIndex(const std::string &location) {
            std::ofstream output(location, std::ios::binary);
            std::streampos position;

            writeBinaryPOD(output, max_elements_);
            writeBinaryPOD(output, size_per_element_);
            writeBinaryPOD(output, cur_element_count);

            output.write(data_, max_elements_ * size_per_element_);

            output.close();
        }

        void loadIndex(const std::string &location, SpaceInterface<dist_t> *s) {


            std::ifstream input(location, std::ios::binary);
            std::streampos position;

            readBinaryPOD(input, max_elements_);
            readBinaryPOD(input, size_per_element_);
            readBinaryPOD(input, cur_element_count);

            data_size_ = s->get_data_size();
            fstdistfunc_ = s->get_dist_func();
            dist_func_param_ = s->get_dist_func_param();
            size_per_element_ = data_size_ + sizeof(labeltype);
            data_ = (char *) malloc(max_elements_ * size_per_element_);
            if (data_ == nullptr)
                std::runtime_error("Not enough memory: loadIndex failed to allocate data");

            input.read(data_, max_elements_ * size_per_element_);

            input.close();

        }

    };
}
