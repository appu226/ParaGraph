/**
 * Copyright 2018 Parakram Majumdar
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef PARA_GRAPH_FUNCTIONAL_H_
#define PARA_GRAPH_FUNCTIONAL_H_

#include <map>
#include <vector>
#include <functional>

namespace para {
namespace graph {

template<typename t_key, typename t_value>
struct f_map {
    typedef std::map<t_key, t_value> data_type;
    const data_type& data;

    template<typename t_value2, typename t_lam>
    std::map<t_key, t_value2> map_values(t_lam lam) const {
        std::map<t_key, t_value2> res;
        for (auto kv : data) {
            res[kv.first] = lam(kv.second);
        }
        return res;
    }

    std::vector<t_key> keys() const {
        std::vector<t_key> res;
        res.reserve(data.size());
        for (auto kv : data)
            res.push_back(kv.first);
        return res;
    }

    std::vector<t_value> values() const {
        std::vector<t_value> res;
        res.reserve(data.size());
        for (auto kv : data)
            res.push_back(kv.second);
        return res;
    }

    template<typename t_key2, typename t_value2, typename t_lam>
    std::map<t_key2, t_value2> map(t_lam lam) {
        std::map<t_key2, t_value2> res;
        for (auto kv : data)
            res.insert(lam(kv));
        return res;
    }

    f_map(const data_type& v_data) :
                    data(v_data) {
    }
};

template<typename t_data>
struct f_vector {
    const std::vector<t_data>& data;
    typedef t_data t_value;

    template<typename t_data2, typename t_lam>
    std::vector<t_data2> map(t_lam lam) {
        std::vector<t_data2> res;
        res.reserve(data.size());
        for (auto v : data)
            res.push_back(lam(v));
        return res;
    }

    template<typename t_result, typename t_data2, typename t_lam>
    std::vector<t_result> zip(const std::vector<t_data2>& that, t_lam lam) {
        size_t sz = std::min(data.size(), that.size());
        std::vector<t_result> res;
        res.reserve(sz);
        for (size_t idx = 0; idx < sz; ++idx)
            res.push_back(lam(data[idx], that[idx]));
        return res;
    }

    template<typename t_container2>
    std::vector<std::pair<t_data, typename t_container2::value_type>> zip(const t_container2& container2) {
        std::vector<std::pair<t_data, typename t_container2::value_type>> result;
        const auto end = data.end();
        const auto end2 = container2.end();
        auto it = data.begin();
        auto it2 = container2.begin();
        for (; it != end && it2 != end2; ++it, ++it2)
            result.push_back(std::make_pair(*it, *it2));
        return result;
    }

    template<typename t_container2>
    std::map<t_data, typename t_container2::value_type> zipToMap(const t_container2& container2) {
        std::map<t_data, typename t_container2::value_type> result;
        const auto end = data.end();
        const auto end2 = container2.end();
        auto it = data.begin();
        auto it2 = container2.begin();
        for (; it != end && it2 != end2; ++it, ++it2)
            result.insert(std::make_pair(*it, *it2));
        return result;
    }

    f_vector(const std::vector<t_data>& v_data) :
                    data(v_data) {
    }
};

template<typename t_key, typename t_value>
f_map<t_key, t_value> functional(const std::map<t_key, t_value>& data) {
    return f_map<t_key, t_value>(data);
}

template<typename t_value>
f_vector<t_value> functional(const std::vector<t_value>& data) {
    return f_vector<t_value>(data);
}

} // end namespace graph
} // end namespace para

#endif /* PARA_GRAPH_FUNCTIONAL_H_ */
