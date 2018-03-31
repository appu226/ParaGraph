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


#ifndef PARA_GRAPH_MATH_H_
#define PARA_GRAPH_MATH_H_

#include <vector>
#include <memory>

namespace para {
namespace graph {

struct tensor {
    typedef std::size_t N;
    typedef std::vector<N> N_vector;

    N_vector dimensionalities;
    std::vector<double> data;

    tensor(const N_vector& dimensionalities, const std::vector<double>& data);
    tensor(N_vector&& dimensionalities, std::vector<double>&& data);
    tensor(const N_vector& dimensionalities, std::vector<double>&& data);

    N compute_offset(const N_vector& position) const;
    N_vector compute_position(N offset) const;

    bool is_valid() const;

    static tensor zero(const N_vector& dimensionalities);
    static tensor zero_derivative(const N_vector& function_dimensionalities, const N_vector& variable_dimensionalities);
    static tensor identity_derivative(const N_vector& dimensionalities);
    static tensor chain_multiplication(const tensor& lhs, const tensor& rhs, int num_common_dims);
    static tensor add(const tensor& lhs, const tensor& rhs);
};

typedef std::shared_ptr<const tensor> tensor_cptr;
typedef std::vector<tensor_cptr> tensor_cptr_vec;

}
}

#endif /* PARA_GRAPH_MATH_H_ */
