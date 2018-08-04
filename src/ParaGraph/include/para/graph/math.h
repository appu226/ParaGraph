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

/**
 * A type representing a multi-dimensional array of doubles.
 * API and implementation are that of a thin wrapper
 *   over a row-major vector f doubles,
 *   restricting it to a dense, random access representation.
 */
// TODO: Abstract the api away from std::vector<double>,
//       in particular, to allow for sparse representations.
struct tensor {
    typedef std::size_t N;
    typedef std::vector<N> N_vector;

    /** The sizes of the various dimensions of the multi-dimensional array. */
    N_vector dimensionalities;
    /**
     * The actual data that is stored in the tensor.
     * The ordering is row major.
     * E.g., if its a 2x3 matrix
     *   "dimensionalities" would contain [2, 3]
     *   and the ordering within "data" would be
     *   [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)]
     */
    std::vector<double> data;

    tensor(const N_vector& dimensionalities, const std::vector<double>& data);
    tensor(N_vector&& dimensionalities, std::vector<double>&& data);
    tensor(const N_vector& dimensionalities, std::vector<double>&& data);

    /** Get the offset in "data" from n-dimensional coordinates */
    N compute_offset(const N_vector& position) const;
    /** Get n-dimensional coordinates from the offset in "data" */
    N_vector compute_position(N offset) const;

    /** Check the consistency of "data" and "dimensionalities" */
    bool is_valid() const;

    /** Create a zero tensor. */
    static tensor zero(const N_vector& dimensionalities);
    /**
     * Create a zero gradient tensor.
     * The resulting dimensionality is concat(variable_dimensionalities, function_dimensionalities)
     * */
    static tensor zero_derivative(const N_vector& function_dimensionalities, const N_vector& variable_dimensionalities);
    /**
     * Create an identity derivative, a generalisation of a identity matrix.
     * The resulting dimensionality is concat(dimensionalities, dimensionalities).
     */
    static tensor identity_derivative(const N_vector& dimensionalities);
    /**
     * A generalisation of chain multiplication to tensors.
     * In particular, this is suitable for approximating
     *   the first order change in a function ∆F = F(x+∆x) - F(x)
     *   using the gradient ∇F(x) and change inputs ∆x
     *   as:
     *     ∆F ≈ chain_multiplication(∆x, ∇F(x), ∆x.dimensionalities.size())
     */
    static tensor chain_multiplication(const tensor& lhs, const tensor& rhs, int num_common_dims);
    /** Add two tensors */
    static tensor add(const tensor& lhs, const tensor& rhs);
};

typedef std::shared_ptr<const tensor> tensor_cptr;
typedef std::vector<tensor_cptr> tensor_cptr_vec;

}
}

#endif /* PARA_GRAPH_MATH_H_ */
