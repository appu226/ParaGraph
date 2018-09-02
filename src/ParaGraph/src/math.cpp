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

#include <para/graph/math.h>
#include <para/graph/exception.h>
#include <algorithm>

namespace para {
namespace graph {

tensor::tensor(const tensor::N_vector& _dimensionalities, const std::vector<double>& _data) :
                dimensionalities(_dimensionalities),
                m_data(_data) {
    assert(is_valid(), "tensor construction invalid, check the size of the data.");
}

tensor::tensor(tensor::N_vector&& _dimensionalities, std::vector<double>&& _data) :
                dimensionalities(std::move(_dimensionalities)),
                m_data(std::move(_data)) {
    assert(is_valid(), "tensor construction invalid, check the size of the data.");
}

tensor::tensor(const tensor::N_vector& _dimensionalities, std::vector<double>&& _data) :
                dimensionalities(std::move(_dimensionalities)),
                m_data(_data) {
    assert(is_valid(), "tensor construction invalid, check the size of the data.");
}

tensor::N tensor::compute_offset(const tensor::N_vector& position) const {
    assert(position.size() == dimensionalities.size(), "Cannot compute offset of a", dimensionalities.size(),
            "-D tensor using a ", position.size(), "-D position.");
    ;
    N skip_size = 1;
    N offset = 0;
    for (int dim = position.size() - 1; dim >= 0; --dim) {
        offset += position[dim] * skip_size;
        skip_size *= dimensionalities[dim];
    }
    return offset;
}

tensor::N_vector tensor::compute_position(tensor::N offset) const {
    N_vector position(dimensionalities.size());
    N skip_size = 1;
    for (auto i_dim = dimensionalities.begin(); i_dim < dimensionalities.end(); ++i_dim) {
        skip_size *= *i_dim;
    }
    for (N dim = 0; dim < position.size(); ++dim) {
        skip_size /= dimensionalities[dim];
        position[dim] = offset / skip_size;
        offset %= skip_size;
    }
    return position;
}

tensor::iterator       tensor::begin()        { return iterator(*this, 0); }
tensor::const_iterator tensor::begin() const  { return const_iterator(*this, 0); }
tensor::iterator       tensor::end()          { return iterator(*this, m_data.size()); }
tensor::const_iterator tensor::end() const    { return const_iterator(*this, m_data.size()); }
tensor::const_iterator tensor::cbegin() const { return const_iterator(*this, 0); }
tensor::const_iterator tensor::cend() const   { return const_iterator(*this, m_data.size()); }


bool tensor::is_valid() const {
    N expected_size = 1;
    for (auto i_dim = dimensionalities.begin(); i_dim != dimensionalities.end(); ++i_dim)
        expected_size *= *i_dim;
    return expected_size == m_data.size();
}

tensor tensor::zero(const N_vector& dimensionalities) {
    std::size_t total_size = std::accumulate(dimensionalities.begin(), dimensionalities.end(), 1,
            [](std::size_t acc, N dim) {return acc * dim;});
    return std::move(tensor(dimensionalities, std::move(std::vector<double>(total_size, 0.0))));
}

tensor tensor::zero_derivative(const N_vector& function_dimensionalities, const N_vector& variable_dimensionalities) {
    std::size_t order = function_dimensionalities.size() + variable_dimensionalities.size();
    N_vector dim;
    dim.reserve(order);
    for (auto vd : variable_dimensionalities)
        dim.push_back(vd);
    for (auto fd : function_dimensionalities)
        dim.push_back(fd);
    return std::move(zero(dim));
}

tensor tensor::identity_derivative(const N_vector& dimensionalities) {
    tensor result(std::move(zero_derivative(dimensionalities, dimensionalities)));
    std::size_t step_size = 1
            + std::accumulate(dimensionalities.begin(), dimensionalities.end(), 1,
                    [](std::size_t acc, N dim) {return acc * dim;});
    for (std::size_t pos = 0; pos < result.m_data.size(); pos += step_size) {
        result.m_data[pos] = 1;
    }
    return result;
}

tensor tensor::chain_multiplication(const tensor& lhs, const tensor& rhs, int num_common_dims) {
    const N_vector& ldim = lhs.dimensionalities, rdim = rhs.dimensionalities;

    assert(num_common_dims >= 0, "Number of dimensions to be chained must be greater than or equal to 0");
    N ncd = static_cast<N>(num_common_dims);
    assert(ldim.size() >= ncd, "lhs tensor is too small for requested chain multiplication");
    assert(rdim.size() >= ncd, "rhs tensor is too small for requested chain multiplication");

    N_vector dim;
    N l_part_size = 1;
    dim.reserve(ldim.size() + rdim.size() - ncd - ncd);
    for (N d = 0; d < ldim.size() - ncd; ++d) {
        dim.push_back(ldim[d]);
        l_part_size *= ldim[d];
    }
    N common_size = 1;
    for (N d = 0; d < ncd; ++d) {
        assert(ldim[d + dim.size()] == rdim[d],
                "Chained dimensionalities of lhs and rhs are not matching while requesting chain multiplication.");
        common_size *= ldim[d + dim.size()];
    }
    N r_part_size = 1;
    for (N d = ncd; d < rdim.size(); ++d) {
        dim.push_back(rdim[d]);
        r_part_size *= rdim[d];
    }

    N data_size = l_part_size * r_part_size;
    std::vector<double> data(data_size);
    for (N data_pos = 0; data_pos < data_size; ++data_pos) {
        N l_part_pos = data_pos / r_part_size;
        N r_part_pos = data_pos % r_part_size;
        data[data_pos] = 0;
        for (N common_pos = 0; common_pos < common_size; ++common_pos) {
            N l_pos = l_part_pos * common_size + common_pos;
            N r_pos = common_pos * r_part_size + r_part_pos;
            data[data_pos] += lhs.m_data[l_pos] * rhs.m_data[r_pos];
        }
    }
    return tensor(std::move(dim), std::move(data));
}

tensor tensor::add(const tensor& lhs, const tensor& rhs) {
    assert(lhs.dimensionalities.size() == rhs.dimensionalities.size(),
            "Tensors must have matching orders for addition.");
    std::size_t order = lhs.dimensionalities.size();
    for (std::size_t i_dim = 0; i_dim < order; ++i_dim)
        assert(lhs.dimensionalities[i_dim] == rhs.dimensionalities[i_dim],
                "Tensors must have matching dimensionalities for addition, mismatch at axis ", i_dim);
    std::size_t size = lhs.m_data.size();
    std::vector<double> result_data(size);
    for (std::size_t i_data = 0; i_data < size; ++i_data) {
        result_data[i_data] = lhs.m_data[i_data] + rhs.m_data[i_data];
    }
    return std::move(tensor(lhs.dimensionalities, std::move(result_data)));
}

} // end namespace para
} // end namespace graph

