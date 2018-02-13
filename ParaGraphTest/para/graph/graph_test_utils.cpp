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

#include "graph_test_utils.h"
#include "unit_test.h"
#include <para/graph/exception.h>
#include <sstream>

namespace para {
namespace graph {

void assert_tensors_are_close(const tensor& lhs, const tensor& rhs, double relative_tolerance,
        const std::string& message) {
    assert(lhs.dimensionalities == rhs.dimensionalities, message, ": dimensionalities mismatch.");
    assert(lhs.data.size() == rhs.data.size(), message, ": data size mismatch, ", lhs.data.size(), " vs ",
            rhs.data.size());
    for (std::size_t i = 0; i < lhs.data.size(); ++i) {
        std::stringstream ss;
        ss << message << ": data mismath at index " << i << ": " << lhs.data[i] << " vs " << rhs.data[i];
        assert_doubles_are_close(lhs.data[i], rhs.data[i], relative_tolerance, ss.str());
    }
}

tensor_cptr generate_random_tensor(const tensor::N_vector& dimensionalities, std::default_random_engine& dre) {
    std::uniform_real_distribution<double> urd(0, 1);
    std::shared_ptr<tensor> t(new tensor(std::move(tensor::zero(dimensionalities))));
    for (std::size_t i = 0; i < t->data.size(); ++i) {
        t->data[i] = urd(dre);
    }
    return t;
}

std::string print_tensor(const tensor& t, const std::string& name) {
    std::stringstream ss;
    ss.precision(17);
    ss << "value of " << name << " is:\n";
    for (std::size_t offset = 0; offset < t.data.size(); ++offset) {
        ss << t.data[offset] << " ";
        tensor::N_vector pos = t.compute_position(offset);
        for (int i_pos = pos.size() - 1; i_pos >= 0; --i_pos) {
            if (pos[i_pos] == t.dimensionalities[i_pos] - 1) {
                ss << "\n";
            } else {
                break;
            }
        }
    }
    ss << std::endl;
    return ss.str();
}

} // end namespace graph
} // end namespace para

