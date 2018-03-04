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

#ifndef PARA_GRAPH_GRAPH_TEST_UTILS_H_
#define PARA_GRAPH_GRAPH_TEST_UTILS_H_

#include <para/graph/math.h>
#include <random>

namespace para {
namespace graph {

void assert_tensors_are_close(const tensor& lhs, const tensor& rhs, double relative_tolerance,
        const std::string& message);

tensor_cptr generate_random_tensor(const tensor::N_vector& dims, std::default_random_engine& dre, double max = 1.0);

std::string print_tensor(const tensor& t, const std::string& name);

} // end namespace graph
} // end namespace para

#endif /* end ifndef PARA_GRAPH_GRAPH_TEST_UTILS_H_ */
