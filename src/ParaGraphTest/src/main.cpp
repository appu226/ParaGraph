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


#include "math_test.h"
#include "graph_test.h"
#include "ml_graph_builder_test.h"
#include "tensor_function_factory_test.h"

namespace {

template<typename t_test>
void register_test(para::graph::unit_test_collection& uts) {
    uts.push_back(para::graph::unit_test_uptr(new t_test));
}

} // end anonymous namespace

int main(int argc, char** argv) {
    using namespace para::graph;
    unit_test_collection uts;
    register_test<tensor_construction_test>(uts);
    register_test<tensor_zero_test>(uts);
    register_test<tensor_zero_derivative_test>(uts);
    register_test<tensor_scalar_test>(uts);
    register_test<tensor_identity_derivative_test>(uts);
    register_test<tensor_chain_multiplication_test>(uts);
    register_test<tensor_add_test>(uts);
    register_test<graph_scalar_test>(uts);
    register_test<graph_tensor_test>(uts);
    register_test<tensor_function_factory_add_test>(uts);
    register_test<tensor_function_factory_chain_multiplication_test>(uts);
    register_test<tensor_function_factory_sigmoid_test>(uts);
    register_test<tensor_function_factory_reduce_sum_test>(uts);
    register_test<tensor_function_factory_log_test>(uts);
    register_test<tensor_function_factory_element_wise_multiplication_test>(uts);
    register_test<tensor_function_factory_negative_test>(uts);
    register_test<ml_graph_builder_test>(uts);
    run_unit_tests(uts);
    return 0;
}

