/*
 * main.cpp
 *
 *  Created on: 26-Dec-2017
 *      Author: parakram
 */

#include "math_test.h"
#include "graph_test.h"
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
    run_unit_tests(uts);
    return 0;
}

