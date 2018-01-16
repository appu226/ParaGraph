/*
 * main.cpp
 *
 *  Created on: 26-Dec-2017
 *      Author: parakram
 */

#include "math_test.h"
#include "graph_test.h"

int main(int argc, char** argv) {
    using namespace para::graph;
    unit_test_collection uts;
    uts.push_back(unit_test_uptr(new tensor_construction_test));
    uts.push_back(unit_test_uptr(new tensor_zero_test));
    uts.push_back(unit_test_uptr(new tensor_zero_derivative_test));
    uts.push_back(unit_test_uptr(new tensor_scalar_test));
    uts.push_back(unit_test_uptr(new tensor_identity_derivative_test));
    uts.push_back(unit_test_uptr(new tensor_chain_multiplication_test));
    uts.push_back(unit_test_uptr(new tensor_add_test));
    uts.push_back(unit_test_uptr(new graph_scalar_test));
    uts.push_back(unit_test_uptr(new graph_tensor_test));
    run_unit_tests(uts);
    return 0;
}

