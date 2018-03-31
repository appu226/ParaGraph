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

#include "tensor_function_factory_test.h"
#include "graph_test_utils.h"

#include <para/graph/ml_graph.h>
#include <para/graph/exception.h>
#include <random>
#include <algorithm>
#include <iostream>

namespace {
using namespace para::graph;

std::default_random_engine dre;

void test_function(const char* name, const tensor_function_csptr& func, const tensor_cptr_vec& inputs,
        const tensor& expected_value, std::default_random_engine& dre) {
    const double tolerance = 1e-15;
    const double step_size = 1e-6;
    const double derivative_tolerance = 1e-10;
    tensor_cptr v = func->value(inputs);
    derivative d = func->deriv(inputs);
    assert_tensors_are_close(*v, expected_value, tolerance,
            std::string("value of function ") + name + " should match expected value.");
    assert_tensors_are_close(*d.node_value, expected_value, tolerance,
            std::string("node_value from derivative of function ") + name + " should match expected value.");
    assert(d.node_derivative.size() == inputs.size(), "node_derivative of function ", name, " has size ",
            d.node_derivative.size(), " expected ", inputs.size());
    tensor_cptr_vec bumped_inputs = inputs;
    for (std::size_t i_input = 0; i_input < inputs.size(); ++i_input) {
        const tensor& input = *inputs[i_input];
        const tensor& d_wrt_input = *d.node_derivative[i_input];
        tensor delta(*generate_random_tensor(input.dimensionalities, dre));
        for (std::size_t i_delta = 0; i_delta < delta.data.size(); ++i_delta) {
            delta.data[i_delta] *= step_size;
        }

        // bump input, compute value, and UN-bump input
        bumped_inputs[i_input] = tensor_cptr(new tensor(std::move(tensor::add(input, delta))));
        tensor_cptr bumped_value = func->value(bumped_inputs);
        bumped_inputs[i_input] = inputs[i_input];

        tensor projected_bump_in_value = tensor::chain_multiplication(delta, d_wrt_input,
                input.dimensionalities.size());
        tensor projected_bumped_value = tensor::add(*v, projected_bump_in_value);

        assert_tensors_are_close(*bumped_value, projected_bumped_value, derivative_tolerance,
                std::string("derivative for function ") + name + " failed to project");
    }
}
} // end anonymous namespace

namespace para {
namespace graph {

std::string tensor_function_factory_add_test::name() const {
    return "tensor_function_factory_add_test";
}

void tensor_function_factory_add_test::run() const {
    tensor::N_vector t1_dims { 2, 3 };
    auto t1 = generate_random_tensor(t1_dims, dre);
    auto t2 = generate_random_tensor(t1_dims, dre);

    tensor t1_plus_t2(std::move(tensor::add(*t1, *t2)));
    test_function("add", tensor_function_factory::add(), tensor_cptr_vec { t1, t2 }, t1_plus_t2, dre);
}

std::string tensor_function_factory_chain_multiplication_test::name() const {
    return "tensor_function_factory_chain_multiplaction_test";
}

void tensor_function_factory_chain_multiplication_test::run() const {
    tensor::N_vector t1_dims { 2, 3 };
    auto t1 = generate_random_tensor(t1_dims, dre);
    tensor::N_vector t2_dims { t1_dims.back(), 5 };
    auto t2 = generate_random_tensor(t2_dims, dre);
    tensor t1_times_t2 = tensor::zero(tensor::N_vector { t1_dims.front(), t2_dims.back() });
    for (std::size_t i0 = 0; i0 < t1_dims.front(); ++i0) {
        for (std::size_t i2 = 0; i2 < t2_dims.back(); ++i2) {
            double v012 = 0;
            for (std::size_t i1 = 0; i1 < t1_dims.back(); ++i1) {
                v012 += t1->data[t1->compute_offset( { i0, i1 })] * t2->data[t2->compute_offset( { i1, i2 })];
            }
            t1_times_t2.data[t1_times_t2.compute_offset( { i0, i2 })] = v012;
        }
    }
    test_function("chain_multiplication", tensor_function_factory::chain_multiplication(1), tensor_cptr_vec { t1, t2 },
            t1_times_t2, dre);
}

std::string tensor_function_factory_sigmoid_test::name() const {
    return "tensor_function_factory_sigmoid_test";
}

void tensor_function_factory_sigmoid_test::run() const {
    tensor::N_vector dims_in { 2, 3 };
    auto t_in = generate_random_tensor(dims_in, dre);
    std::vector<double> t_out_data(t_in->data.size(), 0);
    std::transform(t_in->data.begin(), t_in->data.end(), t_out_data.begin(),
            [](double a) {return 1.0 / ( 1.0 + std::exp(-a));});
    tensor t_out(t_in->dimensionalities, std::move(t_out_data));
    test_function("sigmoid", tensor_function_factory::sigmoid(), tensor_cptr_vec { t_in }, t_out, dre);
}

std::string tensor_function_factory_reduce_sum_test::name() const {
    return "tensor_function_factory_reduce_sum_test";
}

void tensor_function_factory_reduce_sum_test::run() const {
    tensor::N ldim = 2, cdim = 3, rdim = 5;
    auto t_in = generate_random_tensor( { ldim, cdim, rdim }, dre);
    tensor t_out(std::move(tensor::zero( { ldim, rdim })));
    for (tensor::N il = 0; il < ldim; ++il) {
        for (tensor::N ir = 0; ir < rdim; ++ir) {
            auto out_offset = t_out.compute_offset( { il, ir });
            for (tensor::N ic = 0; ic < cdim; ++ic) {
                auto in_offset = t_in->compute_offset( { il, ic, ir });
                t_out.data[out_offset] += t_in->data[in_offset];
            }
        }
    }
    test_function("reduce_sum", tensor_function_factory::reduce_sum(1), tensor_cptr_vec { t_in }, t_out, dre);
}

std::string tensor_function_factory_log_test::name() const {
    return "tensor_function_factory_log_test";
}

void tensor_function_factory_log_test::run() const {
    auto t_in = generate_random_tensor( { 2, 3 }, dre);
    tensor t_out(std::move(tensor::zero(t_in->dimensionalities)));
    std::transform(t_in->data.begin(), t_in->data.end(), t_out.data.begin(), [](double x) {return std::log(x);});
    test_function("log", tensor_function_factory::log(), { t_in }, t_out, dre);
}

std::string tensor_function_factory_element_wise_multiplication_test::name() const {
    return "tensor_function_factory_element_wise_multiplication_test";
}

void tensor_function_factory_element_wise_multiplication_test::run() const {
    auto t1 = generate_random_tensor( { 2, 3 }, dre);
    auto t2 = generate_random_tensor(t1->dimensionalities, dre);
    tensor t_out(std::move(tensor::zero(t1->dimensionalities)));
    for (std::size_t it = 0; it < t_out.data.size(); ++it) {
        t_out.data[it] = t1->data[it] * t2->data[it];
    }
    test_function("element_wise_multiplication", tensor_function_factory::element_wise_multiplication(), { t1, t2 },
            t_out, dre);
}

std::string tensor_function_factory_negative_test::name() const {
    return "tensor_function_factory_negative_test";
}

void tensor_function_factory_negative_test::run() const {
    auto t = generate_random_tensor( { 2, 3 }, dre);
    tensor t_out = *t;
    for (auto &t_out_value : t_out.data)
        t_out_value *= -1;
    test_function("negative", tensor_function_factory::negative(), { t }, t_out, dre);
}

std::string tensor_function_factory_softmax_test::name() const {
    return "tensor_function_factory_softmax_test";
}

void tensor_function_factory_softmax_test::run() const {
    auto t = generate_random_tensor( { 2, 3 }, dre);
    tensor t_out = *t;
    double total = 0.0;
    for (auto &t_out_value : t_out.data) {
        t_out_value = std::exp(t_out_value);
        total += t_out_value;
    }
    for (auto &t_out_value : t_out.data) {
        t_out_value /= total;
    }
    test_function("softmax", tensor_function_factory::softmax(), { t }, t_out, dre);
}

} // end namespace graph
} // end namespace para

