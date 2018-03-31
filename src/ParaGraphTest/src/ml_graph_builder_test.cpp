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

#include "ml_graph_builder_test.h"
#include "graph_test_utils.h"
#include <para/graph/ml_graph.h>
#include <para/graph/exception.h>
#include <para/graph/functional.h>
#include <random>
#include <algorithm>

#include <iostream>

namespace {
using namespace para::graph;

void test_graph(graph_cuptr g, const graph_input_map& inputs, node output_node, const tensor& expected_value,
        std::default_random_engine& dre, const std::string& test_name) {
    const double value_tolerance = 1e-15;
    const double shift_size = 1e-6;
    const double shifted_tolerance = shift_size * shift_size * 100 * inputs.size();
    const auto input_vec = g->create_variable_values(inputs);
    assert_tensors_are_close(expected_value, *g->value(output_node, input_vec), value_tolerance,
            "value did not match in " + test_name);

    const auto input_vars = functional(inputs).keys();
    const auto deriv = g->partial_gradient(output_node, input_vars, input_vec);
    assert_tensors_are_close(expected_value, *deriv.node_value, value_tolerance,
            "derivative.node_value did not match in " + test_name);

    assert(deriv.node_derivative.size() == input_vars.size(),
            "node derivative size should equal input vars size in " + test_name);

    const auto input_vals = functional(input_vars).map<tensor_cptr>([&](const variable v) {return inputs.at(v);});
    auto input_shift_buffer = functional(input_vars).zipToMap(input_vals);
    for (size_t i_input = 0; i_input < input_vars.size(); ++i_input) {
        const tensor_cptr input_value = input_vals[i_input];
        const tensor_cptr input_delta = generate_random_tensor(input_value->dimensionalities, dre, shift_size);
        const tensor_cptr& partial_derivative = deriv.node_derivative[i_input];
        const tensor function_delta(
                std::move(
                        tensor::chain_multiplication(*input_delta, *partial_derivative,
                                input_value->dimensionalities.size())));
        const tensor projected_function_shifted(std::move(tensor::add(expected_value, function_delta)));

        const tensor_cptr input_shifted(new tensor(std::move(tensor::add(*input_value, *input_delta))));
        input_shift_buffer[input_vars[i_input]] = input_shifted; // temporarily shift input in shift buffer
        const tensor_cptr actual_function_shifted = g->value(output_node,
                g->create_variable_values(input_shift_buffer));
        assert_tensors_are_close(projected_function_shifted, *actual_function_shifted, shifted_tolerance,
                "Projected bump should match actual bump for " + test_name);
        input_shift_buffer[input_vars[i_input]] = input_value;   // reset input to original value in shift_buffer
    }
}
} // end anonymous namespace

namespace para {
namespace graph {

std::string ml_graph_builder_test::name() const {
    return "ml_graph_builder_test";
}

void ml_graph_builder_test::run() const {
    auto mgbu = ml_graph_builder::empty();
    std::default_random_engine dre;

// test w*x+b
    auto const w = mgbu->add_variable("w");
    auto const x = mgbu->add_variable("x");
    auto const b = mgbu->add_variable("b");
    auto const wx = mgbu->chain_multiplication(w, x, 1);
    auto const wxpb = mgbu->add(wx, b);

    std::size_t const point_dim = 5;
    std::size_t const num_classes = 3;
    std::size_t const num_points = 17;
    auto const w_val = generate_random_tensor( { num_classes, point_dim }, dre);
    auto const x_val = generate_random_tensor( { point_dim, num_points }, dre);
    auto const b_val = generate_random_tensor( { num_classes, num_points }, dre);
    auto const wxpb_val = tensor::add(tensor::chain_multiplication(*w_val, *x_val, 1), *b_val);

    graph_input_map wxb_input_map { { w, w_val }, { x, x_val }, { b, b_val } };
    test_graph(mgbu->build_graph(), wxb_input_map, wxpb, wxpb_val, dre, "wx_plus_b");

    auto const c = mgbu->add_variable("c");
    auto const p = mgbu->softmax(wxpb);
    auto const j = mgbu->negative(
            mgbu->reduce_sum(mgbu->reduce_sum(mgbu->element_wise_multiplication(c, mgbu->log(p)), 0), 0));

    auto const c_val = generate_random_tensor( { num_classes, num_points }, dre);
    auto const p_val = tensor_function_factory::log()->value( { tensor_cptr(new tensor(wxpb_val)) });
    auto const j_val = tensor_function_factory::negative()->value(
            { tensor_function_factory::reduce_sum(0)->value( { tensor_function_factory::reduce_sum(0)->value( {
                    tensor_function_factory::element_wise_multiplication()->value( { c_val,
                            tensor_function_factory::log()->value( { p_val }) }) }) }) });

    graph_input_map wxbc_input_map { { w, w_val }, { x, x_val }, { b, b_val }, { c, c_val } };
    test_graph(mgbu->build_graph(), wxbc_input_map, j, *j_val, dre, "J");

}

} // end namespace graph
} // end namespace para

