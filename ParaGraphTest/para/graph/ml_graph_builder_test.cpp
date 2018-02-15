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
#include <random>
#include <algorithm>

namespace {
using namespace para::graph;

void test_graph(graph_cuptr g, const graph_input_map& inputs, node output_node, const tensor& expected_value,
        std::default_random_engine& dre, const std::string& test_name) {
    const double value_tolerance = 1e-15;
    const auto input_vec = g->create_variable_values(inputs);
    assert_tensors_are_close(expected_value, *g->value(output_node, input_vec), value_tolerance,
            "value did not match in " + test_name);

    std::vector<variable> input_vars;
    const auto first_getter = [](const graph_input_map::value_type & kv) {return kv.first;};
    std::transform(inputs.begin(), inputs.end(), std::back_inserter(input_vars), first_getter);
    const auto deriv = g->partial_gradient(output_node, input_vars, input_vec);
    assert_tensors_are_close(expected_value, *deriv.node_value, value_tolerance,
            "derivative.node_value did not match in " + test_name);

    assert(deriv.node_derivative.size() == input_vars.size(),
            "node derivative size should equal input vars size in " + test_name);

    //TODO: check the partial derivatives
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
    auto w = mgbu->add_variable("w");
    auto x = mgbu->add_variable("x");
    auto b = mgbu->add_variable("b");
    auto wx = mgbu->chain_multiplication(w, x, 1);
    auto wxpb = mgbu->add(wx, b);

    auto w_val = generate_random_tensor( { 2, 3, 5 }, dre);
    auto x_val = generate_random_tensor( { 5, 7 }, dre);
    auto b_val = generate_random_tensor( { 2, 3, 7 }, dre);
    auto wxpb_val = tensor::add(tensor::chain_multiplication(*w_val, *x_val, 1), *b_val);

    graph_input_map wxb_input_map { { w, w_val }, { x, x_val }, { b, b_val } };
    test_graph(mgbu->build_graph(), wxb_input_map, wxpb, wxpb_val, dre, "wx_plus_b");
}

} // end namespace graph
} // end namespace para

