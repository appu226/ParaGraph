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


#include "graph_test.h"
#include "graph_test_utils.h"
#include <para/graph/math.h>
#include <para/graph/graph.h>
#include <para/graph/exception.h>
#include <para/graph/ml_graph.h>
#include <algorithm>
#include <random>

namespace {
using namespace para::graph;

struct w_x_plus_b {
    graph_cuptr g;
    variable w;
    variable x;
    variable b;
    operation output;

    w_x_plus_b(int common_dims) :
                    w(-1),
                    x(-1),
                    b(-1),
                    output(-1) {
        auto gb = graph_builder::empty();
        w = gb->add_variable("w");
        x = gb->add_variable("x");
        b = gb->add_variable("b");

        auto wx = gb->add_operation("wx", multiply(common_dims), std::vector<node> { w, x });
        output = gb->add_operation("wx+b", add(), std::vector<node> { wx, b });

        g = gb->build_graph();
    }

    static tensor_function_csptr multiply(int num_common_dims) {
        return tensor_function_factory::chain_multiplication(num_common_dims);
    }

    static tensor_function_csptr add() {
        struct ad: tensor_function {
            tensor_cptr value(const tensor_cptr_vec& inputs) const override {
                assert(inputs.size() == 2, "::ad::value can only take inputs of size 2.");
                return tensor_cptr(new tensor(std::move(tensor::add(*inputs[0], *inputs[1]))));
            }
            derivative deriv(const tensor_cptr_vec& inputs) const override {
                assert(inputs.size() == 2, "::ad::deriv can only take inputs of size 2.");
                tensor_cptr v(new tensor(std::move(tensor::add(*inputs[0], *inputs[1]))));
                tensor_cptr d(new tensor(std::move(tensor::identity_derivative(inputs[0]->dimensionalities))));
                return derivative { v, std::vector<tensor_cptr> { d, d } };
            }
        };
        return tensor_function_csptr(new ad);
    }

    tensor_cptr_vec create_inputs(const tensor_cptr& w_value, const tensor_cptr& x_value,
            const tensor_cptr& b_value) const {
        graph_input_map input_map { { w, w_value }, { x, x_value }, { b, b_value } };
        return g->create_variable_values(input_map);
    }

    tensor_cptr value(const tensor_cptr& w_value, const tensor_cptr& x_value, const tensor_cptr& b_value) const {
        return g->value(output, create_inputs(w_value, x_value, b_value));
    }

    derivative deriv(const tensor_cptr& w_value, const tensor_cptr& x_value, const tensor_cptr b_value) const {
        return g->partial_gradient(output, std::vector<variable> { w, x, b }, create_inputs(w_value, x_value, b_value));
    }
};

} // end anonymous namespace

namespace para {
namespace graph {

std::string graph_scalar_test::name() const {
    return "graph_scalar_test";
}

void graph_scalar_test::run() const {

    w_x_plus_b tg(0);

    std::default_random_engine dre;
    std::uniform_real_distribution<double> urd(0, 1);
    auto rand = [&]() {return urd(dre);};

    auto scalar_to_tensor = [](double s) {return tensor_cptr(new tensor(tensor::N_vector(), std::vector<double> {s}));};
    tensor_cptr w = scalar_to_tensor(rand());
    tensor_cptr x = scalar_to_tensor(rand());
    tensor_cptr b = scalar_to_tensor(rand());

    tensor_cptr value = tg.value(w, x, b);
    assert_tensors_are_close(*value, *scalar_to_tensor(w->at(0) * x->at(0) + b->at(0)), 1e-15,
            "graph::value should work for scalars");

    derivative tgd = tg.deriv(w, x, b);
    assert_tensors_are_close(*tgd.node_value, *value, 1e-15,
            "graph::patial_gradient should return correct value for scalars");
    assert(tgd.node_derivative.size() == 3,
            "graph::partial_gradient should return node_derivative of correct size for scalars.");
    assert_tensors_are_close(*tgd.node_derivative[0], *x, 1e-15, "d/dw should be x");
    assert_tensors_are_close(*tgd.node_derivative[1], *w, 1e-15, "d/dx should be w");
    assert_tensors_are_close(*tgd.node_derivative[2], *scalar_to_tensor(1), 1e-15, "d/db should be 1");

}

std::string graph_tensor_test::name() const {
    return "graph_tensor_test";
}

void graph_tensor_test::run() const {
    std::default_random_engine dre;

    tensor_cptr w = generate_random_tensor(tensor::N_vector { 2, 3 }, dre);
    tensor_cptr x = generate_random_tensor(tensor::N_vector { 3, 5 }, dre);
    tensor_cptr b = generate_random_tensor(tensor::N_vector { 2, 5 }, dre);

    const int num_common_dims = 1;
    w_x_plus_b tg(num_common_dims);
    derivative dtg = tg.deriv(w, x, b);
    tensor_cptr computed_base_value = dtg.node_value;
    tensor expected_base_value = tensor::add(tensor::chain_multiplication(*w, *x, num_common_dims), *b);
    assert_tensors_are_close(*computed_base_value, expected_base_value, 1e-15,
            "graph::value should reutrn correct tensor.");

    tensor_cptr dw = generate_random_tensor(w->dimensionalities, dre);
    tensor_cptr w2(new tensor(std::move(tensor::add(*w, *dw))));
    tensor_cptr computed_w_bumped_value = tg.value(w2, x, b);
    const tensor& dvdw = *dtg.node_derivative[0];
    tensor projected_w_bumped_value = tensor::add(expected_base_value,
            tensor::chain_multiplication(*dw, dvdw, w->dimensionalities.size()));
    assert_tensors_are_close(*computed_w_bumped_value, projected_w_bumped_value, 1e-15,
            "graph::deriv should return a derivative that can predict values accurately for linear functions.");

    tensor_cptr dx = generate_random_tensor(x->dimensionalities, dre);
    tensor_cptr x2(new tensor(std::move(tensor::add(*x, *dx))));
    tensor_cptr computed_x_bumped_value = tg.value(w, x2, b);
    const tensor& dvdx = *dtg.node_derivative[1];
    tensor projected_x_bumped_value = tensor::add(expected_base_value,
            tensor::chain_multiplication(*dx, dvdx, x->dimensionalities.size()));
    assert_tensors_are_close(*computed_x_bumped_value, projected_x_bumped_value, 1e-15,
            "graph::deriv should return a derivative that can predict values accurately for linear functions.");

    tensor_cptr db = generate_random_tensor(b->dimensionalities, dre);
    tensor_cptr b2(new tensor(std::move(tensor::add(*b, *db))));
    tensor_cptr computed_b_bumped_value = tg.value(w, x, b2);
    const tensor& dvdb = *dtg.node_derivative[2];
    tensor projected_b_bumped_value = tensor::add(expected_base_value,
            tensor::chain_multiplication(*db, dvdb, b->dimensionalities.size()));
    assert_tensors_are_close(*computed_b_bumped_value, projected_b_bumped_value, 1e-15,
            "graph::deriv should return a derivative that can predict values accurately for linear functions.");

}

} // end namespace graph
} // end namespace para

