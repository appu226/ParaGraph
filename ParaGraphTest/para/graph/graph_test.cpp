/*
 * graph_test.cpp
 *
 *  Created on: 09-Jan-2018
 *      Author: parakram
 */

#include "graph_test.h"
#include "graph_test_utils.h"
#include <para/graph/math.h>
#include <para/graph/graph.h>
#include <para/graph/exception.h>
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
        struct mult: tensor_function {
            int num_common_dims;
            mult(int ncd) :
                            num_common_dims(ncd) {
            }
            tensor_cptr value(const tensor_cptr_vec& inputs) const override {
                assert(inputs.size() == 2, "::mult::value can only work with two inputs.");
                return tensor_cptr(
                        new tensor(std::move(tensor::chain_multiplication(*inputs[0], *inputs[1], num_common_dims))));
            }
            derivative deriv(const tensor_cptr_vec& inputs) const override {
                assert(inputs.size() == 2, "::mult::deriv can only work with two inputs.");
                tensor_cptr v(
                        new tensor(std::move(tensor::chain_multiplication(*inputs[0], *inputs[1], num_common_dims))));
                /*
                 *  Let:
                 *       A  X  B  =  C
                 *      mxn X nxp = mxp
                 *
                 *  In other words:
                 *               n
                 *      C[i,j] = ∑ A[i,k]∙B[k,j]
                 *              k=1
                 *
                 *  Therefore:
                 *      ∂C[i,j]       ∂       n
                 *      --------  =  ----   ( ∑  A[i,r]∙B[r,j])
                 *      ∂A[k,l]     ∂A[k,l]  r=1
                 *
                 *                = {  B[l,j] if i==k
                 *                  {  0      otherwise
                 *
                 *                = dCdA[k,l,i,j]
                 *
                 *  And:
                 *      ∂C[i,j]       ∂       n
                 *      --------  =  ----   ( ∑  A[i,r]∙B[r,j])
                 *      ∂B[k,l]     ∂B[k,l]  r=1
                 *
                 *                = {  A[i,k] if l==j
                 *                  {  0      otherwise
                 *
                 *                = dCdB[k,l,i,j]
                 */
                const tensor& A = *inputs[0];
                const tensor& B = *inputs[1];
                const tensor& C = *v;

                auto calc_size = [](tensor::N_vector::const_iterator begin, tensor::N_vector::const_iterator end) {
                    auto mult_func = [](tensor::N acc, tensor::N next) {return acc * next;};
                    return std::accumulate(begin, end, 1, mult_func);
                };
                typedef std::size_t N;
                typedef const N CN;
                CN m = calc_size(A.dimensionalities.begin(), A.dimensionalities.end() - num_common_dims);
                CN n = calc_size(A.dimensionalities.begin() + num_common_dims, A.dimensionalities.end());
                CN p = calc_size(B.dimensionalities.begin() + num_common_dims, B.dimensionalities.end());

                // set dC/dA for all k, l, i, j using:
                //     dCdA[k,l,i,j] = if (i==k) B[l,j] else 0
                tensor dCdA(std::move(tensor::zero_derivative(C.dimensionalities, A.dimensionalities)));
                // compute step sizes for k, l, i, j for use as offset in dCdA
                CN j_dcda = 1;
                CN i_dcda = j_dcda * p;
                CN l_dcda = i_dcda * m;
                CN k_dcda = l_dcda * n;
                //compute step sizes for l, j for use as offset in B
                CN j_b = 1;
                CN l_b = j_b * p;
                // loop over all k, l, i, j
                for (N i = 0; i < m; ++i) {
                    // we care only when k = i, since otherwise dC/dA is zero
                    CN k = i;
                    for (std::size_t j = 0; j < p; ++j)
                        for (std::size_t l = 0; l < n; ++l)
                            dCdA.data[k * k_dcda + l * l_dcda + i * i_dcda + j * j_dcda] = B.data[l * l_b + j * j_b];
                }

                // set dC/dB for all k, l, i, j using:
                //    dCdB[k,l,i,j] =  if (l==j) A[i,k] else 0
                tensor dCdB(std::move(tensor::zero_derivative(C.dimensionalities, B.dimensionalities)));
                // compute step sizes for k, l, i, j for use as offset in dCdB
                CN j_dcdb = 1;
                CN i_dcdb = j_dcdb * p;
                CN l_dcdb = i_dcdb * m;
                CN k_dcdb = l_dcdb * p;
                // compute step sizes for i, k for use as offset in A
                CN k_a = 1;
                CN i_a = k_a * n;
                // loop over all k, l, i, j
                for (N l = 0; l < p; ++l) {
                    // we care only when l = j, since otherwise dC/dB is zero
                    CN j = l;
                    for (N k = 0; k < n; ++k) {
                        for (N i = 0; i < m; ++i) {
                            dCdB.data[k * k_dcdb + l * l_dcdb + i * i_dcdb + j * j_dcdb] = A.data[i * i_a + k * k_a];
                        }
                    }
                }

                // create tensor_cptrs of derivatives
                tensor_cptr d1(new tensor(std::move(dCdA)));
                tensor_cptr d2(new tensor(std::move(dCdB)));

                return derivative { v, tensor_cptr_vec { d1, d2 } };
            }
        };
        return tensor_function_csptr(new mult(num_common_dims));
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
    assert_tensors_are_close(*value, *scalar_to_tensor(w->data[0] * x->data[0] + b->data[0]), 1e-15,
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

