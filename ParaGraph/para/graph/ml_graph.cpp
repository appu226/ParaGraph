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

#include "ml_graph.h"
#include "exception.h"

#include <algorithm>
#include <sstream>

namespace {
using namespace para::graph;

struct tensor_function_chain_multiplication: tensor_function {
    int num_common_dims;
    tensor_function_chain_multiplication(int ncd) :
                    num_common_dims(ncd) {
    }
    tensor_cptr value(const tensor_cptr_vec& inputs) const override {
        assert(inputs.size() == 2, "::mult::value can only work with two inputs.");
        return tensor_cptr(new tensor(std::move(tensor::chain_multiplication(*inputs[0], *inputs[1], num_common_dims))));
    }
    derivative deriv(const tensor_cptr_vec& inputs) const override {
        assert(inputs.size() == 2, "::mult::deriv can only work with two inputs.");
        tensor_cptr v(new tensor(std::move(tensor::chain_multiplication(*inputs[0], *inputs[1], num_common_dims))));
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
        CN n = calc_size(A.dimensionalities.end() - num_common_dims, A.dimensionalities.end());
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
                for (std::size_t l = 0; l < n; ++l) {
                    dCdA.data[k * k_dcda + l * l_dcda + i * i_dcda + j * j_dcda] = B.data[l * l_b + j * j_b];
                }
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
// end struct tensor_function_chain_multiplication

struct ml_graph_builder_impl: ml_graph_builder {
    typedef std::vector<node> node_vec;
    ml_graph_builder_impl() :
                    gb(graph_builder::empty()),
                    counter(0) {

    }

    variable add_variable(const std::string& name) override {
        return gb->add_variable(name);
    }
    operation add_operation(const std::string& name, const tensor_function_csptr& function,
            const std::vector<node>& dependencies) override {
        return gb->add_operation(name, function, dependencies);
    }
    operation add(node lhs, node rhs) override {
        return add_operation(uid("add"), tensor_function_factory::add(), node_vec { lhs, rhs });
    }
    operation chain_multiplication(node lhs, node rhs, int num_common_dims) override {
        return add_operation(uid("chain_multiplication"),
                tensor_function_factory::chain_multiplication(num_common_dims), node_vec { lhs, rhs });
    }
    operation sigmoid(node n) override {
        return add_operation(uid("sigmoid"), tensor_function_factory::sigmoid(), node_vec { n });
    }
    operation reduce_sum(node n, int axis) override {
        return add_operation(uid("reduce_sum"), tensor_function_factory::reduce_sum(axis), node_vec { n });
    }
    operation log(node n) override {
        return add_operation(uid("log"), tensor_function_factory::log(), node_vec { n });
    }
    operation element_wise_multiplication(node lhs, node rhs) override {
        return add_operation(uid("element_wise_multiplication"), tensor_function_factory::element_wise_multiplication(),
                node_vec { lhs, rhs });
    }
    operation negative(node n) override {
        return add_operation(uid("negative"), tensor_function_factory::negative(), node_vec { n });
    }
    operation softmax(node n) override {
        return add_operation(uid("softmax"), tensor_function_factory::softmax(), node_vec { n });
    }

    graph_cuptr build_graph() const override {
        return gb->build_graph();
    }

    graph_builder_uptr gb;
    int counter;

    std::string uid(const char* prefix) {
        std::stringstream ss;
        ss << prefix << "_" << ++counter;
        return ss.str();
    }
};

} // end anonymous namespace

namespace para {
namespace graph {

tensor_function_csptr tensor_function_factory::add() {
    struct tensor_function_add: tensor_function {
        tensor_cptr value(const tensor_cptr_vec& tv) const override {
            assert(tv.size() == 2, "tensor_function_add only works with two inputs.");
            assert(tv[0]->dimensionalities == tv[1]->dimensionalities,
                    "tensor_function_add only works if input values have the same dimensionality.");
            std::vector<double> data(tv[0]->data);
            const std::size_t size = tv[0]->data.size();
            const auto& rhs = tv[1]->data;
            for (std::size_t i = 0; i < size; ++i) {
                data[i] += rhs[i];
            }
            return tensor_cptr(new tensor(tv[0]->dimensionalities, std::move(data)));
        }

        derivative deriv(const tensor_cptr_vec& tv) const override {
            tensor_cptr v = value(tv);
            tensor_cptr d(new tensor(std::move(tensor::identity_derivative(tv[0]->dimensionalities))));
            return derivative { v, tensor_cptr_vec { d, d } };
        }
    };
    return tensor_function_csptr(new tensor_function_add);
}

tensor_function_csptr tensor_function_factory::chain_multiplication(int num_common_dims) {
    return tensor_function_csptr(new tensor_function_chain_multiplication(num_common_dims));
}

tensor_function_csptr tensor_function_factory::sigmoid() {
    struct tensor_function_sigmoid: tensor_function {
        tensor_cptr value(const tensor_cptr_vec& tv) const override {
            assert(tv.size() == 1, "sigmoid only works on a single input.");
            std::vector<double> data(tv[0]->data);
            std::for_each(data.begin(), data.end(), [](double& d) {d = 1.0 / (1 + std::exp(-d));});
            return tensor_cptr(new tensor(tv[0]->dimensionalities, std::move(data)));
        }
        derivative deriv(const tensor_cptr_vec& tv) const override {
            tensor_cptr v = value(tv);
            tensor d(std::move(tensor::zero_derivative(v->dimensionalities, tv[0]->dimensionalities)));
            const std::size_t step_size = tv[0]->data.size() + 1;
            const std::size_t max_size = d.data.size();
            for (std::size_t i_d = 0, i_tv = 0, i_v = 0; i_d < max_size; i_d += step_size, ++i_tv, ++i_v) {
                d.data[i_d] = std::exp(-tv[0]->data[i_tv]) * v->data[i_v] * v->data[i_v];
            }
            return derivative { v, tensor_cptr_vec { tensor_cptr(new tensor(std::move(d))) } };
        }
    };
    return tensor_function_csptr(new tensor_function_sigmoid);
}

tensor_function_csptr tensor_function_factory::reduce_sum(int axis) {
    struct tensor_function_reduce_sum: tensor_function {
        int axis;
        tensor_function_reduce_sum(int v_axis) :
                        axis(v_axis) {
        }
        tensor_cptr value(const tensor_cptr_vec& tv) const override {
            assert(tv.size() == 1, "reduce_sum only works on a single input.");
            const tensor& input = *tv[0];
            assert(0 <= axis && input.dimensionalities.size() > static_cast<tensor::N>(axis),
                    "reduce_sum cannot reduce input with order ", input.dimensionalities.size(), " on axis ", axis);

            auto mult_func = [](tensor::N acc, tensor::N elem) {return acc * elem;};
            typedef const std::size_t N;
            using std::accumulate;
            const tensor::N_vector& idims = input.dimensionalities;
            N r_size = accumulate(idims.begin() + axis + 1, idims.end(), 1, mult_func);
            N l_size = accumulate(idims.begin(), idims.begin() + axis, 1, mult_func);
            N c_size = idims[axis];
            N input_size = input.data.size();
            std::vector<double> data(r_size * l_size, 0);
            for (std::size_t i_input = 0; i_input < input_size; ++i_input) {
                N i_data_right = i_input % r_size;
                N i_data_left = i_input / r_size / c_size;
                N i_data = i_data_right + i_data_left * r_size;
                data[i_data] += input.data[i_input];
            }
            tensor::N_vector odims(idims);
            odims.erase(odims.begin() + axis);
            return tensor_cptr(new tensor(std::move(odims), std::move(data)));
        }
        derivative deriv(const tensor_cptr_vec& tv) const override {
            assert(tv.size() == 1, "reduce_sum only works on a single input.");
            const tensor& input = *tv[0];
            assert(0 <= axis && input.dimensionalities.size() > static_cast<tensor::N>(axis),
                    "reduce_sum cannot reduce input with order ", input.dimensionalities.size(), " on axis ", axis);

            auto mult_func = [](tensor::N acc, tensor::N elem) {return acc * elem;};
            typedef const std::size_t N;
            using std::accumulate;
            const tensor::N_vector& idims = input.dimensionalities;
            N r_size = accumulate(idims.begin() + axis + 1, idims.end(), 1, mult_func);
            N l_size = accumulate(idims.begin(), idims.begin() + axis, 1, mult_func);
            N c_size = idims[axis];

            N v_size = r_size * l_size * c_size;
            N o_size = r_size * l_size;
            std::vector<double> data(v_size * o_size, 0);
            for (std::size_t v_offset = 0; v_offset < v_size; ++v_offset) {
                N o_offset_l = v_offset / (r_size * c_size);
                N o_offset_r = v_offset % r_size;
                N o_offset = o_offset_l * r_size + o_offset_r;
                N offset = v_offset * o_size + o_offset;
                data[offset] = 1;
            }
            tensor::N_vector odims(idims.size() * 2 - 1);
            std::copy(idims.begin(), idims.end(), odims.begin());
            std::copy(idims.begin(), idims.begin() + axis, odims.begin() + idims.size());
            std::copy(idims.begin() + axis + 1, idims.end(), odims.begin() + idims.size() + axis);
            return derivative { value(tv),
                    tensor_cptr_vec { tensor_cptr(new tensor(std::move(odims), std::move(data))) } };
        }
    };
    return tensor_function_csptr(new tensor_function_reduce_sum { axis });
}

tensor_function_csptr tensor_function_factory::log() {
    struct tensor_function_log: tensor_function {
        tensor_cptr value(const tensor_cptr_vec& tv) const override {
            assert(tv.size() == 1, "log only works on a single input.");
            std::vector<double> data(tv[0]->data);
            std::for_each(data.begin(), data.end(), [](double& d) {d = std::log(d);});
            return tensor_cptr(new tensor(tv[0]->dimensionalities, std::move(data)));
        }
        derivative deriv(const tensor_cptr_vec& tv) const override {
            tensor_cptr v = value(tv);
            tensor d(std::move(tensor::zero_derivative(v->dimensionalities, tv[0]->dimensionalities)));
            const std::size_t step_size = tv[0]->data.size() + 1;
            const std::size_t max_size = d.data.size();
            for (std::size_t i_d = 0, i_tv = 0; i_d < max_size; i_d += step_size, ++i_tv) {
                d.data[i_d] = 1 / tv[0]->data[i_tv];
            }
            return derivative { v, tensor_cptr_vec { tensor_cptr(new tensor(std::move(d))) } };
        }
    };
    return tensor_function_csptr(new tensor_function_log);
}

tensor_function_csptr tensor_function_factory::element_wise_multiplication() {
    struct tensor_function_ewmult: tensor_function {
        tensor_cptr value(const tensor_cptr_vec& tv) const override {
            assert(tv.size() == 2,
                    "element wise multiplication currently implemented to work with exactly 2 inputs, found ",
                    tv.size());
            const tensor& lhs = *tv[0], rhs = *tv[1];
            assert(lhs.dimensionalities == rhs.dimensionalities,
                    "Inputs to elemenent wise multiplication are expected to have identical dimensionalities.");
            std::vector<double> data(lhs.data.size());
            std::transform(lhs.data.begin(), lhs.data.end(), rhs.data.begin(), data.begin(),
                    [](double l, double r) {return l * r;});
            return tensor_cptr(new tensor(lhs.dimensionalities, std::move(data)));

        }
        derivative deriv(const tensor_cptr_vec& tv) const override {
            tensor_cptr v = value(tv);
            return derivative { v, tensor_cptr_vec { mult_deriv(*tv[1]), mult_deriv(*tv[0]) } };
        }
        static tensor_cptr mult_deriv(const tensor& t) {
            tensor::N_vector odim = t.dimensionalities;
            odim.insert(odim.end(), t.dimensionalities.begin(), t.dimensionalities.end());
            const std::size_t t_size = t.data.size();
            const std::size_t step_size = t_size + 1;
            std::vector<double> odata(t_size * t_size);
            for (std::size_t i_odata = 0, i_t = 0; i_t < t_size; i_odata += step_size, ++i_t) {
                odata[i_odata] = t.data[i_t];
            }
            return tensor_cptr(new tensor(std::move(odim), std::move(odata)));
        }
    };
    return tensor_function_csptr(new tensor_function_ewmult);
}

tensor_function_csptr tensor_function_factory::negative() {
    struct tensor_function_negative: tensor_function {
        tensor_cptr value(const tensor_cptr_vec& tv) const override {
            assert(tv.size() == 1, "negative only works on a single input.");
            std::vector<double> data(tv[0]->data);
            std::for_each(data.begin(), data.end(), [](double& d) {d *= -1;});
            return tensor_cptr(new tensor(tv[0]->dimensionalities, std::move(data)));
        }
        derivative deriv(const tensor_cptr_vec& tv) const override {
            tensor_cptr v = value(tv);
            tensor d(std::move(tensor::identity_derivative(v->dimensionalities)));
            for (auto &d_value : d.data)
                d_value *= -1;
            return derivative { v, tensor_cptr_vec { tensor_cptr(new tensor(std::move(d))) } };
        }
    };
    return tensor_function_csptr(new tensor_function_negative);
}

tensor_function_csptr tensor_function_factory::softmax() {
    struct tensor_function_softmax: tensor_function {
        typedef std::pair<double, tensor_cptr> DxT;
        DxT C_and_F(const tensor_cptr_vec& tv) const {
            /*
             * Let F be the softmax of input V
             * Then 
             *   F_i = exp( V_i ) / C
             * Where
             *          n
             *    C  =  ∑ exp( V_k )
             *         k=1 
             */
            assert(tv.size() == 1, "softmax only works on a single input.");
            std::vector<double> F;
            auto const & V = tv[0]->data;
            F.reserve(V.size());
            double C = 0;
            std::transform(V.begin(), V.end(), std::back_inserter(F), [&C](const double in) {
                const double inexp = std::exp(in);
                C += inexp;
                return inexp;
            });
            std::for_each(F.begin(), F.end(), [C](double &in) {in /= C;});
            return std::move(DxT(C, tensor_cptr(new tensor(tv[0]->dimensionalities, std::move(F)))));
        }

        tensor_cptr value(const tensor_cptr_vec& tv) const override {
            return std::move(C_and_F(tv).second);
        }

        derivative deriv(const tensor_cptr_vec& tv) const override {
            /*
             * Let D be the gradient of F wrt V
             * Then
             *          ∂F_j                ∂   1           1     ∂ exp(V_j)
             *   D_ij = ---- = exp(V_j) x ---- ---    +    --- x ------------
             *          ∂V_i              ∂V_i  C           C     ∂   V_i
             * For i == j, we get:
             *                      -1       ∂C             1 
             *   D_ii = exp(V_i) x ----- x ------     +    --- x exp(V_i)      
             *                      C^2     ∂V_i            C
             *           -exp(2*V_i)                        exp(V_i)
             *        = -------------                 +    ----------
             *               C^2                              C
             *                        -              -
             *           exp(V_i)    |      exp(V_i)  |
             *        = ---------- x | 1 - ---------- |
             *              C        |         C      |
             *                        -              -
             * And for i != j, we get:
             *                      -1   ∂ C               1
             *   D_ij = exp(V_j) x ----- ----        +    --- x 0
             *                      C^2  ∂V_i              C
             *             - exp(V_i + V_j)
             *        =  --------------------
             *                     C^2
             */
            auto const cf = C_and_F(tv);
            auto const C = cf.first;
            auto const & F = cf.second;
            auto const V = tv[0]->data;
            auto const f_size = F->data.size();
            auto const d_size = f_size * f_size;
            auto const Csq = C * C;
            std::vector<double> D(d_size);
            for (std::size_t ij = 0; ij < d_size; ++ij) {
                auto const i = ij / f_size;
                auto const j = ij % f_size;
                if (i == j) {
                    auto const evi_by_c = std::exp(V[i]) / C;
                    D[ij] = evi_by_c * (1 - evi_by_c);
                } else {
                    D[ij] = -std::exp(V[i] + V[j]) / Csq;
                }
            }
            auto D_dim = F->dimensionalities;
            D_dim.insert(D_dim.end(), F->dimensionalities.begin(), F->dimensionalities.end());
            return derivative { F, tensor_cptr_vec(1, tensor_cptr(new tensor(std::move(D_dim), std::move(D)))) };
        }
    };
    return tensor_function_csptr(new tensor_function_softmax);
}

ml_graph_builder_uptr ml_graph_builder::empty() {
    return uptr(new ml_graph_builder_impl);
}

ml_graph_builder::~ml_graph_builder() {
}

}
// end namespace graph
}// end namespace para
