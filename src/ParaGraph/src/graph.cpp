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

#include <para/graph/graph.h>
#include <para/graph/exception.h>
#include <algorithm>
#include <iostream>

#define NYI throw std::logic_error("Not yet implemented.")
#define URC throw std::logic_error("Unreachable code.")

namespace {

using namespace para::graph;

struct variable_impl {
    std::string name;
    int index;
    std::vector<operation> consumers;
    int highest_consumer_operation_index;
};
struct operation_impl {
    std::string name;
    int index;
    tensor_function_csptr function;
    std::vector<operation> consumers;
    std::vector<node> dependencies;
    int highest_consumer_operation_index;
};

struct graph_impl: para::graph::graph {

    std::vector<variable_impl> variables;
    std::vector<operation_impl> operations;

    tensor_cptr value(node output_node, const tensor_cptr_vec& input_values) const override {
        switch (output_node.type) {
        case node::nt_variable: {
            // if output_node is a variable, just return it's value
            return input_values[output_node.index];
        }
        case node::nt_operation: {
            // find dependency operations
            std::vector<bool> is_dependency = all_dependency_operations(output_node);

            // initialize storage space for computed values
            tensor_cptr_vec storage(operations.size());

            // no need to compute variable nodes since their values are already provided

            // for each dependency node, in topological order, compute the value
            for (std::size_t iop = 0; iop < operations.size(); ++iop) {
                if (is_dependency[iop]) {
                    const operation_impl& op = operations[iop];
                    tensor_cptr_vec op_inputs(op.dependencies.size());
                    for (std::size_t idep = 0; idep < op.dependencies.size(); ++idep) {
                        node dep = op.dependencies[idep];
                        switch (dep.type) {
                        case node::nt_variable:
                            op_inputs[idep] = input_values[dep.index];
                            break;
                        case node::nt_operation:
                            op_inputs[idep] = storage[dep.index];
                            // free memory for dependency when done:
                            if (operations[dep.index].highest_consumer_operation_index == op.index)
                                storage[dep.index].reset();
                            break;
                        }

                    }

                    storage[iop] = op.function->value(op_inputs);
                }
            }
            return storage[output_node.index];
        }
        }
        URC;
    }

    derivative partial_gradient(node output_node, const std::vector<variable>& moving_variables,
            const tensor_cptr_vec& input_values) const override {
        switch (output_node.type) {
        case node::nt_variable: {
            derivative result { input_values[output_node.index], tensor_cptr_vec(moving_variables.size()) };
            for (std::size_t i_mv = 0; i_mv < moving_variables.size(); ++i_mv) {
                variable mv = moving_variables[i_mv];
                if (mv.index == output_node.index) {
                    auto& dim = input_values[mv.index]->dimensionalities;
                    tensor identity = std::move(tensor::identity_derivative(dim));
                    result.node_derivative[i_mv] = tensor_cptr(new tensor(std::move(identity)));
                } else {
                    auto& mv_dim = input_values[mv.index]->dimensionalities;
                    auto& output_dim = input_values[output_node.index]->dimensionalities;
                    tensor zero = std::move(tensor::zero_derivative(output_dim, mv_dim));
                    result.node_derivative[i_mv] = tensor_cptr(new tensor(std::move(zero)));
                }
            }
            return std::move(result);
        }
        case node::nt_operation: {
            // find and remember all consumer operations of the moving variables
            // compute their union U
            // find all dependency operations dep_ops of the output_node
            // for each operation O (in topological order),
            //     if O is not in dep_ops, skip
            //     if O is not in U
            //         compute the value of O
            //     else
            //         compute the value of O, as well as dO/dD for all dependencies D (virtual function call implemented by user)
            //     for each moving variable MV,
            //         set dO/dMV to 0
            //         if O is a consumer of MV
            //             for each dependency D of O,
            //                 if D is a variable
            //                      if D is same as MV
            //                          dO/dMV += dO/dD
            //                      else
            //                          do nothing
            //                 else (if D is an operation)
            //                     add to dO/dMV the chain_multiplication of dO/dD and dD/dMV
            //     for each dependency D of O,
            //         if O is the highest consumer of D
            //              for each moving variable V,
            //                  release dD/dV from memory

            // find and remember all consumer operations of the moving variables
            // compute their union U
            std::vector<std::vector<bool>> comv;
            std::vector<bool> U(operations.size(), false);
            for (variable v : moving_variables) {
                comv.push_back(all_consumer_operations(v));
                for (std::size_t i_op = 0; i_op < operations.size(); ++i_op) {
                    U[i_op] = U[i_op] or comv.back()[i_op];
                }
            }

            // find all dependency operations dep_ops of the output_node
            std::vector<bool> dep_ops = all_dependency_operations(output_node);

            std::vector<derivative> dOs_dMVs(operations.size()); // to store all dO/dMV values
                                                                 // where MV is the moving variable

            // for each operation O (in topological order),
            for (operation_impl O : operations) {
                // if O is not in dep_ops, skip
                if (!dep_ops[O.index])
                    continue;

                tensor_cptr_vec O_dep_values(O.dependencies.size()); // collecting the values of the dependencies of O for function invocations
                auto extract_O_dep_value = [&](node O_dep) {
                    switch(O_dep.type) {
                        case node::nt_variable:
                        return input_values[O_dep.index];
                        case node::nt_operation:
                        return dOs_dMVs[O_dep.index].node_value;
                    }
                    URC;
                };
                std::transform(O.dependencies.begin(), O.dependencies.end(), O_dep_values.begin(), extract_O_dep_value);

                derivative& dO_dMVs = dOs_dMVs[O.index]; // storage for the derivative (and value) of O
                derivative dOdDs; // place holder for dO/dD for all dependencies of O
                tensor_cptr& O_value = dO_dMVs.node_value; // storage for the value of O

                // if O is not in U
                if (!U[O.index]) {
                    // compute the value of O
                    O_value = O.function->value(O_dep_values);
                }
                // else
                else {
                    // compute the value of O, as well as dO/dD for all dependencies D (virtual function call implemented by user)
                    dOdDs = O.function->deriv(O_dep_values);
                    O_value = dOdDs.node_value;
                }

                // for each moving variable MV,
                for (std::size_t i_MV = 0; i_MV < moving_variables.size(); ++i_MV) {
                    variable MV = moving_variables[i_MV];
                    // set dO/dMV to 0
                    const tensor::N_vector & O_dim = O_value->dimensionalities;
                    const tensor::N_vector & MV_dim = input_values[MV.index]->dimensionalities;
                    tensor dO_dMV(std::move(tensor::zero_derivative(O_dim, MV_dim)));
                    // if O is a consumer of MV
                    if (comv[MV.index][O.index]) {
                        // for each dependency D of O,
                        for (std::size_t i_D = 0; i_D < O.dependencies.size(); ++i_D) {
                            node D = O.dependencies[i_D];
                            switch (D.type) {
                            // if D is a variable
                            case node::nt_variable:
                                // if D is same as MV
                                if (D.index == MV.index) {
                                    // dO/dMV += dO/dD
                                    dO_dMV = std::move(tensor::add(dO_dMV, *dOdDs.node_derivative[i_D]));
                                }
                                break;
                            case node::nt_operation:
                                // else (if D is an operation)
                                // dO/dMV += dO/dD * dD/dMV
                                int d_order = dOs_dMVs[D.index].node_value->dimensionalities.size();
                                tensor multiple(
                                        std::move(
                                                tensor::chain_multiplication(*dOs_dMVs[D.index].node_derivative[i_MV],
                                                        *dOdDs.node_derivative[i_D], d_order)));
                                dO_dMV = std::move(tensor::add(multiple, dO_dMV));
                            }
                        }
                    }
                    dO_dMVs.node_derivative.push_back(tensor_cptr(new tensor(std::move(dO_dMV))));
                }
                // for each dependency D of O,
                for (node D : O.dependencies) {
                    if (D.type == node::nt_operation) {
                        // if O is the highest consumer of D
                        int hcoi = operations[D.index].highest_consumer_operation_index;
                        if (O.index == hcoi) {
                            // release dD/dMV from memory for all moving variables MV
                            dOs_dMVs[D.index] = derivative();
                        }
                    }
                }
            }
            return std::move(dOs_dMVs[output_node.index]);
        }
        }
        URC;
    }

    std::vector<bool> all_dependency_operations(node top_node) const {
        std::vector<bool> result(operations.size(), false);
        all_dependency_operations(top_node, result);
        return std::move(result);
    }

    void all_dependency_operations(node top_node, std::vector<bool>& output) const {
        switch (top_node.type) {
        case node::nt_variable:
            return;
        case node::nt_operation:
            if (output[top_node.index])
                return;
            else {
                output[top_node.index] = true;
                const operation_impl& op = operations[top_node.index];
                for (node dep : op.dependencies) {
                    all_dependency_operations(dep, output);
                }
            }
            return;
        }
    }

    std::vector<bool> all_consumer_operations(node bottom_node) const {
        std::vector<bool> result(operations.size(), false);
        all_consumer_operations(bottom_node, result);
        return std::move(result);
    }

    void all_consumer_operations(node bottom_node, std::vector<bool>& output) const {
        switch (bottom_node.type) {
        case node::nt_variable: {
            const variable_impl& v = variables[bottom_node.index];
            for (operation con : v.consumers) {
                all_consumer_operations(con, output);
            }
            return;
        }
        case node::nt_operation: {
            if (output[bottom_node.index])
                return;
            else {
                output[bottom_node.index] = true;
                const operation_impl& op = operations[bottom_node.index];
                for (operation con : op.consumers) {
                    all_consumer_operations(con, output);
                }
            }
            return;
        }
        }
    }

    tensor_cptr_vec create_variable_values(const graph_input_map& input_value_map) const override {
        tensor_cptr_vec result(variables.size());
        for (auto i_map : input_value_map) {
            assert(i_map.first.type == node::nt_variable, "graph_input_map must have variables only.");
            assert(i_map.first.index >= 0 && static_cast<std::size_t>(i_map.first.index) < result.size(),
                    "input_value_map has invalid variable index, expected [0, ", result.size(), "), found ",
                    i_map.first.index);
            result[i_map.first.index] = i_map.second;
        }
        return std::move(result);
    }

    graph_impl(const std::vector<variable_impl>& _variables, const std::vector<operation_impl>& _operations) :
                    variables(_variables),
                    operations(_operations) {
    }

    std::string get_variable_name(variable v) const override {
        assert(v.index >= 0 && v.index < static_cast<int>(variables.size()),
                "Cannot get name from variable with index ", v.index);
        return variables[v.index].name;
    }

    std::string get_operation_name(operation o) const override {
        assert(o.index >= 0 && o.index < static_cast<int>(operations.size()),
                "Cannot get name from operation with index ", o.index);
        return operations[o.index].name;
    }

    variable get_variable(std::string const& name) const override {
        for (const auto& vimple : variables)
            if (vimple.name == name)
                return variable(vimple.index);
        throw std::runtime_error("Could not find variable with name " + name);
    }

    operation get_operation(std::string const& name) const override {
        for (const auto& oimpl : operations)
            if (oimpl.name == name)
                return operation(oimpl.index);
        throw std::runtime_error("Could not find operation with name " + name);
    }
};

struct graph_builder_impl: graph_builder {
    std::vector<variable_impl> variables;
    std::vector<operation_impl> operations;

    variable add_variable(const std::string& name) override {
        variable_impl vimpl { name, static_cast<int>(variables.size()), std::vector<operation>(), -1 };
        variables.push_back(vimpl);
        return variable(vimpl.index);
    }

    operation add_operation(const std::string& name, const tensor_function_csptr& function,
            const std::vector<node>& dependencies) override {
        operation_impl oimpl { name, static_cast<int>(operations.size()), function, std::vector<operation>(),
                dependencies, -1 };
        operation o(oimpl.index);
        for (node dep : dependencies) {
            switch (dep.type) {
            case node::nt_variable:
                variables[dep.index].consumers.push_back(o);
                variables[dep.index].highest_consumer_operation_index = oimpl.index;
                break;
            case node::nt_operation:
                operations[dep.index].consumers.push_back(o);
                operations[dep.index].highest_consumer_operation_index = oimpl.index;
                break;
            }
        }
        operations.push_back(oimpl);
        return o;
    }

    graph_cuptr build_graph() override {
        return graph_cuptr(new graph_impl { variables, operations });
    }
};

} // end anonymous namespace

namespace para {
namespace graph {

node::node(node_type _type, int _index) :
                type(_type),
                index(_index) {
}

variable::variable(int index) :
                node(node::nt_variable, index) {
}

operation::operation(int index) :
                node(node::nt_operation, index) {
}

tensor_function::~tensor_function() {
}

graph::~graph() {
}

graph_builder::~graph_builder() {
}

graph_builder_uptr graph_builder::empty() {
    return graph_builder_uptr(new graph_builder_impl);
}

} // end namespace graph
} // end namespace para

