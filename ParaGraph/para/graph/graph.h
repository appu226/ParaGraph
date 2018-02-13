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


#ifndef PARA_GRAPH_GRAPH_H_
#define PARA_GRAPH_GRAPH_H_

#include <string>
#include <functional>
#include "math.h"
#include <map>
#include <memory>
#include <vector>

namespace para {
namespace graph {

struct node {
    enum node_type {
        nt_variable, nt_operation
    };
    node_type type;
    int index;

    bool operator==(const node rhs) const {
        return type == rhs.type && index == rhs.index;
    }

    bool operator<(const node rhs) const {
        if (type < rhs.type)
            return true;
        else if (type > rhs.type)
            return false;
        else if (index < rhs.index)
            return true;
        else
            return false;
    }

protected:
    node(node_type type, int index);
};

struct variable: node {
    variable(int index);
};
struct operation: node {
    operation(int index);
};

struct derivative {
    tensor_cptr node_value;
    tensor_cptr_vec node_derivative;
};

struct tensor_function {
    virtual tensor_cptr value(const tensor_cptr_vec& tv) const = 0;
    virtual derivative deriv(const tensor_cptr_vec& tv) const = 0;
    virtual ~tensor_function();
};
typedef std::shared_ptr<const tensor_function> tensor_function_csptr;

typedef std::map<variable, tensor_cptr> graph_input_map;

struct graph {
    virtual tensor_cptr value(node output_node, const tensor_cptr_vec& input_values) const = 0;
    virtual derivative partial_gradient(node output_node, const std::vector<variable>& moving_variables,
            const tensor_cptr_vec& input_values) const = 0;

    virtual tensor_cptr_vec create_variable_values(const graph_input_map& input_value_map) const = 0;

    virtual ~graph();
};
typedef std::unique_ptr<const graph> graph_cuptr;

struct graph_builder {
    typedef std::unique_ptr<graph_builder> graph_builder_uptr;

    virtual variable add_variable(const std::string& name) = 0;
    virtual operation add_operation(const std::string& name, const tensor_function_csptr& function,
            const std::vector<node>& dependencies) = 0;

    virtual graph_cuptr build_graph() = 0;

    static graph_builder_uptr empty();

    virtual ~graph_builder();
};
typedef std::unique_ptr<graph_builder> graph_builder_uptr;

}
 // end namspace graph
}// end namespace para

#endif /* PARA_GRAPH_GRAPH_H_ */
