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

/**
 * Structure to represent a node in the graph.
 * A node can be either a variable or an operation.
 * The type was designed to be light and easy to copy.
 * The type can only be constructed as one of its derived types:
 *   variable and operation.
 * Nodes within a single graph are ordered.
 */
struct node {
	/** Enum representing the node type. */
	enum node_type {
		/** Enum value representing a variable node. */
		nt_variable,
		/** Enum value representing an operation node. */
		nt_operation
	};
	/** The type of this node: whether it is a variable or an operation. */
	node_type type;
	/** The index of this node in its overall graph.  */
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

/** A variable node. */
struct variable: node {
	variable(int index);
};
/** An operation node. **/
struct operation: node {
	operation(int index);
};

/**
 * A type to represent the value and gradient
 *   of an invocation of a tensor function.
 */
struct derivative {
	/** The resulting value from the invocation. */
	tensor_cptr node_value;
	/** The gradients of the invocation, with respect to each input, arranged in the order of the inputs. **/
	tensor_cptr_vec node_derivative;
};

/**
 * An abstract type
 *   representing a function from a vector of tensors to a single tensor.
 */
struct tensor_function {
	/** Function to compute the value of the function on a set of inputs. */
	virtual tensor_cptr value(const tensor_cptr_vec& tv) const = 0;
	/** Function to compute the value and gradients of the function on a set of inputs. */
	virtual derivative deriv(const tensor_cptr_vec& tv) const = 0;
	virtual ~tensor_function();
};
typedef std::shared_ptr<const tensor_function> tensor_function_csptr;

/**
 * A type representing the values of the input variables of a graph.
 */
typedef std::map<variable, tensor_cptr> graph_input_map;

/**
 * An immutable dependency graph describing how tensor_functions depend
 *   on input variables and other tensor_functions.
 * The graph contains the tensor_functions, but does not contain any input tensors.
 * Instead, the inputs are abstracted to place holders called "variables".
 * Thus, the graph describes dependencies among operations and variables.
 * The value/gradient of a particular node of the graph can be computed for
 *   a given set of input tensors.
 * The graph CANNOT describe circular dependencies.
 * Thus, the graph is a tree, with variables forming the leaves.
 * A graph can only be created using a graph_builder.
 */
struct graph {
	/**
	 * Compute the value of a node using a vector of input values.
	 * The create_variable_values method may be used to set-up the input vector.
	 */
	virtual tensor_cptr value(node output_node,
			const tensor_cptr_vec& input_values) const = 0;
	/**
	 * Compute the value and gradients of a node using a vector of input values.
	 * The create_variable_values method may be used to set-up the input vector.
	 */
	virtual derivative partial_gradient(node output_node,
			const std::vector<variable>& moving_variables,
			const tensor_cptr_vec& input_values) const = 0;

	/**
	 * A utility function that
	 *   takes a convenient map from variables to their tensor values,
	 *   and returns a vector which can be used to perform computations efficiently.
	 */
	virtual tensor_cptr_vec create_variable_values(
			const graph_input_map& input_value_map) const = 0;

	/** Function to retrieve the name of a variable. */
	virtual std::string get_variable_name(variable v) const = 0;
	/** Function to retrieve the name of an operation. */
	virtual std::string get_operation_name(operation o) const = 0;
	/**
	 * Function to retrieve a variable from its name.
	 * The name must be unique in a graph, or else the returned variable is unpredictable.
	 */
	virtual variable get_variable(const std::string& name) const = 0;
	/**
	 * Function to retrieve an operation from its name.
	 * The name must be unique in a graph, or else the returned operation is unpredictable.
	 */
	virtual operation get_operation(const std::string& name) const = 0;

	virtual ~graph();
};
typedef std::unique_ptr<const graph> graph_cuptr;

/**
 * A mutable structure for describing how to create a graph.
 * An empty graph_builder is to be created using the empty() static function.
 * The graph is then to be described bottom-up,
 *   starting from the leaves (variables),
 *   then describing operations on these variables and other other previously described operations.
 * Variables can be defined at any point,
 *   but the API forces you to define an operation only after
 *   all its dependencies have been defined.
 * Finally, the build_graph() method creates a graph and returns it.
 * The resulting variable and operation objects may be used
 *   to perform operations on the resultant graph.
 */
struct graph_builder {
	typedef std::unique_ptr<graph_builder> graph_builder_uptr;

	/**
	 * A function to add a variable in the graph that is to be built.
	 * The resulting "variable" object may be used to perform computations
	 *   on the graph that is built.
	 * The name of the variable is mostly for debugging purposes,
	 *   and need not be unique,
	 *   unless you wish to re-retrieve the variable object later again using its name.
	 */
	virtual variable add_variable(const std::string& name) = 0;

	/**
	 * A function to add an operation in the grpah that is to be built.
	 * The resulting "operation" object may be used to perform computations
	 *   on the graph that is built.
	 * The name of the operation is mostly for debugging purposes,
	 *   and need not be unique,
	 *   unless you wish to re-retrieve the operation object later again using its name.
	 */
	virtual operation add_operation(const std::string& name,
			const tensor_function_csptr& function,
			const std::vector<node>& dependencies) = 0;

	/**
	 * Create the graph based on the dependencies that have been described.
	 */
	virtual graph_cuptr build_graph() const = 0;

	/** Create an empty graph_builder. */
	static graph_builder_uptr empty();

	virtual ~graph_builder();
};
typedef std::unique_ptr<graph_builder> graph_builder_uptr;

}
// end namspace graph
}// end namespace para

#endif /* PARA_GRAPH_GRAPH_H_ */
