/*
 * ml_graph.h
 *
 *  Created on: 17-Jan-2018
 *      Author: parakram
 */

#ifndef PARA_GRAPH_ML_GRAPH_H_
#define PARA_GRAPH_ML_GRAPH_H_

#include "graph.h"

namespace para {
namespace graph {

struct tensor_function_factory {
    static tensor_function_csptr add();
    static tensor_function_csptr chain_multiplication(int num_common_dims);
    static tensor_function_csptr sigmoid();
    static tensor_function_csptr reduce_sum(int axis);
    static tensor_function_csptr log();
    static tensor_function_csptr element_wise_multiplication();
    static tensor_function_csptr negative();
};

struct ml_graph_builder {
    typedef std::unique_ptr<ml_graph_builder> uptr;

    virtual variable add_variable(const std::string& name) = 0;
    virtual operation add_operation(const std::string& name, const tensor_function_csptr& function,
            const std::vector<node>& dependencies) = 0;
    virtual operation add(node lhs, node rhs) = 0;
    virtual operation chain_multiplication(node lhs, node rhs, int num_common_dims) = 0;
    virtual operation sigmoid(node n) = 0;
    virtual operation reduce_sum(node n, int axis) = 0;
    virtual operation log(node n) = 0;
    virtual operation element_wise_multiplication(node lhs, node rhs) = 0;
    virtual operation negative(node lhs) = 0;

    virtual graph_cuptr build_graph() const = 0;

    virtual ~ml_graph_builder();

    static uptr empty();
};
typedef ml_graph_builder::uptr ml_graph_builder_uptr;

} // graph
} // para

#endif /* PARA_GRAPH_ML_GRAPH_H_ */
