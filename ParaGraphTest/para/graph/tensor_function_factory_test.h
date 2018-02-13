/*
 * tensor_function_factory_test.h
 *
 *  Created on: 28-Jan-2018
 *      Author: parakram
 */

#ifndef PARA_GRAPH_TENSOR_FUNCTION_FACTORY_TEST_H_
#define PARA_GRAPH_TENSOR_FUNCTION_FACTORY_TEST_H_

#include "unit_test.h"

namespace para {
namespace graph {

struct tensor_function_factory_add_test: unit_test {
    std::string name() const override;
    void run() const override;
};

struct tensor_function_factory_chain_multiplication_test: unit_test {
    std::string name() const override;
    void run() const override;
};

struct tensor_function_factory_sigmoid_test: unit_test {
    std::string name() const override;
    void run() const override;
};

struct tensor_function_factory_reduce_sum_test: unit_test {
    std::string name() const override;
    void run() const override;
};

struct tensor_function_factory_log_test: unit_test {
    std::string name() const override;
    void run() const override;
};

struct tensor_function_factory_element_wise_multiplication_test: unit_test {
    std::string name() const override;
    void run() const override;
};

struct tensor_function_factory_negative_test: unit_test {
    std::string name() const override;
    void run() const override;
};

} // end namespace graph
} // end namespace para

#endif /* PARA_GRAPH_TENSOR_FUNCTION_FACTORY_TEST_H_ */
