/*
 * math_test.h
 *
 *  Created on: 26-Dec-2017
 *      Author: parakram
 */

#ifndef PARA_GRAPH_MATH_TEST_H_
#define PARA_GRAPH_MATH_TEST_H_

#include "unit_test.h"

namespace para {
namespace graph {

struct tensor_construction_test: unit_test {
    std::string name() const override;
    void run() const override;
};

struct tensor_zero_test: unit_test {
    std::string name() const override;
    void run() const override;
};

struct tensor_zero_derivative_test: unit_test {
    std::string name() const override;
    void run() const override;
};

struct tensor_scalar_test: unit_test {
    std::string name() const override;
    void run() const override;
};

struct tensor_identity_derivative_test: unit_test {
    std::string name() const override;
    void run() const override;
};

struct tensor_chain_multiplication_test: unit_test {
    std::string name() const override;
    void run() const override;
};

struct tensor_add_test: unit_test {
    std::string name() const override;
    void run() const override;
};

} // end namespace graph
} // end namespace para

#endif /* PARA_GRAPH_MATH_TEST_H_ */
