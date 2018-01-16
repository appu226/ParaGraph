/*
 * graph_test.h
 *
 *  Created on: 09-Jan-2018
 *      Author: parakram
 */

#ifndef PARA_GRAPH_TEST_H_
#define PARA_GRAPH_TEST_H_

#include "unit_test.h"

namespace para {
namespace graph {

struct graph_scalar_test: unit_test {
    std::string name() const override;
    void run() const override;
};

struct graph_tensor_test: unit_test {
    std::string name() const override;
    void run() const override;
};

} // end namespace graph
} // end namespace para

#endif /* PARA_GRAPH_TEST_H_ */
