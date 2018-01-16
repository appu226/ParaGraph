/*
 * graph_test_utils.h
 *
 *  Created on: 13-Jan-2018
 *      Author: parakram
 */

#ifndef PARA_GRAPH_GRAPH_TEST_UTILS_H_
#define PARA_GRAPH_GRAPH_TEST_UTILS_H_

#include <para/graph/math.h>
#include <random>

namespace para {
namespace graph {

void assert_tensors_are_close(const tensor& lhs, const tensor& rhs, double relative_tolerance,
        const std::string& message);

tensor_cptr generate_random_tensor(const tensor::N_vector& dims, std::default_random_engine& dre);

std::string print_tensor(const tensor& t, const std::string& name);

} // end namespace graph
} // end namespace para

#endif /* end ifndef PARA_GRAPH_GRAPH_TEST_UTILS_H_ */
