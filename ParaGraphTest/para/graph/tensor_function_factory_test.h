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

struct tensor_function_factory_softmax_test: unit_test {
    std::string name() const override;
    void run() const override;
};

}
// end namespace graph
}// end namespace para

#endif /* PARA_GRAPH_TENSOR_FUNCTION_FACTORY_TEST_H_ */
