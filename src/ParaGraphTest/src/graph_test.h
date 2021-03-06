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
