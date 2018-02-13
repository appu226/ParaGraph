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


#ifndef PARA_GRAPH_UNIT_TEST_H_
#define PARA_GRAPH_UNIT_TEST_H_

#include <vector>
#include <string>
#include <memory>

namespace para {
namespace graph {

struct unit_test {
    virtual std::string name() const = 0;
    virtual void run() const = 0;
    virtual ~unit_test() {
    }
};

typedef std::unique_ptr<unit_test> unit_test_uptr;
typedef std::vector<unit_test_uptr> unit_test_collection;

void run_unit_tests(const unit_test_collection& uts);

void assert_doubles_are_close(double lhs, double rhs, double relative_tolerance, const std::string& message);

template<typename t_func>
bool is_failing(t_func func) {
    try {
        func();
    } catch (...) {
        return true;
    }
    return false;
}

} // end namespace graph
} // end namespace para

#endif /* PARA_GRAPH_UNIT_TEST_H_ */
