/*
 * unit_test.h
 *
 *  Created on: 26-Dec-2017
 *      Author: parakram
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
