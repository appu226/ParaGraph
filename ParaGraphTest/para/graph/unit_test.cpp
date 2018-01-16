/*
 * unit_test.cpp
 *
 *  Created on: 26-Dec-2017
 *      Author: parakram
 */

#include "unit_test.h"
#include <iostream>
#include <cmath>

namespace para {
namespace graph {

void run_unit_tests(const unit_test_collection& uts) {
    std::vector<std::string> passed;
    std::vector<std::string> failed;
    std::vector<std::string> unknown;
    for (auto uti = uts.begin(); uti != uts.end(); ++uti) {
        std::string name = (*uti)->name();
        try {
            std::cout << "Starting test " << name << std::endl;
            unknown.push_back(name);
            (*uti)->run();
            passed.push_back(name);
            unknown.pop_back();
            std::cout << "Finished test " << name << std::endl;
        } catch (std::exception& e) {
            std::cout << "FAILED   test " << name << " with error: " << e.what() << std::endl;
            failed.push_back(name);
            unknown.pop_back();
        } catch (...) {
            std::cout << "FAILED   test " << name << std::endl;
            failed.push_back(name);
            unknown.pop_back();
        }
    }

    for (auto pi = passed.begin(); pi != passed.end(); ++pi) {
        std::cout << "[ \u2713 ] passed  " << *pi << std::endl;
    }
    for (auto fi = failed.begin(); fi != failed.end(); ++fi) {
        std::cout << "[ \u2717 ] failed  " << *fi << std::endl;
    }
    for (auto ui = unknown.begin(); ui != unknown.end(); ++ui) {
        std::cout << "[ ? ] unknown " << *ui << std::endl;
    }

    if (unknown.size() > 0)
        std::cout << "(Unknown: " << unknown.size() << "), ";
    if (failed.size() > 0)
        std::cout << "(Failed: " << failed.size() << "), ";
    if (passed.size() > 0)
        std::cout << "(Passed: " << passed.size() << "), ";
    std::cout << "(Total: " << uts.size() << ")" << std::endl;

}

void assert_doubles_are_close(double lhs, double rhs, double relative_tolerance, const std::string& message) {
    double abs_tolerance = (std::abs(lhs) + std::abs(rhs)) * relative_tolerance;
    double abs_diff = std::abs(lhs - rhs);
    if (abs_diff > abs_tolerance)
        throw std::runtime_error(message);
}

} // end namespace graph
} // end namespace para

