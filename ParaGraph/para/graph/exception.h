/*
 * exception.h
 *
 *  Created on: 25-Dec-2017
 *      Author: parakram
 */

#ifndef PARA_GRAPH_EXCEPTION_H_
#define PARA_GRAPH_EXCEPTION_H_

#include <string>
#include <sstream>
#include <stdexcept>

namespace para {
namespace graph {

void stream_all(std::stringstream& ss);

template<typename t_first, typename ... t_remaining>
void stream_all(std::stringstream& ss, const t_first& first, const t_remaining& ... remaining) {
    ss << first;
    return stream_all(ss, remaining ...);
}

template<typename ... t_messages>
void assert(bool condition, const t_messages& ... remaining) {
    if (!condition) {
        std::stringstream err_ss;
        stream_all(err_ss, remaining ...);
        throw std::runtime_error(err_ss.str());
    }
}

} // end namespace graph
} // end namespace para

#endif /* PARA_GRAPH_EXCEPTION_H_ */
