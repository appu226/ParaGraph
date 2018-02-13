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
