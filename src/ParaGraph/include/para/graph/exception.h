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

/**
 * Base case for variadic template.
 * Does nothing.
 */
void stream_all(std::stringstream& ss);

/**
 * Variadic template to stream all arguments into an std::stringsream.
 */
template<typename t_first, typename ... t_remaining>
void stream_all(std::stringstream& ss, const t_first& first,
		const t_remaining& ... remaining) {
	ss << first;
	return stream_all(ss, remaining ...);
}

/**
 * Stream all arguments to an std::stringstream
 * and return the final concatenated string.
 */
template<typename t_first, typename ... t_remaining>
std::string concat(const t_first& first, const t_remaining& ... remaining) {
	std::stringstream ss;
	stream_all(ss, first, remaining ...);
	return ss.str();
}

/**
 * Assert a condition,
 * and if false, throw a runtime_error with the message objects.
 */
template<typename ... t_messages>
void assert(bool condition, const t_messages& ... messages) {
	if (!condition) {
		throw std::runtime_error(concat(messages ...));
	}
}

} // end namespace graph
} // end namespace para

#endif /* PARA_GRAPH_EXCEPTION_H_ */
