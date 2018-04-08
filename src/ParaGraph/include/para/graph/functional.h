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

#ifndef PARA_GRAPH_FUNCTIONAL_H_
#define PARA_GRAPH_FUNCTIONAL_H_

#include <map>
#include <vector>
#include <functional>

namespace para {
namespace graph {

/**
 * Immutable wrapper around an std::map with some utility functions.
 * t_key is the key type of the underlying map.
 * t_value is the value type of the underlying map.
 */
template<typename t_key, typename t_value>
struct f_map {
	typedef std::map<t_key, t_value> data_type;
	const data_type& data;

	/**
	 * Create a new std::map
	 *   with lam applied to all the values in the original map.
	 * t_value2 is the value type of the resulting map.
	 * t_lam is a functor type going from  (t_value const &) to t_value2.
	 */
	template<typename t_value2, typename t_lam>
	std::map<t_key, t_value2> map_values(t_lam lam) const {
		std::map<t_key, t_value2> res;
		for (auto kv : data) {
			res[kv.first] = lam(kv.second);
		}
		return res;
	}

	/**
	 * Return a vector of all the keys in the map.
	 */
	std::vector<t_key> keys() const {
		std::vector<t_key> res;
		res.reserve(data.size());
		for (auto kv : data)
			res.push_back(kv.first);
		return res;
	}

	/**
	 * Return a vector of all the values in the map.
	 */
	std::vector<t_value> values() const {
		std::vector<t_value> res;
		res.reserve(data.size());
		for (auto kv : data)
			res.push_back(kv.second);
		return res;
	}

	/**
	 * Create a new std::map with using lam as a convertor function.
	 * t_key2 is the key type of the new map.
	 * t_value2 is the value type of the new map.
	 * t_lam is a functor type
	 *   from (std::pair<t_key, t_value> const&) to std::pair<t_key2, t_value2>.
	 */
	template<typename t_key2, typename t_value2, typename t_lam>
	std::map<t_key2, t_value2> map(t_lam lam) {
		std::map<t_key2, t_value2> res;
		for (auto kv : data)
			res.insert(lam(kv));
		return res;
	}

	/**
	 * Constructor to wrap an std::map<t_key, t_value>.
	 */
	f_map(const data_type& v_data) :
			data(v_data) {
	}
};

/**
 * Immutable wrapper around std::vector with some utility functions.
 * t_data is the element_type of the underlying vector.
 */
template<typename t_data>
struct f_vector {
	const std::vector<t_data>& data;
	typedef t_data t_value;

	/**
	 * Create a new vector
	 *   by applying the lam functor over all elements
	 *   of the underlying vector.
	 * t_data2 is the type of the resulting vector.
	 * t_lam is a functor from (t_data const &) to t_data2
	 */
	template<typename t_data2, typename t_lam>
	std::vector<t_data2> map(t_lam lam) {
		std::vector<t_data2> res;
		res.reserve(data.size());
		for (auto v : data)
			res.push_back(lam(v));
		return res;
	}

	/**
	 * Create a new vector by "zipping" together this and another vector ("that")
	 *   by applying the binary functor "lam" to each element of this and that.
	 * t_result is the element type of the resulting vector.
	 * t_data2 is the element type of the second input vector of the "zip" operation.
	 * t_lam is a binary functor that goes from (t_data const &, t_data2 const&) to t_result
	 * The size of the resulting vector is the min of the sizes of the two inputs.
	 * Hence, if the sizes of the two input vectors mismatch,
	 *   the "trailing" extra elements at the end of the larger vector
	 *   are silently ignored.
	 */
	template<typename t_result, typename t_data2, typename t_lam>
	std::vector<t_result> zip(const std::vector<t_data2>& that, t_lam lam) {
		size_t sz = std::min(data.size(), that.size());
		std::vector<t_result> res;
		res.reserve(sz);
		for (size_t idx = 0; idx < sz; ++idx)
			res.push_back(lam(data[idx], that[idx]));
		return res;
	}

	/**
	 * Zip the elements of this vector and another container
	 *   into a single vector containing the pairs of elements.
	 * t_container2 is the type of the container
	 *   that is to be zipped with this vector.
	 * t_container2 must have begin() and end() iterator getters.
	 * The size of the resulting vector is the min of the two sizes
	 *   so if the sizes mismatch then
	 *   the trailing extra elements in the larger input
	 *   are silently ignored.
	 */
	template<typename t_container2>
	std::vector<std::pair<t_data, typename t_container2::value_type>> zip(
			const t_container2& container2) {
		std::vector<std::pair<t_data, typename t_container2::value_type>> result;
		const auto end = data.end();
		const auto end2 = container2.end();
		auto it = data.begin();
		auto it2 = container2.begin();
		for (; it != end && it2 != end2; ++it, ++it2)
			result.push_back(std::make_pair(*it, *it2));
		return result;
	}

	/**
	 * Create a map
	 *   with the elements of this vector as keys,
	 *   and elements of another container as values.
	 * t_container2 is the type of the container containing the values.
	 * t_container2 must have begin() and end() iterator getters.
	 * The key-value bindings are based on the position of the keys and values
	 *   in their respective containers.
	 * The size of the resulting map
	 *   is the min of the sizes of the two inputs containers,
	 *   so if the input containers mismatch in size,
	 *   then the trailing extra elements at the end of the larger input
	 *   are silently ignored.
	 */
	template<typename t_container2>
	std::map<t_data, typename t_container2::value_type> zipToMap(
			const t_container2& container2) {
		std::map<t_data, typename t_container2::value_type> result;
		const auto end = data.end();
		const auto end2 = container2.end();
		auto it = data.begin();
		auto it2 = container2.begin();
		for (; it != end && it2 != end2; ++it, ++it2)
			result.insert(std::make_pair(*it, *it2));
		return result;
	}

	/**
	 * Constructor to wrap a vector.
	 */
	f_vector(const std::vector<t_data>& v_data) :
			data(v_data) {
	}
};

/**
 * Utility function to wrap a map into an f_map.
 */
template<typename t_key, typename t_value>
f_map<t_key, t_value> functional(const std::map<t_key, t_value>& data) {
	return f_map<t_key, t_value>(data);
}

/**
 * Utility function to wrap a vector into an f_vector.
 */
template<typename t_value>
f_vector<t_value> functional(const std::vector<t_value>& data) {
	return f_vector<t_value>(data);
}

} // end namespace graph
} // end namespace para

#endif /* PARA_GRAPH_FUNCTIONAL_H_ */
