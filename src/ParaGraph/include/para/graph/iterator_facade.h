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


#pragma once
#include <cstddef>
#include <iterator>
#include "exception.h"


template<typename TParent, typename TElement>
class random_access_iterator_facade {
public:
	typedef TElement value_type;
	typedef int difference_type;
	typedef value_type * pointer;
	typedef value_type & reference;
	typedef std::random_access_iterator_tag iterator_category;
	typedef random_access_iterator_facade<TParent, TElement> ThisType;
	random_access_iterator_facade(TParent & parent, std::size_t const offset):
		m_parent(parent),
		m_offset(offset)
    { }

	reference operator * () const {
		return m_parent[m_offset];
	}
	random_access_iterator_facade operator + (std::size_t const offset_offset) const {
		return random_access_iterator_facade(m_parent, m_offset + offset_offset);
	}
	random_access_iterator_facade operator - (std::size_t const offset_offset) const {
		return random_access_iterator_facade(m_parent, m_offset - offset_offset);
	}
	random_access_iterator_facade & operator += (std::size_t const offset_offset) {
		m_offset += offset_offset;
		return *this;
	}
	random_access_iterator_facade & operator -= (std::size_t const offset_offset) {
		m_offset -= offset_offset;
		return *this;
	}
	random_access_iterator_facade & operator ++ () {
		++m_offset;
		return *this;
	}
	random_access_iterator_facade & operator -- () {
		--m_offset;
		return *this;
	}

	difference_type operator -(ThisType const & that) {
		return static_cast<difference_type>(m_offset) - static_cast<difference_type>(that.m_offset);
	}

	bool operator < (ThisType const & that) const {
		return this->m_offset < that.m_offset;
	}
	bool operator ==(ThisType const & that) const {
		return this->m_offset == that.m_offset;
	}
	bool operator >(ThisType const & that) const {
		return this->m_offset > that.m_offset;
	}
	bool operator <=(ThisType const & that) const {
		return this->m_offset <= that.m_offset;
	}
	bool operator >=(ThisType const & that) const {
		return this->m_offset >= that.m_offset;
	}
	bool operator !=(ThisType const & that) const {
		return this->m_offset != that.m_offset;
	}


private:
	TParent & m_parent;
	std::size_t m_offset;

};
