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


#include "math_test.h"
#include <para/graph/math.h>
#include <para/graph/exception.h>
#include <algorithm>
#include <random>

namespace para {
namespace graph {

std::string tensor_construction_test::name() const {
    return "tensor_construction_test";
}

void tensor_construction_test::run() const {
    const int n1 = 2, n2 = 3, n3 = 5;
    const int n = n1 * n2 * n3;
    std::vector<double> data(n);
    for (size_t i = 0; i < n; ++i)
        data[i] = i;

    tensor t(tensor::N_vector { n1, n2, n3 }, std::move(data));

    assert(t.data[19] == 19, "tensor::tensor should copy data correctly.");
    assert(t.dimensionalities == tensor::N_vector { n1, n2, n3 },
            "tensor::tensor should copy dimensionalities correctly");

    assert(t.compute_offset(tensor::N_vector { 1, 2, 3 }) == 28, "tensor::compute_offset should work correctly.");
    assert(t.compute_position(17) == tensor::N_vector { 1, 0, 2 }, "tensor::compute_position should work correctly.");

    assert(is_failing([]() {
        tensor(tensor::N_vector {n1}, std::vector<double>(n1 + 1));
    }), "tensor construction should fail if data is invalid.");

}

std::string tensor_zero_test::name() const {
    return "tensor_zero_test";
}

void tensor_zero_test::run() const {
    const int n1 = 2, n2 = 3;
    tensor zero = tensor::zero(tensor::N_vector { n1, n2 });
    assert(zero.dimensionalities == tensor::N_vector { n1, n2 },
            "tensor::zero must return tensor of correct dimensionality.");
    std::for_each(zero.data.begin(), zero.data.end(),
            [](double d) {assert_doubles_are_close(d, 0, 1e-16, "tensor::zero must return correct data.");});

}

std::string tensor_zero_derivative_test::name() const {
    return "tensor_zero_derivative_test";
}

void tensor_zero_derivative_test::run() const {
    const int fn1 = 2, fn2 = 3;
    const int vn1 = 5;
    tensor zero = tensor::zero_derivative(tensor::N_vector { fn1, fn2 }, tensor::N_vector { vn1 });
    assert(zero.dimensionalities == tensor::N_vector { vn1, fn1, fn2 },
            "tensor::zero_derivative must return vector of correct dimensionality.");
    std::for_each(zero.data.begin(), zero.data.end(),
            [](double d) {assert_doubles_are_close(d, 0, 1e-16, "tensor::zero_derivative must return correct data.");});
}

std::string tensor_scalar_test::name() const {
    return "tensor_scalar_test";
}

void tensor_scalar_test::run() const {
    const double v = 3.14159;
    tensor scalar(tensor::N_vector(), { v });
    assert(scalar.dimensionalities.size() == 0,
            "tensor constructor must return correct order for scalars (order 0 tensors).");
    assert_doubles_are_close(v, scalar.data[0], 1e-15,
            "tensor constructor must return correct data for scalars (order 0 tensors).");
}

std::string tensor_identity_derivative_test::name() const {
    return "tensor_identity_derivative_test";
}

void tensor_identity_derivative_test::run() const {
    tensor::N_vector dims { 2, 3 };
    tensor I = std::move(tensor::identity_derivative(dims));
    assert(I.dimensionalities == tensor::N_vector { 2, 3, 2, 3 },
            "tensor::identity_derivative should return correct dimensionalities.");
    std::size_t data_size = std::accumulate(dims.begin(), dims.end(), 1,
            [](std::size_t a, std::size_t b) {return a * b;});
    data_size *= data_size;
    assert(I.data.size() == data_size, "tensor::identity_derivative should return correct data size.");
    std::vector<double> data(data_size, 0);
    for (tensor::N x = 0; x < dims[0]; ++x)
        for (tensor::N y = 0; y < dims[1]; ++y)
            data[I.compute_offset(tensor::N_vector { x, y, x, y })] = 1;
    for (std::size_t i_data = 0; i_data < data.size(); ++i_data) {
        assert_doubles_are_close(data[i_data], I.data[i_data], 1e-15,
                "tensor::identity_derivative should return correct data");
    }
}

std::string tensor_chain_multiplication_test::name() const {
    return "tensor_chain_multiplication_test";
}
void tensor_chain_multiplication_test::run() const {
    tensor::N an1 = 7, an2 = 5, an3 = 2;
    tensor::N bn1 = 5, bn2 = 2, bn3 = 3, bn4 = 2;

    std::default_random_engine dre;
    std::uniform_real_distribution<double> urd(0, 1);

    std::vector<double> adata(an1 * an2 * an3);
    std::for_each(adata.begin(), adata.end(), [&](double& d) {d = urd(dre);});
    std::vector<double> bdata(bn1 * bn2 * bn3 * bn4);
    std::for_each(bdata.begin(), bdata.end(), [&](double& d) {d = urd(dre);});

    tensor a(tensor::N_vector { an1, an2, an3 }, adata);
    tensor b(tensor::N_vector { bn1, bn2, bn3, bn4 }, bdata);
    tensor c = tensor::chain_multiplication(a, b, 2);

    assert(c.dimensionalities == tensor::N_vector { an1, bn3, bn4 },
            "tensor::chain_multiplication must return correct dimensionalities.");

    std::vector<double> result(an1 * bn3 * bn4);
    for (std::size_t i1 = 0; i1 < an1; ++i1)
        for (std::size_t i4 = 0; i4 < bn3; ++i4)
            for (std::size_t i5 = 0; i5 < bn4; ++i5) {
                double cd_expected = 0;
                for (std::size_t i2 = 0; i2 < an2; ++i2)
                    for (std::size_t i3 = 0; i3 < an3; ++i3) {
                        double ad = a.data[a.compute_offset(tensor::N_vector { i1, i2, i3 })];
                        double bd = b.data[b.compute_offset(tensor::N_vector { i2, i3, i4, i5 })];
                        cd_expected += ad * bd;
                    }
                double cd = c.data[c.compute_offset(tensor::N_vector { i1, i4, i5 })];
                assert_doubles_are_close(cd, cd_expected, 1e-15,
                        "tensor::chain_multiplication must return correct data.");
            }

}

std::string tensor_add_test::name() const {
    return "tensor_add_test";
}
void tensor_add_test::run() const {
    const tensor::N n1 = 2, n2 = 3;
    tensor::N_vector dims { n1, n2 };

    std::default_random_engine dre;
    std::uniform_real_distribution<double> urd(0, 1);

    std::vector<double> d1(n1 * n2), d2(n1 * n2);
    std::for_each(d1.begin(), d1.end(), [&](double& d) {d = urd(dre);});
    std::for_each(d2.begin(), d2.end(), [&](double& d) {d = urd(dre);});

    tensor t1(dims, std::move(d1)), t2(dims, std::move(d2));
    tensor ta = tensor::add(t1, t2);

    assert(ta.dimensionalities == dims, "tensor::add must return correct dimensionalities");
    assert(ta.data.size() == n1 * n2, "tensor::add must return data of correct size.");
    for (std::size_t i = 0; i < n1 * n2; ++i) {
        assert_doubles_are_close(ta.data[i], t1.data[i] + t2.data[i], 1e-15, "tensor::add must return correct data.");
    }

}

} // end namespace graph
} // end namespace para

