#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/matchers/catch_matchers_range_equals.hpp>
#include <vector>
#include <cmath>

#include "Tensor.hpp"

auto WithinTensorMargin(const std::vector<float>& expected, float tolerance = 0.001f) {
    return Catch::Matchers::UnorderedRangeEquals(expected, [tolerance](float lhs, float rhs) {
        return std::abs(lhs - rhs) < tolerance;
    });
}

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp> // Assumed for tensor matching
#include <vector>

TEST_CASE( "Matmul Testing", "[matmul] [matrix]" ) {

    SECTION("Identity matrix multiplication") {
        Tensor a(
            {
                1.0f, 0.0f, 
                0.0f, 1.0f
            }, 
            {2, 2}
        );
        Tensor b({2, 2});
        Tensor dst({2, 2});

        b.fill(1.0f); // b becomes: {1, 1, 1, 1}

        Tensor::matmul(a, b, dst);

        std::vector<float> expected = {1.0f, 1.0f, 1.0f, 1.0f};
        REQUIRE_THAT(dst, WithinTensorMargin(expected));
    }

    SECTION("Standard 2x2 multiplication with diverse values") {
        Tensor a(
            {
                1.0f, 2.0f,
                3.0f, 4.0f
            },
            {2, 2}
        );
        Tensor b(
            {
                5.0f, 6.0f,
                7.0f, 8.0f
            },
            {2, 2}
        );
        Tensor dst({2, 2});

        Tensor::matmul(a, b, dst);

        std::vector<float> expected = {19.0f, 22.0f, 43.0f, 50.0f};
        REQUIRE_THAT(dst, WithinTensorMargin(expected));
    }

    SECTION("Non-square rectangular matrices (2x3 * 3x2)") {
        Tensor a(
            {
                1.0f, 2.0f, 3.0f,
                4.0f, 5.0f, 6.0f
            },
            {2, 3}
        );
        Tensor b(
            {
                7.0f, 8.0f,
                9.0f, 1.0f,
                2.0f, 3.0f
            },
            {3, 2}
        );
        Tensor dst({2, 2}); // Resulting matrix must be 2x2

        Tensor::matmul(a, b, dst);

        std::vector<float> expected = {31.0f, 19.0f, 85.0f, 55.0f};
        REQUIRE_THAT(dst, WithinTensorMargin(expected));
    }

    SECTION("Handling negative numbers and zeros") {
        Tensor a(
            {
                0.0f, 2.0f,
                -3.0f, 1.0f
            },
            {2, 2}
        );
        Tensor b(
            {
                4.0f, -1.0f,
                2.0f, 0.0f
            },
            {2, 2}
        );
        Tensor dst({2, 2});

        Tensor::matmul(a, b, dst);

        std::vector<float> expected = {4.0f, 0.0f, -10.0f, 3.0f};
        REQUIRE_THAT(dst, WithinTensorMargin(expected));
    }

    SECTION("Dimension mismatch should throw exception") {
        Tensor a({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
        Tensor b({1.0f, 2.0f}, {1, 2}); // Inner dimensions don't match (2 != 1)
        Tensor dst({2, 2});

        // Verifies the code throws an error instead of crashing/segfaulting
        REQUIRE_THROWS(Tensor::matmul(a, b, dst)); 
    }
}


TEST_CASE("Matmul Benchmarking", "[.][matmul_benchmark]") {
    SECTION("Small Matrix (32 x 32)") {
        size_t N = 32;
        Tensor a({N, N});
        Tensor b({N, N});
        Tensor dst({N, N});
        a.fill(1.5f);
        b.fill(2.0f);

        BENCHMARK("32x32 matmul") {
            return Tensor::matmul(a, b, dst);
        };
    }

    SECTION("Medium Matrix (256 x 256)") {
        size_t N = 256;
        Tensor a({N, N});
        Tensor b({N, N});
        Tensor dst({N, N});
        a.fill(1.5f);
        b.fill(2.0f);

        BENCHMARK("256x256 matmul") {
            return Tensor::matmul(a, b, dst);
        };
    }

        SECTION("Large Matrix (512 x 512)") {
        size_t N = 512;
        Tensor a({N, N});
        Tensor b({N, N});
        Tensor dst({N, N});
        a.fill(1.5f);
        b.fill(2.0f);

        BENCHMARK("512x512 matmul") {
            return Tensor::matmul(a, b, dst);
        };
    }
}


