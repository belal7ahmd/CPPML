cmake_minimum_required(VERSION 3.15)

project(CPPML VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags 
if(MSVC)
    add_compile_options(/O2 /arch:AVX2)
else()
    add_compile_options(-O3 -march=native)
endif()


# Bundle files into lib
add_library(ml_core STATIC
    src/Tensor.cpp
    src/Layers.cpp
    src/Networks.cpp
    src/LossF.cpp
    src/DataHandler.cpp
)

# Tell compiler where header files are
target_include_directories(ml_core PUBLIC include)

# Make the run executable
add_executable(run_app src/main.cpp)
target_link_libraries(run_app PRIVATE ml_core)

# Fetch Catch2 from repo
Include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.8.1 # or a later release
)

FetchContent_MakeAvailable(Catch2)

# Make the test executable
add_executable(run_tests tests/test.cpp)
target_link_libraries(run_tests PRIVATE ml_core)
target_link_libraries(run_tests PRIVATE Catch2::Catch2WithMain)
