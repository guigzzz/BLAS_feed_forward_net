cmake_minimum_required(VERSION 2.8)

set(CWD ${CMAKE_CURRENT_SOURCE_DIR})

set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -DNDEBUG")
#set(CMAKE_BUILD_TYPE Release)
set(CMAKE_BUILD_TYPE Debug)

set(FANN_PATH ~/fann/)
set(BOOST_PATH ~/boost_1_61_0/)
set(BOOST_SIMD_PATH ~/boost.simd/)
set(BLAS_PATH /home/monster/OpenBLAS/)

include_directories(${FANN_PATH}/src/include)
link_directories(${FANN_PATH}/build/src/)
include_directories(${BOOST_PATH})
include_directories(${BOOST_SIMD_PATH}/include)

include_directories(${CWD}/nnlib/include)
add_library(nnlib ${CWD}/nnlib/nnlib_src.cpp)
target_link_libraries(nnlib openblas pthread)

add_executable(c2b convert_to_binary.cpp)
target_link_libraries(c2b -lfann)
add_executable(test_data_to_binary test_data_to_binary.cpp)
target_link_libraries(test_data_to_binary -lfann)
add_executable(run_net_blas run_net_blas.cpp)
target_link_libraries(run_net_blas nnlib)
add_executable(mult_data mult_data.cpp)
target_link_libraries(mult_data nnlib)
add_executable(binary_write binary_write.cpp)
target_link_libraries(binary_write -lfann)

