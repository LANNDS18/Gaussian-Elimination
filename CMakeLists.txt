cmake_minimum_required(VERSION 3.21)
project(mpi_test_set C)
set(CMAKE_C_STANDARD 99)

find_package(MPI)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

find_package(OpenMP)
include_directories(/usr/local/include)

if(OpenMP_CXX_FOUND)
    target_link_libraries(mpi_test_set PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(base baseline.c)
add_executable(mpi mpi.c)
add_executable(omp omp.c)