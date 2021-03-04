cmake_minimum_required(VERSION 3.17)
project(alphazero)

find_package(Torch REQUIRED PATHS libtorch)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(alphazero main.cc model.cc model.h)
target_link_libraries(alphazero "${TORCH_LIBRARIES}")
set_property(TARGET alphazero PROPERTY CXX_STANDARD 14)