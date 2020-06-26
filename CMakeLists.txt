cmake_minimum_required(VERSION 3.15)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(_TARGET_NAME cppitch)

project(${_TARGET_NAME})

find_package(alpaka REQUIRED)

alpaka_add_executable(${_TARGET_NAME} alpaka_strided_copy.cpp)
target_link_libraries(${_TARGET_NAME} PUBLIC alpaka)

