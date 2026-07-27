cmake_minimum_required(VERSION 3.20)
project(Graphyne VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Use Clang explicitly (matches your MSYS2 install)
set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)

# Export compile commands for VSCode IntelliSense
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build type default
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
add_compile_options(
  -Wall
  -Wextra
  -O2
  $<$<CONFIG:Debug>:-g>
  $<$<CONFIG:Debug>:-O0>
)

# Subdirectories (each owns its CMakeLists.txt)
add_subdirectory(core)
add_subdirectory(tests)
# add_subdirectory(api)   # uncomment when ready