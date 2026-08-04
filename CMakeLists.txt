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
  -O3
  -mavx2
  -mfma
  -mtune=native
  $<$<CONFIG:Debug>:-g>
  $<$<CONFIG:Debug>:-O0>
)

# Profile-guided optimisation. Off by default; enable one phase at a time.
#   1. cmake -S . -B build -G Ninja -DGRAPHYNE_PGO_GENERATE=ON && cmake --build build
#      ./build/start/graphyne bin50k
#      llvm-profdata merge -output=pgo/graphyne.profdata pgo/*.profraw
#   2. cmake -S . -B build -G Ninja -DGRAPHYNE_PGO_GENERATE=OFF -DGRAPHYNE_PGO_USE=ON
# Regenerate the profile after any change to the hot path.
option(GRAPHYNE_PGO_GENERATE "Instrument the build to collect a PGO profile" OFF)
option(GRAPHYNE_PGO_USE "Build using a previously collected PGO profile" OFF)
set(GRAPHYNE_PGO_DIR "${CMAKE_SOURCE_DIR}/pgo" CACHE PATH "Where .profraw files are written")
set(GRAPHYNE_PGO_PROFILE "${GRAPHYNE_PGO_DIR}/graphyne.profdata" CACHE FILEPATH "Merged profile to build against")

if(GRAPHYNE_PGO_GENERATE AND GRAPHYNE_PGO_USE)
  message(FATAL_ERROR "Enable only one of GRAPHYNE_PGO_GENERATE / GRAPHYNE_PGO_USE")
endif()

if(GRAPHYNE_PGO_GENERATE)
  add_compile_options(-fprofile-generate=${GRAPHYNE_PGO_DIR})
  add_link_options(-fprofile-generate=${GRAPHYNE_PGO_DIR})
elseif(GRAPHYNE_PGO_USE)
  if(NOT EXISTS ${GRAPHYNE_PGO_PROFILE})
    message(FATAL_ERROR "no profile at ${GRAPHYNE_PGO_PROFILE} -- run the GRAPHYNE_PGO_GENERATE phase first")
  endif()
  # -Wno-profile-instr-unprofiled: new functions legitimately have no profile data.
  # profile-instr-out-of-date is left on -- it is the signal that the profile is stale.
  add_compile_options(-fprofile-use=${GRAPHYNE_PGO_PROFILE} -Wno-profile-instr-unprofiled)
  add_link_options(-fprofile-use=${GRAPHYNE_PGO_PROFILE})
endif()

# Subdirectories (each owns its CMakeLists.txt)
add_subdirectory(core)
add_subdirectory(start)
add_subdirectory(tests)
# add_subdirectory(api)   # uncomment when ready