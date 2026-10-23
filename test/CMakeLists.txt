cmake_minimum_required (VERSION 3.25.2)
project (isaac_device_globals_test LANGUAGES CXX)

# Search in <PACKAGENAME>_ROOT:
#   https://cmake.org/cmake/help/latest/policy/CMP0144.html
if(POLICY CMP0144)
    cmake_policy(SET CMP0144 NEW)
endif()

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)

set(ISAAC_DIR "${CMAKE_CURRENT_LIST_DIR}/../lib")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${ISAAC_DIR})

# Select the alpaka3 backend. This mirrors example/CMakeLists.txt so the test
# target is compiled with exactly the same accelerator options as ISAAC itself.
set(ISAAC_BACKEND "cpu" CACHE STRING "Accelerator backend to compile for (cpu, omp, cuda, hip)")
set_property(CACHE ISAAC_BACKEND PROPERTY STRINGS "cpu" "omp" "cuda" "hip")

set(alpaka_DEP_CUDA OFF CACHE BOOL "Enable the CUDA dependency of alpaka" FORCE)
set(alpaka_DEP_HIP OFF CACHE BOOL "Enable the HIP dependency of alpaka" FORCE)
set(alpaka_DEP_OMP OFF CACHE BOOL "Enable the OpenMP dependency of alpaka" FORCE)
set(alpaka_EXEC_CpuSerial ON CACHE BOOL "Enable the serial CPU executor of alpaka" FORCE)
set(alpaka_EXEC_CpuOmpBlocks OFF CACHE BOOL "Enable the OpenMP blocks executor of alpaka" FORCE)
set(alpaka_EXEC_GpuCuda OFF CACHE BOOL "Enable the CUDA executor of alpaka" FORCE)
set(alpaka_EXEC_GpuHip OFF CACHE BOOL "Enable the HIP executor of alpaka" FORCE)
set(alpaka_RELOCATABLE_DEVICE_CODE OFF CACHE STRING "Enable relocatable device code" FORCE)

if(ISAAC_BACKEND STREQUAL "cpu")
    # serial CPU executor is enabled by default
elseif(ISAAC_BACKEND STREQUAL "omp")
    set(alpaka_DEP_OMP ON CACHE BOOL "Enable the OpenMP dependency of alpaka" FORCE)
    set(alpaka_EXEC_CpuOmpBlocks ON CACHE BOOL "Enable the OpenMP blocks executor of alpaka" FORCE)
elseif(ISAAC_BACKEND STREQUAL "cuda")
    set(alpaka_DEP_CUDA ON CACHE BOOL "Enable the CUDA dependency of alpaka" FORCE)
    set(alpaka_EXEC_GpuCuda ON CACHE BOOL "Enable the CUDA executor of alpaka" FORCE)
    set(alpaka_CUDA_NvidiaGpu ON CACHE BOOL "Enable api::cuda and deviceKind::nvidiaGpu" FORCE)
    set(alpaka_RELOCATABLE_DEVICE_CODE ON CACHE STRING "Enable relocatable device code" FORCE)
elseif(ISAAC_BACKEND STREQUAL "hip")
    set(alpaka_DEP_HIP ON CACHE BOOL "Enable the HIP dependency of alpaka" FORCE)
    set(alpaka_EXEC_GpuHip ON CACHE BOOL "Enable the HIP executor of alpaka" FORCE)
    set(alpaka_HIP_AmdGpu ON CACHE BOOL "Enable api::hip and deviceKind::amdGpu" FORCE)
    set(alpaka_RELOCATABLE_DEVICE_CODE ON CACHE STRING "Enable relocatable device code" FORCE)
else()
    message(FATAL_ERROR "Unknown ISAAC_BACKEND '${ISAAC_BACKEND}', expected cpu, omp, cuda, or hip")
endif()

# Consume the same pinned alpaka3 submodule that ISAAC uses.
add_subdirectory(
    "${CMAKE_CURRENT_LIST_DIR}/../thirdParty/alpaka"
    "${CMAKE_CURRENT_BINARY_DIR}/alpaka"
)

# Dependency discovery mirrors lib/ISAACConfig.cmake. The kernel headers under
# test only include alpaka, Boost and glm, so only those are linked into the
# test binary. jansson, IceT and MPI are discovered the same way ISAAC does it
# because isaac_helper.hpp (reached from isaac_common_kernel.hpp) needs
# jansson; the test itself only uses the device globals.
find_package(Boost 1.70.0 CONFIG QUIET)
if(NOT Boost_FOUND)
    message(FATAL_ERROR "Boost (>= 1.70) is required by the ISAAC kernel headers")
endif()

# ISAACConfig.cmake asks for glm 1.0.0. Prefer the same requirement here. The
# validation container ships glm 0.9.9.8, and the kernel headers only use core
# glm types and matrix math, so fall back to an unversioned package with a
# warning instead of failing configure on that container.
find_package(glm 1.0.0 QUIET PATH_SUFFIXES "lib64/cmake/glm/" "lib/cmake/glm/")
if(NOT glm_FOUND)
    message(WARNING "glm >= 1.0.0 not found; falling back to the installed glm package for this test build")
    find_package(glm REQUIRED PATH_SUFFIXES "lib64/cmake/glm/" "lib/cmake/glm/")
endif()

find_package(jansson CONFIG QUIET)
if(TARGET jansson::jansson)
    set(ISAAC_TEST_JANSSON_TARGET jansson::jansson)
else()
    # Distributions without the jansson CMake package still provide the header
    # and library; the ISAAC headers need both (isaac_helper.hpp calls json_*).
    find_path(ISAAC_TEST_JANSSON_INCLUDE_DIR jansson.h)
    find_library(ISAAC_TEST_JANSSON_LIBRARY jansson)
    if(NOT ISAAC_TEST_JANSSON_INCLUDE_DIR OR NOT ISAAC_TEST_JANSSON_LIBRARY)
        message(FATAL_ERROR "jansson is required by the ISAAC kernel headers")
    endif()
endif()
find_package(IceT CONFIG QUIET PATH_SUFFIXES "/lib/")
find_package(MPI MODULE QUIET)
find_package(Threads MODULE QUIET)

if(TARGET Boost::boost)
    set(ISAAC_TEST_BOOST_TARGET Boost::boost)
elseif(TARGET Boost::headers)
    set(ISAAC_TEST_BOOST_TARGET Boost::headers)
else()
    message(FATAL_ERROR "No usable Boost header target found")
endif()

add_executable(isaac_device_globals_test
    device_globals.cpp
    device_globals_second_tu.cpp
)

# The second TU includes only device_globals_second_tu.hpp, which pulls
# isaac_functor_chain.hpp. That header declares the device globals without
# reaching isaac_helper.hpp, whose six non-inline host helper definitions would
# otherwise be emitted by both TUs and fail the link for reasons unrelated to
# the device globals. isaac_common_kernel.hpp and isaac_ssao_kernel.hpp reach
# isaac_helper.hpp, so only the main TU includes them. No linker flag relaxes
# duplicate-symbol checking: a duplicate *device global* definition still fails
# the link and the cross-TU ODR check stays strict.

# The task requires the ISAAC include root; the test directory itself holds the
# shared declarations used by the second translation unit.
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../lib)
target_include_directories(isaac_device_globals_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# ISAAC's headers test BOOST_LANG_CUDA/BOOST_LANG_HIP but never include the
# Boost.Predef header that defines them. With alpaka's pedantic flags this is a
# -Wundef error, so provide the backend-correct values explicitly.
if(ISAAC_BACKEND STREQUAL "cuda")
    set(ISAAC_LANG_DEFINITIONS BOOST_LANG_CUDA=1 BOOST_LANG_HIP=0)
elseif(ISAAC_BACKEND STREQUAL "hip")
    set(ISAAC_LANG_DEFINITIONS BOOST_LANG_CUDA=0 BOOST_LANG_HIP=1)
else()
    set(ISAAC_LANG_DEFINITIONS BOOST_LANG_CUDA=0 BOOST_LANG_HIP=0)
endif()

target_compile_definitions(isaac_device_globals_test PRIVATE
    -DISAAC_MAX_FUNCTORS=3
    -DISAAC_FUNCTOR_POW_ENABLED=0
    -DISAAC_VALGRIND_TWEAKS=0
    ${ISAAC_LANG_DEFINITIONS}
)

if(NOT TARGET jansson::jansson)
    target_include_directories(isaac_device_globals_test PRIVATE ${ISAAC_TEST_JANSSON_INCLUDE_DIR})
    target_link_libraries(isaac_device_globals_test PRIVATE ${ISAAC_TEST_JANSSON_LIBRARY})
else()
    target_link_libraries(isaac_device_globals_test PRIVATE ${ISAAC_TEST_JANSSON_TARGET})
endif()

target_link_libraries(isaac_device_globals_test PRIVATE
    alpaka::alpaka
    glm::glm
    ${ISAAC_TEST_BOOST_TARGET}
)

alpaka_finalize(isaac_device_globals_test)

enable_testing()
# Full suite: initializer state, two init/read iterations on one non-blocking
# queue, and the in-kernel function-pointer chain used for the NV2 regression.
add_test(NAME device_globals COMMAND isaac_device_globals_test)
# Separate process so the device globals are still in their declared initializer
# state; the read kernel runs without any host initialisation copy.
add_test(NAME device_globals_no_init COMMAND isaac_device_globals_test --negative-control)
