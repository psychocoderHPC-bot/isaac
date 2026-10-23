cmake_minimum_required (VERSION 3.25.2)
project (example)

# Search in <PACKAGENAME>_ROOT:
#   https://cmake.org/cmake/help/latest/policy/CMP0144.html
if(POLICY CMP0144)
    cmake_policy(SET CMP0144 NEW)
endif()

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)

set(ISAAC_DIR "${CMAKE_CURRENT_LIST_DIR}/../lib")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${ISAAC_DIR})

# Select the alpaka3 backend. The value is translated to the matching alpaka
# options below; only one accelerator dependency is enabled at a time.
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
    set(ISAAC_BACKEND_DEFINITION ISAAC_BACKEND_CPU)
elseif(ISAAC_BACKEND STREQUAL "omp")
    set(alpaka_DEP_OMP ON CACHE BOOL "Enable the OpenMP dependency of alpaka" FORCE)
    set(alpaka_EXEC_CpuOmpBlocks ON CACHE BOOL "Enable the OpenMP blocks executor of alpaka" FORCE)
    set(ISAAC_BACKEND_DEFINITION ISAAC_BACKEND_CPU)
elseif(ISAAC_BACKEND STREQUAL "cuda")
    set(alpaka_DEP_CUDA ON CACHE BOOL "Enable the CUDA dependency of alpaka" FORCE)
    set(alpaka_EXEC_GpuCuda ON CACHE BOOL "Enable the CUDA executor of alpaka" FORCE)
    set(alpaka_CUDA_NvidiaGpu ON CACHE BOOL "Enable api::cuda and deviceKind::nvidiaGpu" FORCE)
    set(alpaka_RELOCATABLE_DEVICE_CODE ON CACHE STRING "Enable relocatable device code" FORCE)
    set(ISAAC_BACKEND_DEFINITION ISAAC_BACKEND_CUDA)
elseif(ISAAC_BACKEND STREQUAL "hip")
    set(alpaka_DEP_HIP ON CACHE BOOL "Enable the HIP dependency of alpaka" FORCE)
    set(alpaka_EXEC_GpuHip ON CACHE BOOL "Enable the HIP executor of alpaka" FORCE)
    set(alpaka_HIP_AmdGpu ON CACHE BOOL "Enable api::hip and deviceKind::amdGpu" FORCE)
    set(alpaka_RELOCATABLE_DEVICE_CODE ON CACHE STRING "Enable relocatable device code" FORCE)
    set(ISAAC_BACKEND_DEFINITION ISAAC_BACKEND_HIP)
else()
    message(FATAL_ERROR "Unknown ISAAC_BACKEND '${ISAAC_BACKEND}', expected cpu, omp, cuda, or hip")
endif()

# Consume alpaka3 from the pinned submodule.
add_subdirectory(
    "${CMAKE_CURRENT_LIST_DIR}/../thirdParty/alpaka"
    "${CMAKE_CURRENT_BINARY_DIR}/alpaka"
)

find_package(ISAAC 1.1.0 REQUIRED)

set(ISAAC_DEFINITIONS ${ISAAC_DEFINITIONS} "-DISAAC_MAX_FUNCTORS=3")
set(ISAAC_DEFINITIONS ${ISAAC_DEFINITIONS} "-DISAAC_FUNCTOR_POW_ENABLED=0")

add_definitions(${ISAAC_DEFINITIONS})
include_directories(${ISAAC_INCLUDE_DIRS})

set(ISAAC_STEREO "No" CACHE STRING "Using stereoscopy")
set_property(CACHE ISAAC_STEREO PROPERTY STRINGS No SideBySide Anaglyph)

if(${ISAAC_STEREO} STREQUAL "No")
	add_definitions(-DISAAC_STEREO=0)
endif()
if(${ISAAC_STEREO} STREQUAL "SideBySide")
	add_definitions(-DISAAC_STEREO=1)
endif()
if(${ISAAC_STEREO} STREQUAL "Anaglyph")
	add_definitions(-DISAAC_STEREO=2)
endif()


add_executable(isaac_example ${CMAKE_SOURCE_DIR}/example.cpp)
target_compile_definitions(isaac_example PRIVATE ${ISAAC_BACKEND_DEFINITION})
target_link_libraries(isaac_example PUBLIC ${ISAAC_LIBRARIES})
alpaka_finalize(isaac_example)
install(TARGETS isaac_example DESTINATION bin )
