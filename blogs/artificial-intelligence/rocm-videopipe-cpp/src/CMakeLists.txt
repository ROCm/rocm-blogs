cmake_minimum_required(VERSION 3.10)

# ROCM Path
if(DEFINED ENV{ROCM_PATH})
  set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Default ROCm installation path")
elseif(ROCM_PATH)
  message("-- INFO:ROCM_PATH Set -- ${ROCM_PATH}")
else()
  set(ROCM_PATH /opt/rocm CACHE PATH "Default ROCm installation path")
endif()
message("-- INFO:ROCM_PATH Effective value -- ${ROCM_PATH}")
# Set AMD Clang as default compiler
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS ON)
if (NOT DEFINED CMAKE_CXX_COMPILER)
  set(CMAKE_C_COMPILER ${ROCM_PATH}/lib/llvm/bin/clang)
  set(CMAKE_CXX_COMPILER ${ROCM_PATH}/lib/llvm/bin/clang++)
endif()

project(videopipe LANGUAGES CXX)

list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/lib/cmake ${ROCM_PATH}/include/rocdecode ${ROCM_PATH}/share/rocmcmakebuildtools/cmake)

find_package(HIP QUIET)
find_package(rocDecode QUIET)
find_package(migraphx QUIET)

if (NOT HIP_FOUND)
    message(FATAL_ERROR "-- ERROR!: HIP Not Found! - please install ROCm and HIP!")
endif()
if (rocdecode_FOUND) #7.0.0
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${rocdecode_LIBRARY})
    include_directories(${rocdecode_INCLUDE_DIR})
else()
    message(FATAL_ERROR "-- ERROR!: rocdecode Not Found! - please install rocdecode!")
endif()

if (NOT migraphx_FOUND)
    message(FATAL_ERROR "-- ERROR!: MIGraphX Not Found! - please install MIGraphX!")
endif()

# HIP
set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} hip::device)
# set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} hip::host)

# MIGraphX
set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} migraphx::c)

# rocdecode and utils
include_directories (${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/../../utils ${CMAKE_CURRENT_SOURCE_DIR}/../../utils/rocvideodecode ${CMAKE_CURRENT_SOURCE_DIR}/../../utils/ffmpegvideodecode)

file(GLOB SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME} ${LINK_LIBRARY_LIST})
