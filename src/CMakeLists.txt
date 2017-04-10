cmake_minimum_required(VERSION 2.8)
project(GIMC)

#########################################################
# FIND FREEIMAGE
#########################################################
find_path(FREEIMAGE_INCLUDE FreeImage.h)
find_library(FREEIMAGE_LIB freeimage REQUIRED)
include_directories(${FREEIMAGE_INCLUDE})

find_package(OpenCL REQUIRED)

set(COMMON_SRC clutil.c)
add_library(COMMON_LIB ${COMMON_SRC})

set(BASE_SRC base.c)
add_executable(Base ${BASE_SRC})
target_link_libraries(Base ${COMMON_LIB} ${FREEIMAGE_LIB})
