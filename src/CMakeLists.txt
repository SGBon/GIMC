cmake_minimum_required(VERSION 2.8)
project(GIMC)

#########################################################
# FIND FREEIMAGE
#########################################################
find_path(FREEIMAGE_INCLUDE FreeImage.h)
find_library(FREEIMAGE_LIB freeimage REQUIRED)
include_directories(${FREEIMAGE_INCLUDE})

# find opencl
find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})

set(COMMON_SRC SHARED clutil.c)
add_library(Common ${COMMON_SRC})

set(GIMC_IMAGE_SRC image.c)
add_library(GimcImage SHARED ${GIMC_IMAGE_SRC})

set(BASE_SRC base.c)
add_executable(Base ${BASE_SRC})
target_link_libraries(Base GimcImage Common ${FREEIMAGE_LIB})
