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

set(GIMC_IMAGE_SRC image.c filter.c)
add_library(GimcImage SHARED ${GIMC_IMAGE_SRC})
#set_property(TARGET GimcImage PROPERTY POSITION_INDEPENDENT_CODE ON)

set(BASE_SRC base.c)
add_executable(Base ${BASE_SRC})
target_link_libraries(Base GimcImage Common ${FREEIMAGE_LIB} ${OpenCL_LIBRARIES})

set(NCONV_SRC nconv.c)
add_executable(Nconv ${NCONV_SRC})
target_link_libraries(Nconv GimcImage Common ${FREEIMAGE_LIB} ${OpenCL_LIBRARIES})

set(NCONV_LWF_SRC nconv_lwf.c)
add_executable(Nconv_lwf ${NCONV_LWF_SRC})
target_link_libraries(Nconv_lwf GimcImage Common ${FREEIMAGE_LIB} ${OpenCL_LIBRARIES})

set(NCONV_LWF_P_SRC nconv_lwf_partials.c)
add_executable(Nconv_lwf_p ${NCONV_LWF_P_SRC})
target_link_libraries(Nconv_lwf_p GimcImage Common ${FREEIMAGE_LIB} ${OpenCL_LIBRARIES})

set(NCONV_LWF_L_SRC nconv_lwf_local.c)
add_executable(Nconv_lwf_l ${NCONV_LWF_L_SRC})
target_link_libraries(Nconv_lwf_l GimcImage Common ${FREEIMAGE_LIB} ${OpenCL_LIBRARIES})
