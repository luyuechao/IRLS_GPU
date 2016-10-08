CMAKE_MINIMUM_REQUIRED(VERSION 2.8.0)

PROJECT(IRLS_CUDA)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

FIND_PACKAGE(CUDA REQUIRED)

## MAGMA
set(MAGMADIR /Users/yc-lu/work/magma-2.1.0)
FIND_PACKAGE(MAGMA REQUIRED)

if (MAGMA_FOUND)
    add_definitions(-DADD_)
    add_definitions(-DMAGMA_SETAFFINITY)
    add_definitions(-DMAGMA_WITH_MKL)
    include_directories(${MAGMA_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "MAGMA was not found, please install it first")
endif()

#OpenBLAS
FIND_PACKAGE(OpenBLAS REQUIRED)

##EIGEN3
#remove the following if you have FindEigen.cmake
  get_filename_component(EIGEN_ROOT "/usr/include/local/eigen3" PATH)
    if(PKG_CONFIG_FOUND)
        pkg_check_modules(PC_EIGEN eigen3)
    endif(PKG_CONFIG_FOUND)
    find_path(EIGEN_INCLUDE_DIRS Eigen/Core
    HINTS ${PC_EIGEN_INCLUDEDIR} ${PC_EIGEN_INCLUDE_DIRS} 
            "${EIGEN_ROOT}" "$ENV{EIGEN_ROOT}"
    PATHS "$ENV{PROGRAMFILES}/Eigen 3.2.9" "$ENV{PROGRAMW6432}/Eigen 3.2.9"
          "$ENV{PROGRAMFILES}/Eigen" "$ENV{PROGRAMW6432}/Eigen"   
          PATH_SUFFIXES eigen3 include/eigen3 include)
    find_package_handle_standard_args(eigen DEFAULT_MSG EIGEN_INCLUDE_DIRS)
    set(EIGEN_DEFINITIONS ${EIGEN_DEFINITIONS} -DEIGEN_USE_NEW_STDVECTOR
        -DEIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET)


include_directories(${MAGMADIR}/testing 
                    ${CUDA_INCLUDE_DIRS} 
                    /usr/local/cuda/samples/common/inc/
                    ${EIGEN_INCLUDE_DIRS}
                    )

FILE(GLOB SOURCES "src/*.cu" "src/*.cpp" "src/*.hpp")
 
CUDA_ADD_EXECUTABLE(IRLS_CUDA ${SOURCES})

## -Wno-unused-variable shall be removed before release
LIST(APPEND CMAKE_CXX_FLAGS "-std=c++11 -O3 -Wall -Wno-unused-variable")

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}  "-Xcompiler;-fPIC;-std=c++11--ftz=true;--prec-div=false;--prec-sqrt=false")

LIST(APPEND CUDA_NVCC_FLAGS --compiler-options -fno-strict-aliasing -lineinfo -Xptxas -dlcm=cg)
## change arch and code according to your GPU architecture
LIST(APPEND CUDA_NVCC_FLAGS  -gencode arch=compute_52,code=sm_52)

target_link_libraries(IRLS_CUDA ${CUDA_LIBRARIES}
                                ${MAGMA_LIBRARIES}
                                ${CUDA_cublas_LIBRARY}
                                ${OpenBLAS_LIB}
                                ${Eigen_LIBRARIES}
                                ${MAGMADIR}/testing/lin/sqpt01.o
                                )

