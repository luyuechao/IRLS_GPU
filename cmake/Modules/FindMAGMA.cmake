# - Find the MAGMA library
#
# Usage:
#   find_package(MAGMA [REQUIRED] [QUIET] )
#
# It sets the following variables:
#   MAGMA_FOUND               ... true if magma is found on the system
#   MAGMA_LIBRARY_DIRS        ... full path to magma library
#   MAGMA_INCLUDE_DIRS        ... magma include directory
#   MAGMA_LIBRARIES           ... magma libraries
#
# The following variables will be checked by the function
#   MAGMA_USE_STATIC_LIBS     ... if true, only static libraries are found
#   MAGMADIR                ... if set, the libraries are exclusively searched
#                                 under this path

#If environment variable MAGMADIR is specified, it has same effect as MAGMADIR
if(MAGMADIR)
    # set library directories
    set(MAGMA_LIBRARY_DIRS ${MAGMADIR}/lib)
    # set include directories
    set(MAGMA_INCLUDE_DIRS ${MAGMADIR}/include)
    # set libraries
    find_library(
        MAGMA_LIBRARIES
        NAMES "magma"
        PATHS ${MAGMADIR}
        PATH_SUFFIXES "lib"
        NO_DEFAULT_PATH
    )
    set(MAGMA_FOUND TRUE)
    message(STATUS "MAGMA found: " "INC:${MAGMA_INCLUDE_DIRS} " "MAGMA_LIBRARY_DIRS ${MAGMA_LIBRARY_DIRS}\n")
else()
    set(MAGMA_FOUND FALSE)
    message(STATUS "MAGMA NOT found! Please reset MAGMADIR in CMakeList.txt.\n")
endif()

