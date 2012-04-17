#
# Find the packages required by this module
#

find_package(PythonLibs REQUIRED)
find_package(NUMARRAY REQUIRED)

if(NOT PYTHON_NUMARRAY_FOUND)
    message(WARNING "Numpy not found. Please set PYTHON_NUMARRAY_INCLUDE_DIR.")
endif()

