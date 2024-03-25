cmake_minimum_required( VERSION 2.8 )
project( orb_extractor )

set( CMAKE_BUILD_TYPE "Debug" )
set( CMAKE_CXX_FLAGS "-std=c++11" )
set( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin )
set( LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib )

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

#add_library( orb_extractor src/orb_extractor.cpp )

add_executable( orb_extractor main.cpp ORBExtractor.cpp )
target_link_libraries( orb_extractor ${OpenCV_LIBS} )