cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
project(ParaGraph VERSION 18.03.31 LANGUAGES CXX)

# Must use GNUInstallDirs to install libraries into correct
# locations on all platforms.
include(GNUInstallDirs)

add_subdirectory(ParaGraph)
add_subdirectory(ParaGraphTest)
