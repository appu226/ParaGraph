project(libParaGraph)

# Source files
add_library(libParaGraph
	src/exception.cpp
	src/graph.cpp
	src/math.cpp
	src/ml_graph.cpp)

# Headers
target_include_directories(libParaGraph PUBLIC
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	$<INSTALL_INTERFACE:include>
	PRIVATE src)

# Compiler requirements
target_compile_features(libParaGraph
	PUBLIC cxx_auto_type
	PRIVATE cxx_variadic_templates)


# 'make install' to the correct locations (provided by GNUInstallDirs).
install(TARGETS libParaGraph EXPORT libParaGraphConfig
	ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
	LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
	RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})  # This is for Windows
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# This makes the project importable from the install directory
# Put config file in per-project dir (name MUST match), can also
# just go into 'cmake'.
install(EXPORT libParaGraphConfig DESTINATION share/libParaGraph/cmake)



# This makes the project importable from the build directory
export(TARGETS libParaGraph FILE libParaGraphConfig.cmake)


