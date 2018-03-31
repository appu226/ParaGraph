# Define an executable
add_executable(ParaGraphTest
	src/graph_test.cpp
	src/graph_test_utils.cpp
	src/main.cpp
	src/math_test.cpp
	src/ml_graph_builder_test.cpp
	src/tensor_function_factory_test.cpp
	src/unit_test.cpp)

# Define the libraries this project depends upon
target_link_libraries(ParaGraphTest
	libParaGraph)
