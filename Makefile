all:	build/lib/libParaGraph.so build/bin/ParaGraphTest

clean: clean_ParaGraph clean_ParaGraphTest

CPPFLAGS = -Wall -std=c++11 -fpic -c

build/lib:
	mkdir -p build/lib

build/bin:
	mkdir -p build/bin

ParaGraphObjects = $(patsubst ParaGraph/para/graph/%.cpp,build/obj/ParaGraph/%.o,$(wildcard ParaGraph/para/graph/*.cpp))

build/lib/libParaGraph.so:	$(ParaGraphObjects) build/lib
	g++ -shared -o $@ $(ParaGraphObjects)

build/obj/ParaGraph/%.o:	ParaGraph/para/graph/%.cpp build/obj/ParaGraph
	g++ $(CPPFLAGS) $< -o $@

build/obj/ParaGraph:
	mkdir -p build/obj/ParaGraph

clean_ParaGraph:	clean_libParaGraph clean_ParaGraphObjects

clean_libParaGraph: 
	rm -f build/lib/libParaGraph.so

clean_ParaGraphObjects:
	rm -rf build/obj/ParaGraph

ParaGraphTestObjects = $(patsubst ParaGraphTest/para/graph/%.cpp,build/obj/ParaGraphTest/%.o,$(wildcard ParaGraphTest/para/graph/*.cpp))

build/bin/ParaGraphTest:	$(ParaGraphTestObjects) build/bin build/lib/libParaGraph.so
	g++ -o $@ -Lbuild/lib -Wl,-rpath=build/lib $(ParaGraphTestObjects) -lParaGraph

build/obj/ParaGraphTest/%.o:	ParaGraphTest/para/graph/%.cpp build/obj/ParaGraphTest
	g++ $(CPPFLAGS) $< -o $@ -IParaGraph

build/obj/ParaGraphTest:
	mkdir -p build/obj/ParaGraphTest

clean_ParaGraphTest:	clean_binParaGraphTest clean_ParaGraphTestObjects

clean_binParaGraphTest:
	rm -f build/bin/ParaGraphTest

clean_ParaGraphTestObjects:
	rm -rf build/obj/ParaGraphTest
