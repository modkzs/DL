# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.6

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Applications/CLion.app/Contents/bin/cmake/bin/cmake

# The command to remove a file.
RM = /Applications/CLion.app/Contents/bin/cmake/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/yixuanhe/code/study/DL

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/yixuanhe/code/study/DL/cmake-build-debug

# Include any dependencies generated for this target.
include bin/Layer/CMakeFiles/LAYER.dir/depend.make

# Include the progress variables for this target.
include bin/Layer/CMakeFiles/LAYER.dir/progress.make

# Include the compile flags for this target's objects.
include bin/Layer/CMakeFiles/LAYER.dir/flags.make

bin/Layer/CMakeFiles/LAYER.dir/BasicLayer.cpp.o: bin/Layer/CMakeFiles/LAYER.dir/flags.make
bin/Layer/CMakeFiles/LAYER.dir/BasicLayer.cpp.o: ../src/Layer/BasicLayer.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/yixuanhe/code/study/DL/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object bin/Layer/CMakeFiles/LAYER.dir/BasicLayer.cpp.o"
	cd /Users/yixuanhe/code/study/DL/cmake-build-debug/bin/Layer && /Library/Developer/CommandLineTools/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/LAYER.dir/BasicLayer.cpp.o -c /Users/yixuanhe/code/study/DL/src/Layer/BasicLayer.cpp

bin/Layer/CMakeFiles/LAYER.dir/BasicLayer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/LAYER.dir/BasicLayer.cpp.i"
	cd /Users/yixuanhe/code/study/DL/cmake-build-debug/bin/Layer && /Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/yixuanhe/code/study/DL/src/Layer/BasicLayer.cpp > CMakeFiles/LAYER.dir/BasicLayer.cpp.i

bin/Layer/CMakeFiles/LAYER.dir/BasicLayer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/LAYER.dir/BasicLayer.cpp.s"
	cd /Users/yixuanhe/code/study/DL/cmake-build-debug/bin/Layer && /Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/yixuanhe/code/study/DL/src/Layer/BasicLayer.cpp -o CMakeFiles/LAYER.dir/BasicLayer.cpp.s

bin/Layer/CMakeFiles/LAYER.dir/BasicLayer.cpp.o.requires:

.PHONY : bin/Layer/CMakeFiles/LAYER.dir/BasicLayer.cpp.o.requires

bin/Layer/CMakeFiles/LAYER.dir/BasicLayer.cpp.o.provides: bin/Layer/CMakeFiles/LAYER.dir/BasicLayer.cpp.o.requires
	$(MAKE) -f bin/Layer/CMakeFiles/LAYER.dir/build.make bin/Layer/CMakeFiles/LAYER.dir/BasicLayer.cpp.o.provides.build
.PHONY : bin/Layer/CMakeFiles/LAYER.dir/BasicLayer.cpp.o.provides

bin/Layer/CMakeFiles/LAYER.dir/BasicLayer.cpp.o.provides.build: bin/Layer/CMakeFiles/LAYER.dir/BasicLayer.cpp.o


bin/Layer/CMakeFiles/LAYER.dir/CNNLayer.cpp.o: bin/Layer/CMakeFiles/LAYER.dir/flags.make
bin/Layer/CMakeFiles/LAYER.dir/CNNLayer.cpp.o: ../src/Layer/CNNLayer.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/yixuanhe/code/study/DL/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object bin/Layer/CMakeFiles/LAYER.dir/CNNLayer.cpp.o"
	cd /Users/yixuanhe/code/study/DL/cmake-build-debug/bin/Layer && /Library/Developer/CommandLineTools/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/LAYER.dir/CNNLayer.cpp.o -c /Users/yixuanhe/code/study/DL/src/Layer/CNNLayer.cpp

bin/Layer/CMakeFiles/LAYER.dir/CNNLayer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/LAYER.dir/CNNLayer.cpp.i"
	cd /Users/yixuanhe/code/study/DL/cmake-build-debug/bin/Layer && /Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/yixuanhe/code/study/DL/src/Layer/CNNLayer.cpp > CMakeFiles/LAYER.dir/CNNLayer.cpp.i

bin/Layer/CMakeFiles/LAYER.dir/CNNLayer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/LAYER.dir/CNNLayer.cpp.s"
	cd /Users/yixuanhe/code/study/DL/cmake-build-debug/bin/Layer && /Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/yixuanhe/code/study/DL/src/Layer/CNNLayer.cpp -o CMakeFiles/LAYER.dir/CNNLayer.cpp.s

bin/Layer/CMakeFiles/LAYER.dir/CNNLayer.cpp.o.requires:

.PHONY : bin/Layer/CMakeFiles/LAYER.dir/CNNLayer.cpp.o.requires

bin/Layer/CMakeFiles/LAYER.dir/CNNLayer.cpp.o.provides: bin/Layer/CMakeFiles/LAYER.dir/CNNLayer.cpp.o.requires
	$(MAKE) -f bin/Layer/CMakeFiles/LAYER.dir/build.make bin/Layer/CMakeFiles/LAYER.dir/CNNLayer.cpp.o.provides.build
.PHONY : bin/Layer/CMakeFiles/LAYER.dir/CNNLayer.cpp.o.provides

bin/Layer/CMakeFiles/LAYER.dir/CNNLayer.cpp.o.provides.build: bin/Layer/CMakeFiles/LAYER.dir/CNNLayer.cpp.o


bin/Layer/CMakeFiles/LAYER.dir/RNNLayer.cpp.o: bin/Layer/CMakeFiles/LAYER.dir/flags.make
bin/Layer/CMakeFiles/LAYER.dir/RNNLayer.cpp.o: ../src/Layer/RNNLayer.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/yixuanhe/code/study/DL/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object bin/Layer/CMakeFiles/LAYER.dir/RNNLayer.cpp.o"
	cd /Users/yixuanhe/code/study/DL/cmake-build-debug/bin/Layer && /Library/Developer/CommandLineTools/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/LAYER.dir/RNNLayer.cpp.o -c /Users/yixuanhe/code/study/DL/src/Layer/RNNLayer.cpp

bin/Layer/CMakeFiles/LAYER.dir/RNNLayer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/LAYER.dir/RNNLayer.cpp.i"
	cd /Users/yixuanhe/code/study/DL/cmake-build-debug/bin/Layer && /Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/yixuanhe/code/study/DL/src/Layer/RNNLayer.cpp > CMakeFiles/LAYER.dir/RNNLayer.cpp.i

bin/Layer/CMakeFiles/LAYER.dir/RNNLayer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/LAYER.dir/RNNLayer.cpp.s"
	cd /Users/yixuanhe/code/study/DL/cmake-build-debug/bin/Layer && /Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/yixuanhe/code/study/DL/src/Layer/RNNLayer.cpp -o CMakeFiles/LAYER.dir/RNNLayer.cpp.s

bin/Layer/CMakeFiles/LAYER.dir/RNNLayer.cpp.o.requires:

.PHONY : bin/Layer/CMakeFiles/LAYER.dir/RNNLayer.cpp.o.requires

bin/Layer/CMakeFiles/LAYER.dir/RNNLayer.cpp.o.provides: bin/Layer/CMakeFiles/LAYER.dir/RNNLayer.cpp.o.requires
	$(MAKE) -f bin/Layer/CMakeFiles/LAYER.dir/build.make bin/Layer/CMakeFiles/LAYER.dir/RNNLayer.cpp.o.provides.build
.PHONY : bin/Layer/CMakeFiles/LAYER.dir/RNNLayer.cpp.o.provides

bin/Layer/CMakeFiles/LAYER.dir/RNNLayer.cpp.o.provides.build: bin/Layer/CMakeFiles/LAYER.dir/RNNLayer.cpp.o


# Object files for target LAYER
LAYER_OBJECTS = \
"CMakeFiles/LAYER.dir/BasicLayer.cpp.o" \
"CMakeFiles/LAYER.dir/CNNLayer.cpp.o" \
"CMakeFiles/LAYER.dir/RNNLayer.cpp.o"

# External object files for target LAYER
LAYER_EXTERNAL_OBJECTS =

bin/Layer/libLAYER.a: bin/Layer/CMakeFiles/LAYER.dir/BasicLayer.cpp.o
bin/Layer/libLAYER.a: bin/Layer/CMakeFiles/LAYER.dir/CNNLayer.cpp.o
bin/Layer/libLAYER.a: bin/Layer/CMakeFiles/LAYER.dir/RNNLayer.cpp.o
bin/Layer/libLAYER.a: bin/Layer/CMakeFiles/LAYER.dir/build.make
bin/Layer/libLAYER.a: bin/Layer/CMakeFiles/LAYER.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/yixuanhe/code/study/DL/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX static library libLAYER.a"
	cd /Users/yixuanhe/code/study/DL/cmake-build-debug/bin/Layer && $(CMAKE_COMMAND) -P CMakeFiles/LAYER.dir/cmake_clean_target.cmake
	cd /Users/yixuanhe/code/study/DL/cmake-build-debug/bin/Layer && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/LAYER.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
bin/Layer/CMakeFiles/LAYER.dir/build: bin/Layer/libLAYER.a

.PHONY : bin/Layer/CMakeFiles/LAYER.dir/build

bin/Layer/CMakeFiles/LAYER.dir/requires: bin/Layer/CMakeFiles/LAYER.dir/BasicLayer.cpp.o.requires
bin/Layer/CMakeFiles/LAYER.dir/requires: bin/Layer/CMakeFiles/LAYER.dir/CNNLayer.cpp.o.requires
bin/Layer/CMakeFiles/LAYER.dir/requires: bin/Layer/CMakeFiles/LAYER.dir/RNNLayer.cpp.o.requires

.PHONY : bin/Layer/CMakeFiles/LAYER.dir/requires

bin/Layer/CMakeFiles/LAYER.dir/clean:
	cd /Users/yixuanhe/code/study/DL/cmake-build-debug/bin/Layer && $(CMAKE_COMMAND) -P CMakeFiles/LAYER.dir/cmake_clean.cmake
.PHONY : bin/Layer/CMakeFiles/LAYER.dir/clean

bin/Layer/CMakeFiles/LAYER.dir/depend:
	cd /Users/yixuanhe/code/study/DL/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/yixuanhe/code/study/DL /Users/yixuanhe/code/study/DL/src/Layer /Users/yixuanhe/code/study/DL/cmake-build-debug /Users/yixuanhe/code/study/DL/cmake-build-debug/bin/Layer /Users/yixuanhe/code/study/DL/cmake-build-debug/bin/Layer/CMakeFiles/LAYER.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : bin/Layer/CMakeFiles/LAYER.dir/depend

