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
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.6.2/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.6.2/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/yixuanhe/code/study/DL

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/yixuanhe/code/study/DL/build

# Include any dependencies generated for this target.
include bin/CMakeFiles/DL.dir/depend.make

# Include the progress variables for this target.
include bin/CMakeFiles/DL.dir/progress.make

# Include the compile flags for this target's objects.
include bin/CMakeFiles/DL.dir/flags.make

bin/CMakeFiles/DL.dir/main.cpp.o: bin/CMakeFiles/DL.dir/flags.make
bin/CMakeFiles/DL.dir/main.cpp.o: ../src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/yixuanhe/code/study/DL/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object bin/CMakeFiles/DL.dir/main.cpp.o"
	cd /Users/yixuanhe/code/study/DL/build/bin && /Library/Developer/CommandLineTools/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/DL.dir/main.cpp.o -c /Users/yixuanhe/code/study/DL/src/main.cpp

bin/CMakeFiles/DL.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/DL.dir/main.cpp.i"
	cd /Users/yixuanhe/code/study/DL/build/bin && /Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/yixuanhe/code/study/DL/src/main.cpp > CMakeFiles/DL.dir/main.cpp.i

bin/CMakeFiles/DL.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/DL.dir/main.cpp.s"
	cd /Users/yixuanhe/code/study/DL/build/bin && /Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/yixuanhe/code/study/DL/src/main.cpp -o CMakeFiles/DL.dir/main.cpp.s

bin/CMakeFiles/DL.dir/main.cpp.o.requires:

.PHONY : bin/CMakeFiles/DL.dir/main.cpp.o.requires

bin/CMakeFiles/DL.dir/main.cpp.o.provides: bin/CMakeFiles/DL.dir/main.cpp.o.requires
	$(MAKE) -f bin/CMakeFiles/DL.dir/build.make bin/CMakeFiles/DL.dir/main.cpp.o.provides.build
.PHONY : bin/CMakeFiles/DL.dir/main.cpp.o.provides

bin/CMakeFiles/DL.dir/main.cpp.o.provides.build: bin/CMakeFiles/DL.dir/main.cpp.o


bin/CMakeFiles/DL.dir/Layer/BasicLayer.cpp.o: bin/CMakeFiles/DL.dir/flags.make
bin/CMakeFiles/DL.dir/Layer/BasicLayer.cpp.o: ../src/Layer/BasicLayer.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/yixuanhe/code/study/DL/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object bin/CMakeFiles/DL.dir/Layer/BasicLayer.cpp.o"
	cd /Users/yixuanhe/code/study/DL/build/bin && /Library/Developer/CommandLineTools/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/DL.dir/Layer/BasicLayer.cpp.o -c /Users/yixuanhe/code/study/DL/src/Layer/BasicLayer.cpp

bin/CMakeFiles/DL.dir/Layer/BasicLayer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/DL.dir/Layer/BasicLayer.cpp.i"
	cd /Users/yixuanhe/code/study/DL/build/bin && /Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/yixuanhe/code/study/DL/src/Layer/BasicLayer.cpp > CMakeFiles/DL.dir/Layer/BasicLayer.cpp.i

bin/CMakeFiles/DL.dir/Layer/BasicLayer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/DL.dir/Layer/BasicLayer.cpp.s"
	cd /Users/yixuanhe/code/study/DL/build/bin && /Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/yixuanhe/code/study/DL/src/Layer/BasicLayer.cpp -o CMakeFiles/DL.dir/Layer/BasicLayer.cpp.s

bin/CMakeFiles/DL.dir/Layer/BasicLayer.cpp.o.requires:

.PHONY : bin/CMakeFiles/DL.dir/Layer/BasicLayer.cpp.o.requires

bin/CMakeFiles/DL.dir/Layer/BasicLayer.cpp.o.provides: bin/CMakeFiles/DL.dir/Layer/BasicLayer.cpp.o.requires
	$(MAKE) -f bin/CMakeFiles/DL.dir/build.make bin/CMakeFiles/DL.dir/Layer/BasicLayer.cpp.o.provides.build
.PHONY : bin/CMakeFiles/DL.dir/Layer/BasicLayer.cpp.o.provides

bin/CMakeFiles/DL.dir/Layer/BasicLayer.cpp.o.provides.build: bin/CMakeFiles/DL.dir/Layer/BasicLayer.cpp.o


# Object files for target DL
DL_OBJECTS = \
"CMakeFiles/DL.dir/main.cpp.o" \
"CMakeFiles/DL.dir/Layer/BasicLayer.cpp.o"

# External object files for target DL
DL_EXTERNAL_OBJECTS =

bin/DL: bin/CMakeFiles/DL.dir/main.cpp.o
bin/DL: bin/CMakeFiles/DL.dir/Layer/BasicLayer.cpp.o
bin/DL: bin/CMakeFiles/DL.dir/build.make
bin/DL: bin/Layer/libLAYER.a
bin/DL: bin/CMakeFiles/DL.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/yixuanhe/code/study/DL/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable DL"
	cd /Users/yixuanhe/code/study/DL/build/bin && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/DL.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
bin/CMakeFiles/DL.dir/build: bin/DL

.PHONY : bin/CMakeFiles/DL.dir/build

bin/CMakeFiles/DL.dir/requires: bin/CMakeFiles/DL.dir/main.cpp.o.requires
bin/CMakeFiles/DL.dir/requires: bin/CMakeFiles/DL.dir/Layer/BasicLayer.cpp.o.requires

.PHONY : bin/CMakeFiles/DL.dir/requires

bin/CMakeFiles/DL.dir/clean:
	cd /Users/yixuanhe/code/study/DL/build/bin && $(CMAKE_COMMAND) -P CMakeFiles/DL.dir/cmake_clean.cmake
.PHONY : bin/CMakeFiles/DL.dir/clean

bin/CMakeFiles/DL.dir/depend:
	cd /Users/yixuanhe/code/study/DL/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/yixuanhe/code/study/DL /Users/yixuanhe/code/study/DL/src /Users/yixuanhe/code/study/DL/build /Users/yixuanhe/code/study/DL/build/bin /Users/yixuanhe/code/study/DL/build/bin/CMakeFiles/DL.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : bin/CMakeFiles/DL.dir/depend

