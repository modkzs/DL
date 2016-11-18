add_subdirectory(Layer)
add_subdirectory(Util)

ADD_EXECUTABLE(DL main.cpp)

target_link_libraries(DL LAYER)
target_link_libraries(DL UTIL)