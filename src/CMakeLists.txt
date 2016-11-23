add_subdirectory(Layer)
add_subdirectory(DAG)
add_subdirectory(ThreadPool)

ADD_EXECUTABLE(DL main.cpp)

target_link_libraries(DL LAYER)
target_link_libraries(DL DAG)
target_link_libraries(DL THREAD_POOL)