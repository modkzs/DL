add_subdirectory(Layer)

ADD_EXECUTABLE(DL main.cpp Layer/BasicLayer.cpp Layer/BasicLayer.h)

target_link_libraries(DL LAYER)