cmake_minimum_required(VERSION 2.8.3)

project(autofocus)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  image_transport
  std_msgs
  cv_bridge
)

find_package(OpenCV 3.2.0 REQUIRED)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  include
)

catkin_package()

add_executable(${PROJECT_NAME} src/autofocus.cpp)

target_link_libraries(${PROJECT_NAME}
${catkin_LIBRARIES}
${OpenCV_LIBRARIES}
X11
)