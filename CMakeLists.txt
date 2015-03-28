cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")


PROJECT(main)



find_package(VTK REQUIRED)
include(${VTK_USE_FILE})




add_executable(main MACOSX_BUNDLE main)




if(VTK_LIBRARIES)

  target_link_libraries(main ${VTK_LIBRARIES})

else()

  target_link_libraries(main vtkHybrid vtkWidgets)

endif()




find_package( OpenCV REQUIRED )

target_link_libraries(main ${OpenCV_LIBS})
