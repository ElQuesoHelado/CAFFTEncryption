cmake_minimum_required(VERSION 3.26)

# SET (CMAKE_CXX_COMPILER "/usr/bin/clang++" CACHE STRING "C++ compiler" FORCE)
project(CelularAutomata VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
# set(CMAKE_CXX_CLANG_TIDY clang-tidy)
set(CMAKE_EXPORT_COMPILE_COMMANDS=ON)
include(CTest)
enable_testing()
# set(CMAKE_CXX)
# set(CMAKE_BUILD_TYPE Debug)

add_subdirectory("/home/josealva/Software/kfr" kfr-bin)

find_package(OpenCV REQUIRED)
find_package(ImageMagick COMPONENTS Magick++)
find_library(LIBSND NAMES SndFile)
find_package(Boost COMPONENTS program_options REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

# add_subdirectory(plotting)

add_executable(CelularAutomata
main.cpp CA.hpp ImgHide.hpp encryption.hpp
encryption.cc dft_traits.hpp dft_audio.hpp)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
target_link_libraries(
  CelularAutomata
  ${OpenCV_LIBS}
  ImageMagick::Magick++
  sndfile
  kfr
  kfr_dft
  kfr_io
  # lwindow
  Boost::program_options)

# add_custom_command(TARGET CelularAutomata POST_BUILD
#   COMMAND ${CMAKE_COMMAND} -E copy_directory
#   ${CMAKE_SOURCE_DIR}/pictures/ $<TARGET_FILE_DIR:CelularAutomata>/pictures
# )

add_custom_command(TARGET CelularAutomata POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
  ${CMAKE_SOURCE_DIR}/pictures128/ $<TARGET_FILE_DIR:CelularAutomata>/pictures128
)

add_custom_command(TARGET CelularAutomata POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
  ${CMAKE_SOURCE_DIR}/sounds/ $<TARGET_FILE_DIR:CelularAutomata>/sounds
)

