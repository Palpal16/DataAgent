cmake_minimum_required(VERSION 3.15)
project(resultcmp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(resultcmp src/main.cpp)

# On Windows, set console subsystem explicitly if needed
if (MSVC)
  target_compile_options(resultcmp PRIVATE /W4 /permissive-)
else()
  target_compile_options(resultcmp PRIVATE -Wall -Wextra -Wpedantic)
endif()


