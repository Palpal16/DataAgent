cmake_minimum_required(VERSION 3.15)
project(resultcmp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(resultcmp src/main.cpp)
add_library(runner_common src/runner_common.cpp)
target_include_directories(runner_common PUBLIC src)

add_executable(agent_config_runner src/agent_config_runner.cpp src/agent_config_runner_config.cpp)
target_link_libraries(agent_config_runner PRIVATE runner_common)

add_executable(agent_config_runner_jmeter src/agent_config_runner_jmeter.cpp)
target_link_libraries(agent_config_runner_jmeter PRIVATE runner_common)

# Optional: OpenMP acceleration for resultcmp (CSV comparator)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(resultcmp PRIVATE OpenMP::OpenMP_CXX)
  target_compile_definitions(resultcmp PRIVATE RESULTCMP_HAS_OPENMP=1)
endif()

# On Windows, set console subsystem explicitly if needed
if (MSVC)
  target_compile_options(resultcmp PRIVATE /W4 /permissive-)

  target_compile_options(agent_config_runner PRIVATE /W4 /permissive-)
  target_compile_options(agent_config_runner_jmeter PRIVATE /W4 /permissive-)
else()
  target_compile_options(resultcmp PRIVATE -Wall -Wextra -Wpedantic)

  target_compile_options(agent_config_runner PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(agent_config_runner_jmeter PRIVATE -Wall -Wextra -Wpedantic)
endif()

install(TARGETS agent_config_runner DESTINATION bin)


