add_library(hnsw
  hnsw.cpp
)

target_include_directories(hnsw PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)