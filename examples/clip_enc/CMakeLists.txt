set (TEST_TARGET clip_enc)
add_executable(${TEST_TARGET} clip.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common)