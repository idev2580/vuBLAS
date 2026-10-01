cmake_minimum_required(VERSION 3.14)
project(vuBLAS VERSION 0.1.0 LANGUAGES CXX)

if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(
        CMAKE_INSTALL_PREFIX
        "/opt/vucol"
        CACHE PATH "vuBLAS installation prefix"
        FORCE
    )
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(
    VUBLAS_BUILD_TESTS
    "Build vuBLAS tests and performance executables"
    ON
)
find_package(VUCOL CONFIG REQUIRED HINTS "${CMAKE_INSTALL_PREFIX}")

set(GEMM_NAIVE_TEMPLATE
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/templates/GemmNaiveFP32.comp"
)
set(MATMUL_NAIVE_TEMPLATE
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/templates/MatMulNaiveFP32.comp"
)
set(GEMM_OUT_PLACE_NAIVE_TEMPLATE
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/templates/GemmOutPlaceNaiveFP32.comp"
)
set(UNARY_ELEMENTWISE_TEMPLATE
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/templates/UnaryElementwiseFP32.comp"
)
set(BINARY_ELEMENTWISE_TEMPLATE
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/templates/BinaryElementwiseFP32.comp"
)
set(EMBEDDED_SHADER_TEMPLATES_CPP
    "${CMAKE_CURRENT_BINARY_DIR}/shaders/EmbeddedShaderTemplates.cpp"
)
add_custom_command(
    OUTPUT "${EMBEDDED_SHADER_TEMPLATES_CPP}"
    COMMAND ${CMAKE_COMMAND}
        -DGEMM_TEMPLATE=${GEMM_NAIVE_TEMPLATE}
        -DMATMUL_TEMPLATE=${MATMUL_NAIVE_TEMPLATE}
        -DGEMM_OUT_PLACE_TEMPLATE=${GEMM_OUT_PLACE_NAIVE_TEMPLATE}
        -DUNARY_ELEMENTWISE_TEMPLATE=${UNARY_ELEMENTWISE_TEMPLATE}
        -DBINARY_ELEMENTWISE_TEMPLATE=${BINARY_ELEMENTWISE_TEMPLATE}
        -DOUTPUT_FILE=${EMBEDDED_SHADER_TEMPLATES_CPP}
        -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedShaderTemplates.cmake"
    DEPENDS
        "${GEMM_NAIVE_TEMPLATE}"
        "${MATMUL_NAIVE_TEMPLATE}"
        "${GEMM_OUT_PLACE_NAIVE_TEMPLATE}"
        "${UNARY_ELEMENTWISE_TEMPLATE}"
        "${BINARY_ELEMENTWISE_TEMPLATE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedShaderTemplates.cmake"
    COMMENT "Embedding vuBLAS shader templates"
    VERBATIM
)

add_subdirectory(shaders)
add_subdirectory(src)
if(VUBLAS_BUILD_TESTS)
    add_subdirectory(tests)
endif()

add_library(
    vublas SHARED
    ${SOURCES}
    "${EMBEDDED_SHADER_TEMPLATES_CPP}"
)
add_library(vuBLAS::vublas ALIAS vublas)
set_target_properties(vublas PROPERTIES
    VERSION "${PROJECT_VERSION}"
    SOVERSION "${PROJECT_VERSION_MAJOR}"
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)
add_dependencies(vublas vublas_compile_shaders)
target_compile_features(vublas PUBLIC cxx_std_20)
target_include_directories(vublas
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${CMAKE_CURRENT_BINARY_DIR}/shaders"
)
target_link_libraries(vublas PUBLIC VUCOL::vucol)

install(
    TARGETS vublas
    EXPORT vuBLASTargets
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)
install(
    DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/vublas"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)
install(
    FILES "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/vuBLAS"
)
install(
    EXPORT vuBLASTargets
    FILE vuBLASTargets.cmake
    NAMESPACE vuBLAS::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/vuBLAS"
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/vuBLASConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/vuBLASConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/vuBLAS"
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/vuBLASConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMajorVersion
)
install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/vuBLASConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/vuBLASConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/vuBLAS"
)

function(copy_target_runtime_dlls target_name)
    if(WIN32)
        add_custom_command(TARGET ${target_name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_RUNTIME_DLLS:${target_name}>
                $<TARGET_FILE_DIR:${target_name}>
            COMMAND_EXPAND_LISTS
        )
    endif()
endfunction()

if(VUBLAS_BUILD_TESTS)
    enable_testing()
    include(FetchContent)

    if(MSVC)
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    endif()
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.14.0
    )
    FetchContent_MakeAvailable(googletest)
    add_executable(vublas_tests ${TEST_SOURCES})
    target_link_libraries(vublas_tests PRIVATE vublas GTest::gtest_main)
    copy_target_runtime_dlls(vublas_tests)

    include(GoogleTest)
    gtest_discover_tests(vublas_tests)

    add_executable(vublas_performance_tests ${PERFORMANCE_TEST_SOURCES})
    target_link_libraries(vublas_performance_tests PRIVATE vublas)
    copy_target_runtime_dlls(vublas_performance_tests)

    add_executable(
        vublas_performance_sweep_tests
        ${PERFORMANCE_SWEEP_TEST_SOURCES}
    )
    target_link_libraries(
        vublas_performance_sweep_tests
        PRIVATE vublas
    )
    copy_target_runtime_dlls(vublas_performance_sweep_tests)
endif()
