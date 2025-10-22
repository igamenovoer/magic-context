# Project Principles and Constitution

## Targeting C++ Users

- prioritize modern C++ idioms (C++17/20) with clear, maintainable patterns
- ensure seamless integration with industry-standard libraries and frameworks
- code should be well-documented with Doxygen comments for new contributors
- favor RAII, smart pointers, and move semantics for resource management

## Coding Qualities

- prefer reusable, modular components with clear separation of concerns
- value clarity and maintainability over cleverness
- emphasize const correctness, type safety, and zero-cost abstractions
- design interfaces before implementations (header-driven development)

## C++ Style Guide

### 1.1 Naming conventions

- namespaces: lowercase, short, and specific (e.g., `inference`, `preprocess`)
- types (classes, structs, enums, type aliases): PascalCase (e.g., `ImageData`, `EngineConfig`)
- functions and methods: lower_snake_case (e.g., `load_model`, `resize_image`)
- variables (local/global): lower_snake_case (e.g., `batch_size`, `temp_buffer`)
- data members: prefix with `m_` (e.g., `m_name`, `m_device_id`)
- constants: `kConstantCase` (e.g., `kMaxBatchSize`), avoid ALL_CAPS except for macros
- enums: prefer `enum class` with PascalCase type and enumerators (e.g., `enum class Precision { FP32, FP16 };`)
- macros: ALL_CAPS with a project prefix when unavoidable (e.g., `PROJECT_EXPORT`)

### General guidelines

- use modern C++ features (auto, range-for, structured bindings, std::optional, std::variant)
- maintain const correctness: mark methods `const` when they don't modify state; use `const&` for read-only parameters
- prefer `#pragma once` over include guards for header files
- group includes in order: corresponding header, project headers, third-party libraries, standard library (separate with blank lines)
- use RAII for all resource management (memory, file handles, locks, CUDA resources)
- prefer stack allocation and `std::unique_ptr` over raw `new`/`delete`
- use `std::shared_ptr` only when shared ownership semantics are genuinely needed
- avoid `using namespace` in headers; minimize in implementation files
- use `[[nodiscard]]` for functions whose return values must be checked
- all C++ code should compile without warnings at `-Wall -Wextra -Wpedantic` level

### 1.2 Accessors and properties

- prefer property-style getters for read-only properties: `name() const`, not `get_name()` or `getName()`
- boolean accessors should read naturally: `is_enabled() const`, `has_value() const`, `empty() const`
- setters use `set_name(...)`; choose lower_snake_case to match functions in this codebase
- return by value for small trivially copyable types; return `const T&` for large/heavy objects when ownership stays with the class
- for non-owning string/byte views, consider `std::string_view`/`std::span<const T>` when lifetime safety is clear—document the lifetime contract
- avoid output parameters; prefer return values or dedicated result types (e.g., `std::optional`, `std::expected`)

### Functional classes

- follow object-oriented style with clear conventions for class design
- prefix member variables with `m_` and initialize them in constructor initializer lists
- provide read-only access via const getter methods (e.g., `std::string_view name() const noexcept`)
- make changes through explicit setter methods with validation (e.g., `void set_name(std::string name)`)
- keep constructors private or protected; use factory methods for object creation
- factory methods should return `std::shared_ptr<T>` or `std::unique_ptr<T>` for heap-allocated objects
- factory pattern example:

  ```cpp
  class MyClass {
  public:
      static std::shared_ptr<MyClass> create(const Config& config);

    [[nodiscard]] std::string_view name() const noexcept { return m_name; }
    bool is_enabled() const noexcept { return m_enabled; }
    void set_name(std::string name) { m_name = std::move(name); }
    void set_enabled(bool enabled) noexcept { m_enabled = enabled; }

  private:
      MyClass() = default;  // Private constructor

      std::string m_name;
    bool m_enabled{false};
      int m_value{0};  // In-class initialization
  };
  ```

### Data model structs

- use plain `struct` for data-only types (no `m_` prefix for public fields)
- define fields with clear types; use in-class member initialization for defaults
- keep business logic outside data structs; use separate service/helper classes for behavior
- make structs aggregate-initializable when possible (no private members, no user-defined constructors)
- document fields using Doxygen `///` or `/**` comments
- example pattern:
  ```cpp
  /// Configuration for inference engine.
  struct EngineConfig {
      std::string model_path;          ///< Path to model file
      int batch_size{1};               ///< Batch size for inference
      bool use_fp16{false};            ///< Enable FP16 precision
      std::optional<int> device_id;    ///< CUDA device ID (nullopt = CPU)
  };
  ```
- for more complex data types with invariants, use classes with validation

### Memory and resource management

- always use RAII wrappers for non-C++ resources (CUDA memory, file handles, etc.)
- create custom deleters for smart pointers when wrapping C APIs:

  ```cpp
  using CudaMemoryPtr = std::unique_ptr<void, std::function<void(void*)>>;

  CudaMemoryPtr allocate_cuda(size_t size) {
      void* ptr = nullptr;
      cudaMalloc(&ptr, size);
      return CudaMemoryPtr(ptr, [](void* p) { cudaFree(p); });
  }
  ```

- prefer move semantics over copying for large objects; implement move constructors/assignment when needed
- use `= delete` to explicitly disable copy/move when inappropriate

### Error handling

- use exceptions for error conditions (not error codes or return values)
- create custom exception types inheriting from `std::exception` or `std::runtime_error`
- document exception guarantees (no-throw, strong exception safety, basic exception safety)
- use `noexcept` for functions guaranteed not to throw
- for library boundaries or performance-critical paths, consider `std::expected` or `std::optional` for error reporting

### API design defaults

- mark single-argument converting constructors `explicit` to avoid unintended conversions
- use `override` on virtual overrides; add `final` where sub-classing must be prohibited
- default and delete special member functions intentionally: `=default`, `=delete`
- prefer move semantics; accept setter parameters by value and move-assign internally when beneficial
- prefer `enum class` over unscoped `enum`
- avoid owning raw pointers; ownership via `std::unique_ptr` (default) or `std::shared_ptr` (only when shared ownership is required)
- prefer `std::span`/`std::string_view` for non-owning parameters; document lifetime expectations
- avoid macros in APIs; use `constexpr`, `inline`, and templating instead

### Concurrency and thread safety

- document thread-safety guarantees in class/method comments
- use standard library primitives (`std::mutex`, `std::lock_guard`, `std::atomic`, etc.)
- prefer immutable data and message passing over shared mutable state
- use `const` methods to indicate thread-safe read operations
- mark non-thread-safe classes explicitly in documentation

## C++ Build Environment

- avoid ad-hoc dependency management; always use Conan (or CMake FetchContent as fallback)
- determine the project's build system in this order:
  1. **Conan** (preferred): check for `conanfile.py` or `conanfile.txt`; use `conan install` to manage dependencies
  2. **CMake FetchContent/ExternalProject**: for dependencies not available in Conan
  3. **Manual builds** (last resort): only for complex dependencies like ONNX Runtime with custom build flags
- maintain CMakeLists.txt with clear targets, dependencies, and installation rules
- use CMake find modules or custom Find\*.cmake scripts for locating external libraries
- provide custom Conan profiles when targeting specific environments (e.g., CUDA compute capability, compiler versions)

### Build workflow

```bash
# 1. Install dependencies via Conan
conan install . --output-folder=build --build=missing

# 2. Configure CMake with Conan toolchain
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=build/conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release

# 3. Build
cmake --build build --config Release -j$(nproc)

# 4. Test
cd build && ctest --output-on-failure
```

## Documentation Standards

- all public APIs must have Doxygen-style comments with:
  - brief description (one line after `///` or `/** */`)
  - detailed description with usage examples where helpful
  - `@param` for each parameter with description
  - `@return` for return value description
  - `@throws` for exceptions that may be thrown
  - `@note`, `@warning` for important remarks
- created markdown documents should be readable and well-structured with section numbers (x.y.z)
- in design docs, describe C++ interfaces (classes, functions, APIs) using C++ code blocks with Doxygen comments
- example:
  ```cpp
  /**
   * @brief Load a model from the specified path.
   *
   * This method initializes the inference engine with the model
   * at the given path. The engine must be in an unloaded state.
   *
   * @param model_path Absolute path to the model file
   * @param config Configuration for model loading
   * @return true if model loaded successfully, false otherwise
   * @throws std::runtime_error if file doesn't exist or is invalid
   * @note This method is not thread-safe
   *
   * Example usage:
   * @code
   * auto engine = ONNXEngine::create();
   * EngineConfig config{.batch_size = 4, .use_fp16 = true};
   * if (!engine->load_model("/path/to/model.onnx", config)) {
   *     // Handle error
   * }
   * @endcode
   */
  virtual bool load_model(const std::string& model_path,
                          const EngineConfig& config) = 0;
  ```

## Testing Requirements

We use three complementary test types: manual tests, unit tests, and integration tests.

**Determining test locations**: Before writing tests, establish the location conventions for the project:

1. First, check the project README.md or other project documentation for test location and organization instructions
2. If not documented, ask the user or use project-standard conventions if evident from existing test files
3. Common defaults (use as last resort): `tests/unit/`, `tests/integration/`, `tests/manual/` or similar

Use placeholders like `<unit_tests_root>`, `<integration_tests_root>`, `<manual_tests_root>` when referring to base test directories in the guidelines below.

### Manual tests (preferred for feature validation)

- provide manual test executables or scripts for major functionality
- design for interactive use with clear console output
- use minimal dependencies; avoid test frameworks in manual tests
- structure as standalone `main()` programs with straightforward logic
- prioritize visibility: print intermediate results, timings, and validation outcomes
- use command-line arguments (CLI11, cxxopts, etc.) for configuration
- typical location pattern: `<manual_tests_root>/<feature_area>/test_<name>.cpp`
- example:

  ```cpp
  // tests/manual/inference/test_onnx_load.cpp
  #include <iostream>
  #include "inference/onnx_engine.hpp"

  int main(int argc, char** argv) {
      if (argc < 2) {
          std::cerr << "Usage: " << argv[0] << " <model_path>\n";
          return 1;
      }

      auto engine = ONNXEngine::create();
      std::cout << "Loading model: " << argv[1] << "\n";

      if (!engine->load_model(argv[1], {})) {
          std::cerr << "Failed to load model\n";
          return 1;
      }

      std::cout << "Model loaded successfully\n";
      return 0;
  }
  ```

### Unit tests (targeted automation)

- use a framework like Catch2 (preferred), Google Test, or doctest
- location pattern: `<unit_tests_root>/<subdir>/test_<name>.cpp` where `<subdir>` mirrors the module being tested
- test individual components in isolation; use mocks/fakes for dependencies
- handle external resources via environment variables; log resolved paths in test output
- follow the Arrange-Act-Assert pattern
- use test fixtures for setup/teardown of complex test state
- example with Catch2:

  ```cpp
  #include <catch2/catch_test_macros.hpp>
  #include "preprocessing/cpu_preprocessor.hpp"

  TEST_CASE("CPU Preprocessor resizes images correctly", "[preprocessing]") {
      CPUPreprocessor preprocessor;
      ImageData input{/* ... */};

      SECTION("Resize to 224x224") {
          auto output = preprocessor.resize(input, 224, 224);
          REQUIRE(output.width == 224);
          REQUIRE(output.height == 224);
      }
  }
  ```

### Integration tests (system behavior over units)

- focus on end-to-end flows across modules/components
- fewer but higher-value tests compared to unit tests
- may reuse patterns from unit tests for environment/resource handling
- location pattern: `<integration_tests_root>/<subdir>/test_<name>.cpp`
- test realistic scenarios with actual file I/O, model loading, inference pipelines
- validate performance characteristics (not just correctness)

### Test execution

- use CTest for test orchestration: `cd build && ctest --output-on-failure`
- support filtering: `ctest -R <pattern>` or framework-specific filters
- generate coverage reports in CI (gcov/lcov for GCC/Clang)
- run tests with sanitizers (AddressSanitizer, UndefinedBehaviorSanitizer) in CI

## Performance and Optimization

### Benchmarking

- use Google Benchmark or similar framework for micro-benchmarks
- measure performance at module boundaries (not internal functions unless profiling shows bottlenecks)
- report statistics (mean, median, stddev) and throughput metrics
- example:

  ```cpp
  #include <benchmark/benchmark.h>
  #include "preprocessing/cpu_preprocessor.hpp"

  static void BM_CPUPreprocess(benchmark::State& state) {
      CPUPreprocessor preprocessor;
      ImageData input = load_test_image();

      for (auto _ : state) {
          auto output = preprocessor.process(input);
          benchmark::DoNotOptimize(output);
      }

      state.SetItemsProcessed(state.iterations());
  }
  BENCHMARK(BM_CPUPreprocess);
  ```

### Optimization guidelines

- profile before optimizing: use perf, gprof, or vendor tools (NVIDIA Nsight for CUDA)
- prefer algorithmic improvements over micro-optimizations
- enable compiler optimizations: `-O3`, `-march=native`, `-flto` for release builds
- use compiler hints: `[[likely]]`, `[[unlikely]]`, `__builtin_expect`, `__builtin_prefetch` sparingly
- for CUDA code: measure kernel launch overhead, memory transfer costs, and occupancy
- document performance characteristics and expected throughput in critical path code

## Code Review and Quality Assurance

### Pre-commit checks

- code must compile without warnings (`-Wall -Wextra -Wpedantic`)
- run clang-format or similar formatter for consistent style
- static analysis: run clang-tidy, cppcheck, or similar tools
- all unit tests must pass
- update documentation when changing public APIs

### Code review focus areas

- correct resource management (no leaks, proper RAII usage)
- const correctness and API design
- exception safety guarantees
- thread safety and data races (use ThreadSanitizer)
- performance implications of design choices
- clarity of documentation and examples
