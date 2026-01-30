# Conan Project Directory Structure Best Practices

## Standard Layout

For a standard C++ library managed by Conan, the recommended directory structure promotes a clean separation of concerns and integrates seamlessly with build systems like CMake.

```text
my-project/
├── conanfile.py           # The recipe defining the package
├── CMakeLists.txt         # The main build script
├── README.md              # Documentation
├── include/
│   └── myproject/         # Public headers (namespaced)
│       └── mylib.h
├── src/
│   └── mylib.cpp          # Implementation files
└── test_package/          # A standalone consumer project (Critical!)
    ├── conanfile.py
    ├── CMakeLists.txt
    └── src/
        └── example.cpp
```

## Key Components

### 1. `conanfile.py` & The `layout()` Method
In Conan 2.0, defining a `layout()` method in your `conanfile.py` is the gold standard. It tells Conan exactly where to expect sources and where to put build artifacts.

**Best Practice:** Use the built-in `cmake_layout` for CMake projects.

```python
from conan import ConanFile
from conan.tools.cmake import cmake_layout

class MyPkg(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeDeps", "CMakeToolchain"

    def layout(self):
        # Automatically sets up standard CMake folders (build/, build/Release, etc.)
        # compatible with IDEs like VS Code and CLion.
        cmake_layout(self)
```

### 2. `test_package/`
This directory is **mandatory** for a healthy Conan package. It is a minimal "client" application that consumes the package you are building.
- **Purpose**: Verifies that the package was created correctly and can be linked against.
- **Execution**: Runs automatically when you execute `conan create .`.
- **Structure**: It has its own `conanfile.py` (usually containing a `test()` method) and `CMakeLists.txt`.

### 3. Public vs. Private Headers
- **`include/myproject/`**: Put headers that users of your library need here. Using a subdirectory (namespace) prevents filename collisions.
- **`src/`**: Put private headers (not exposed to consumers) and `.cpp` files here.

### 4. Out-of-Source Builds
Conan (and CMake) strongly discourages building inside the source tree.
- When you run `conan install .`, `cmake_layout` will direct generated files to a `build/` (or similar) directory.
- Keep your root directory clean of `*.o`, `*.a`, or `*.dll` files.

## Monorepo vs. Single Package

### Single Package (Repo = Package)
The structure above is for a "one repo, one package" model. This is the simplest and most common approach.

### Monorepo (Multiple Packages)
For repositories containing multiple libraries:

```text
monorepo/
├── common/
│   ├── conanfile.py
│   └── ...
├── network/
│   ├── conanfile.py
│   └── ...
└── conanfile.py (Optional: A workspace or "umbrella" recipe)
```
Each sub-folder acts as a standalone package with its own recipe.
