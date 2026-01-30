# Build Artifacts & Runtime Layout

Understanding where Conan builds files and how it locates libraries at runtime is crucial for debugging and deployment.

## Build Directory Layout

When using `layout()` (specifically `cmake_layout`), Conan enforces a structured build directory to keep the source tree clean.

### During Development (`conan install .`)

When you run `conan install .`, Conan generates files in the folder specified by your layout (usually `build/` or `build/<config>`).

```text
my-project/
├── CMakeLists.txt
├── conanfile.py
└── build/                       # Created by cmake_layout(self)
    ├── Release/                 # If build_type=Release
    │   ├── generators/          # Conan-generated files (toolchain, deps)
    │   │   ├── conan_toolchain.cmake
    │   │   ├── conanbuild.sh    # Build environment setup
    │   │   └── conanrun.sh      # Runtime environment setup
    │   └── myapp*               # Your compiled executable (often here)
    └── Debug/                   # If build_type=Debug
        └── ...
```

*Note: The exact location of the executable depends on your `CMakeLists.txt` configuration, but `cmake_layout` sets `CMAKE_BINARY_DIR` to these folders.*

### Local Cache (`conan create .`)

When you create a package, the build happens in the Conan cache (`~/.conan2/p/...`), not your local folder. The artifacts are stored in a hashed directory structure.
- **Headers**: Copied to the `package/include` directory in the cache.
- **Libs/Binaries**: Copied to the `package/lib` or `package/bin` directory in the cache.

## Runtime: Finding Shared Libraries

When your application depends on shared libraries (`.so` on Linux, `.dll` on Windows, `.dylib` on macOS), the operating system needs to know where to find them at runtime. Conan handles this via the `VirtualRunEnv` generator.

### The `conanrun` Environment

After running `conan install .`, Conan generates a script to set up the runtime environment:

- **Linux/macOS**: `source build/Release/generators/conanrun.sh`
- **Windows**: `build\Release\generators\conanrun.bat`

**What it does:**
1.  **`LD_LIBRARY_PATH` (Linux)**: Appends paths to the `lib` folders of all your dependencies in the Conan cache.
2.  **`DYLD_LIBRARY_PATH` (macOS)**: Same as above.
3.  **`PATH` (Windows/Linux)**: Appends paths to the `bin` folders (for `.dll`s or tools).

### Workflow Example

```bash
# 1. Install dependencies
conan install . --build=missing

# 2. Build your project
cd build/Release
cmake ../.. -DCMAKE_TOOLCHAIN_FILE=generators/conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release
cmake --build .

# 3. Run your application
# WITHOUT conanrun: Likely fails (libfoo.so not found)
./myapp 

# WITH conanrun: Success!
source generators/conanrun.sh
./myapp
source generators/deactivate_conanrun.sh # Cleanup
```

### RPATH (Alternative)

If you don't want to rely on `conanrun.sh`, you can configure CMake to set the RPATH in your binary.
- This "bakes in" the paths to the shared libraries.
- **Pros**: You can run `./myapp` directly.
- **Cons**: The binary is tied to the specific location of libraries on your machine (not portable).

To enable this in CMake:
```cmake
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
```
