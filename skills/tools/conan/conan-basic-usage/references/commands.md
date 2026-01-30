# Conan Common Commands Cheat Sheet

## Creation & Setup

- `conan new cmake_lib -d name=mypkg -d version=1.0`: Create a new CMake-based library recipe.
- `conan new meson_lib -d name=mypkg -d version=1.0`: Create a new Meson-based library recipe.
- `conan config install <url>`: Install configuration (profiles, remotes, settings) from a URL or local path.
- `conan profile detect`: Detect the default profile for the current machine.

## Dependency Management

- `conan install . --build=missing`: Install dependencies from `conanfile.py` (or `.txt`), building binaries from source if not found.
- `conan install . --output-folder=build`: Install dependencies and generate files in a specific output folder.
- `conan graph info .`: Show the dependency graph without installing.

## Building & Packaging

- `conan create .`: Build the package from the recipe in the current directory and export it to the local cache.
- `conan create . --version=1.0`: Override the version in the recipe.
- `conan build .`: Run the `build()` method of the recipe locally (useful for debugging).
- `conan export .`: Export the recipe to the local cache without building.
- `conan test test_package mypkg/1.0`: Run the test package consumer against the created package.

## Repository & Cache

- `conan search <pattern>`: Search for packages in the local cache or remote.
- `conan upload <pattern> -r <remote>`: Upload packages to a remote repository.
- `conan list`: List recipes and binaries in the local cache.
- `conan remote list`: List configured remote repositories.
- `conan remote add <name> <url>`: Add a new remote repository.
- `conan remove <pattern>`: Remove packages from the local cache.

## Inspection

- `conan inspect .`: Inspect the recipe in the current directory.
- `conan version`: Display the installed Conan version.
