# Internal Conan Files

Conan generates several internal files to manage package metadata and integrity. While you rarely need to edit these, understanding them helps with debugging.

## `conanmanifest.txt`

**Location**: Inside each package directory in the local cache (`~/.conan2/p/...`).

**Purpose**: Ensures package integrity. It lists all files in the package along with their MD5 checksums.

**Usage**:
- Conan uses this to verify files during downloads and uploads.
- You can manually trigger a check:
  ```bash
  conan cache check-integrity <package/version>
  ```

## `conaninfo.txt`

**Location**: Inside the package directory.

**Purpose**: Stores the configuration (settings, options, requirements) that was used to generate this specific binary package.

**Usage**:
- Allows Conan to calculate the `package_id` and determine if a binary is compatible with your current configuration.

## `conanfile.txt` vs `conanfile.py`

These are **user-facing** files, unlike the ones above.

- **`conanfile.txt`**: A simplified configuration file for *consuming* dependencies. cannot be used to create packages.
- **`conanfile.py`**: A full Python script (recipe) used to *create* packages and define complex logic for consuming them.
