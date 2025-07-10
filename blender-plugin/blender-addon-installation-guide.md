# Blender Add-on Installation Guide

## HEADER
- **Purpose**: Generic guide for installing Blender add-ons during development and deployment
- **Status**: Active
- **Date**: 2025-07-10
- **Dependencies**: Blender 4.0+, zip file creation tools
- **Target**: AI assistants and developers working on Blender plugins

## Overview

This guide covers how to install Blender add-ons, offering both a general approach for any add-on and specific instructions for plugins under development.

---

## Part 1: General Guide to Installing Blender Add-ons

This section explains the standard methods for installing any Blender add-on from a `.zip` file.

### Method 1: Using the Graphical User Interface (GUI)

This is the most common and straightforward method for installing add-ons.

1.  **Open Blender.**
2.  Go to `Edit > Preferences` from the top menu bar.
3.  In the Preferences window, select the `Add-ons` tab.
4.  Click the `Install...` button. This will open Blender's file browser.
5.  Navigate to and select the `.zip` file for the add-on you want to install.
6.  Click `Install Add-on`.
7.  After installation, the add-on will appear in the list. **Enable it by ticking the checkbox** next to its name.
8.  Your preferences may be configured to save automatically. If not, you can save them manually to ensure the add-on remains enabled for future sessions. In the bottom-left of the Preferences window, click the hamburger menu and select `Save Preferences`.

### Method 2: Using the Command Line Interface (CLI)

For automated setups or users who prefer the terminal, you can install and enable add-ons using a single command, without needing to create any extra script files. This method works for Blender 4.0 and newer.

**Prerequisites:**
*   You must know the path to the add-on's `.zip` file.
*   You must know the add-on's **module name**. This is the name Blender uses to identify the add-on internally. It's typically the name of the main `.py` file or the folder containing the `__init__.py` file. For a zipped add-on, the module name is often the same as the zip file's name (e.g., `my_addon.zip` -> `my_addon`).

For convenience, it's recommended to set the `BLENDER_EXEC_PATH` environment variable to the absolute path of your Blender executable.

#### Command Template

The command uses Blender's `--python-expr` argument to run Python code directly.

**Linux / macOS**
```bash
ADDON_ZIP_PATH="/path/to/your/addon.zip"
ADDON_MODULE_NAME="addon_module_name"

"$BLENDER_EXEC_PATH" --background --python-expr "import bpy; bpy.ops.preferences.addon_install(filepath='$ADDON_ZIP_PATH', overwrite=True); bpy.ops.preferences.addon_enable(module='$ADDON_MODULE_NAME'); bpy.ops.wm.save_userpref()"
```

**Windows (PowerShell)**
```powershell
$env:ADDON_ZIP_PATH = "C:\path\to\your\addon.zip"
$env:ADDON_MODULE_NAME = "addon_module_name"

& $env:BLENDER_EXEC_PATH --background --python-expr "import bpy; bpy.ops.preferences.addon_install(filepath='$env:ADDON_ZIP_PATH', overwrite=True); bpy.ops.preferences.addon_enable(module='$env:ADDON_MODULE_NAME'); bpy.ops.wm.save_userpref()"
```

---

## Part 2: Installing Your Plugin Under Development

Apply the knowledge from Part 1 to install your own plugin during development.

### Step 1: Prepare the Add-on File

First, you need to create the plugin zip file from the source directory.

```bash
# Navigate to your addon directory from the project root
cd path/to/addon/directory

# Create the zip file containing the addon
zip -r plugin-in-development.zip plugin-in-development/
```

This will create `plugin-in-development.zip` in your addon directory. The **module name** for this add-on is `plugin-in-development`.

### Step 2: Install the Add-on

#### Using the GUI

Follow the steps in **Part 1, Method 1**. When prompted to select a file, choose the `plugin-in-development.zip` file you just created. After installing, search for your plugin name and enable it.

#### Using the CLI

Follow the steps in **Part 1, Method 2**. Use the following template with your specific paths and names.

**Linux / macOS**
```bash
# Run from the project's root directory
ADDON_ZIP_PATH="path/to/addon/directory/plugin-in-development.zip"
ADDON_MODULE_NAME="plugin-in-development"

"$BLENDER_EXEC_PATH" --background --python-expr "import bpy; bpy.ops.preferences.addon_install(filepath='$ADDON_ZIP_PATH', overwrite=True); bpy.ops.preferences.addon_enable(module='$ADDON_MODULE_NAME'); bpy.ops.wm.save_userpref()"
```

**Windows (PowerShell)**
```powershell
# Run from the project's root directory
$env:ADDON_ZIP_PATH = "path\to\addon\directory\plugin-in-development.zip"
$env:ADDON_MODULE_NAME = "plugin-in-development"

& $env:BLENDER_EXEC_PATH --background --python-expr "import bpy; bpy.ops.preferences.addon_install(filepath='$env:ADDON_ZIP_PATH', overwrite=True); bpy.ops.preferences.addon_enable(module='$env:ADDON_MODULE_NAME'); bpy.ops.wm.save_userpref()"
```

### Step 3: Verify the Installation

Verification depends on your plugin's functionality. Here are common approaches:

#### Check Add-on List
```python
# Via Python console in Blender
import bpy
print('plugin-in-development' in bpy.context.preferences.addons)
```

#### Check System Console for Log Messages

How to open the system console:
*   **Windows:** Go to `Window > Toggle System Console`.
*   **macOS/Linux:** Start Blender from a terminal. Log messages will appear in that terminal window.

Look for your plugin's registration messages, typically something like:
```
=== PLUGIN-IN-DEVELOPMENT REGISTRATION STARTING ===
Loading Plugin v1.0.0...
Plugin registered successfully
=== PLUGIN-IN-DEVELOPMENT REGISTRATION COMPLETED ===
```

#### Test Plugin Functionality
```python
# Check if plugin operators are registered
import bpy
print(hasattr(bpy.ops, 'your_plugin_category'))
print(hasattr(bpy.ops.your_plugin_category, 'your_operator'))

# Check if plugin panels are registered
print(hasattr(bpy.types, 'YourPluginPanel'))
```

### Development Workflow: Hot Reload

For rapid development iteration, use this pattern to update your plugin without restarting Blender:

```bash
# 1. Update plugin files
cp -r source_directory/ ~/.config/blender/4.4/scripts/addons/plugin-in-development/

# 2. Reload in Blender (via Python console)
import bpy
bpy.ops.preferences.addon_disable(module="plugin-in-development")
bpy.ops.preferences.addon_enable(module="plugin-in-development")
```

### Automated Installation Script

Create a development script for automated installation:

```bash
#!/bin/bash
# install_plugin_dev.sh

# Configuration
PLUGIN_NAME="plugin-in-development"
PLUGIN_SOURCE="./plugin_source/"
BLENDER_ADDONS_DIR="$HOME/.config/blender/4.4/scripts/addons/"

# Build and install
echo "Building plugin..."
cd $(dirname "$PLUGIN_SOURCE")
zip -r "${PLUGIN_NAME}.zip" "${PLUGIN_NAME}/"

echo "Installing plugin..."
"$BLENDER_EXEC_PATH" --background --python-expr "
import bpy
bpy.ops.preferences.addon_install(filepath='${PWD}/${PLUGIN_NAME}.zip', overwrite=True)
bpy.ops.preferences.addon_enable(module='${PLUGIN_NAME}')
bpy.ops.wm.save_userpref()
print('Plugin ${PLUGIN_NAME} installed successfully')
"

echo "Installation complete!"
```

### Troubleshooting Common Issues

#### Plugin Not Appearing in List
- Check that the zip file contains the correct directory structure
- Verify the `__init__.py` file has proper `bl_info` dictionary
- Check console for Python errors during installation

#### Plugin Installs But Won't Enable
- Look for import errors in the system console
- Check that all required dependencies are available
- Verify Python version compatibility

#### Hot Reload Not Working
- Some changes require full Blender restart (C extensions, etc.)
- Check that file permissions allow overwriting
- Verify the correct addon directory path

This guide provides a complete workflow for installing and developing Blender add-ons, from initial installation to rapid development iteration.