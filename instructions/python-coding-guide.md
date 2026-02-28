# Python Coding Guide

## Overview
This guide outlines the coding conventions and best practices for Python class design across all projects.

## General Guidelines

### 1. Import Style
- Use absolute imports whenever possible
- Avoid relative imports unless absolutely necessary
- Group imports: standard library, third-party, local modules

```python
# Good - Absolute imports
from typing import Any, Optional
from pathlib import Path
import json
import numpy as np
from myproject.core.models import BaseModel
from myproject.utils.helpers import validate_data

# Avoid - Relative imports
from ..core.models import BaseModel
from .helpers import validate_data
```

### 2. Documentation Style
- Use NumPy documentation style for all docstrings
- Document all modules, classes, methods, and functions
- Private functions/classes (single leading underscore) MUST still have a brief docstring explaining intent (one line is fine)
- Include parameter types, return types, and examples

### 3. Module Documentation
- Add module-level docstring at the top of each file
- Explain the module's purpose and main components
- List key classes/functions if applicable

```python
"""
Core data processing module for document handling.

This module provides the primary classes and functions for loading,
processing, and managing document data. It includes validation utilities
and export functionality.

Classes
-------
Document : Main document class for content management
DocumentProcessor : Processing pipeline for documents

Functions
---------
load_document : Load document from various sources
validate_content : Validate document content format
"""

from typing import Any, Optional
import json
```

## Core Principles

### 1. Member Variable Naming
- Prefix all member variables with `m_`
- Define all member variables in `__init__()`
- Initialize to `None` by default
- Use strong typing with type hints

Note on scope
- The `m_` naming convention applies to generic classes that encapsulate
  behavior/state (service, helper, controller, algorithm classes, etc.).
- Do not use `m_` for data model classes built with `attrs` or `pydantic`.
  These represent structured data and should use regular field names to ensure
  compatibility with validators, serializers, and schema generation.

Example (pydantic data model)
```python
from typing import Optional
from pydantic import BaseModel

class User(BaseModel):
    id: int
    name: str
    email: Optional[str] = None
```

Example (`attrs` data model)
```python
from typing import Optional
import attrs

@attrs.define
class Person:
    first_name: str
    last_name: str
    age: int = 0
    nickname: Optional[str] = None
```

```python
from typing import Any, Optional

class MyClass:
    def __init__(self) -> None:
        self.m_data: Optional[Any] = None
        self.m_config: Optional[dict] = None
        self.m_status: Optional[str] = None
```

### 2. Read-Only Properties
- Use `@property` decorator for read-only access
- No property setters allowed
- Type annotate return values

```python
from typing import Any, Optional

class MyClass:
    def __init__(self) -> None:
        self.m_data: Optional[Any] = None
    
    @property
    def data(self) -> Optional[Any]:
        return self.m_data
```

### 3. Explicit Setter Methods
- Use `set_xxx()` methods for modifications
- Include validation when needed
- Type annotate parameters and return values

```python
from typing import Any, Optional

class MyClass:
    def __init__(self) -> None:
        self.m_data: Optional[Any] = None
    
    def set_data(self, data: Any) -> None:
        if data is None:
            raise ValueError("Data cannot be None")
        self.m_data = data
```

### 4. Constructor and Factory Pattern
- Constructors take no arguments (except `self`)
- Use factory methods `cls.from_xxx()` for initialization
- Type annotate all methods and return types

```python
from typing import Any, Optional, Type, TypeVar, Dict

T = TypeVar('T', bound='MyClass')

class MyClass:
    def __init__(self) -> None:
        self.m_data: Optional[Any] = None
        self.m_config: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_file(cls: Type[T], file_path: str) -> T:
        instance = cls()
        with open(file_path, 'r') as f:
            instance.m_data = f.read()
        return instance
    
    @classmethod
    def from_config(cls: Type[T], config_dict: Dict[str, Any]) -> T:
        instance = cls()
        instance.m_config = config_dict
        return instance
```

## Complete Example

```python
"""
Document management module.

This module provides the Document class for handling text documents
with metadata support and various loading mechanisms.

Classes
-------
Document : Primary document class with content and metadata management
"""

from typing import Any, Optional, Type, TypeVar, Dict

T = TypeVar('T', bound='Document')

class Document:
    """
    A document class for managing text content with metadata.
    
    This class provides a structured way to handle documents with
    content, titles, and metadata. It supports loading from files
    and creating from strings with validation.
    
    Attributes
    ----------
    content : str or None
        The main text content of the document
    title : str or None
        The document title
    metadata : dict or None
        Additional metadata as key-value pairs
    file_path : str or None
        Path to the source file if loaded from file
    
    Examples
    --------
    Create an empty document and configure it:
    
    >>> doc = Document()
    >>> doc.set_content("Hello World")
    >>> doc.set_title("My Document")
    
    Load from file:
    
    >>> doc = Document.from_file("document.txt")
    >>> print(doc.content)
    """
    
    def __init__(self) -> None:
        """
        Initialize an empty document instance.
        
        All member variables are initialized to None and can be
        configured later using setter methods or factory methods.
        """
        self.m_content: Optional[str] = None
        self.m_title: Optional[str] = None
        self.m_metadata: Optional[Dict[str, Any]] = None
        self.m_file_path: Optional[str] = None
    
    @property
    def content(self) -> Optional[str]:
        """
        Get the document content.
        
        Returns
        -------
        str or None
            The document content, or None if not set
        """
        return self.m_content
    
    @property
    def title(self) -> Optional[str]:
        """
        Get the document title.
        
        Returns
        -------
        str or None
            The document title, or None if not set
        """
        return self.m_title
    
    @property
    def metadata(self) -> Optional[Dict[str, Any]]:
        """
        Get the document metadata.
        
        Returns
        -------
        dict or None
            The document metadata as key-value pairs, or None if not set
        """
        return self.m_metadata
    
    @property
    def file_path(self) -> Optional[str]:
        """
        Get the source file path.
        
        Returns
        -------
        str or None
            The path to the source file, or None if not loaded from file
        """
        return self.m_file_path
    
    def set_content(self, content: str) -> None:
        """
        Set the document content.
        
        Parameters
        ----------
        content : str
            The text content to set
            
        Raises
        ------
        TypeError
            If content is not a string
        """
        if not isinstance(content, str):
            raise TypeError("Content must be a string")
        self.m_content = content
    
    def set_title(self, title: str) -> None:
        """
        Set the document title.
        
        Parameters
        ----------
        title : str
            The title to set
        """
        self.m_title = title
    
    def set_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Set the document metadata.
        
        Parameters
        ----------
        metadata : dict
            Dictionary containing metadata key-value pairs
        """
        self.m_metadata = metadata
    
    @classmethod
    def from_file(cls: Type[T], file_path: str) -> T:
        """
        Create a document by loading content from a file.
        
        Parameters
        ----------
        file_path : str
            Path to the file to load
            
        Returns
        -------
        Document
            A new document instance with content loaded from file
            
        Examples
        --------
        >>> doc = Document.from_file("my_document.txt")
        >>> print(doc.content)
        """
        instance = cls()
        with open(file_path, 'r') as f:
            instance.m_content = f.read()
        instance.m_file_path = file_path
        return instance
    
    @classmethod
    def from_string(cls: Type[T], content: str, title: Optional[str] = None) -> T:
        """
        Create a document from a string with optional title.
        
        Parameters
        ----------
        content : str
            The document content
        title : str, optional
            The document title (default is None)
            
        Returns
        -------
        Document
            A new document instance with the specified content
            
        Examples
        --------
        >>> doc = Document.from_string("Hello World", "My Title")
        >>> print(doc.title)
        My Title
        """
        instance = cls()
        instance.set_content(content)
        if title:
            instance.set_title(title)
        return instance
```

## Benefits

- **Clear State**: `m_` prefix shows internal state
- **Controlled Access**: Read-only properties prevent accidents
- **Explicit Changes**: Setter methods provide validation
- **Flexible Creation**: Factory methods for different scenarios
- **Consistent API**: Predictable patterns across classes
- **Type Safety**: Strong typing prevents runtime errors

## Usage

```python
from typing import Optional

# Empty instance
doc: Document = Document()
doc.set_content("Hello World")
doc.set_title("My Document")

# From factory methods
doc: Document = Document.from_file("document.txt")
doc: Document = Document.from_string("Content", "Title")

# Read-only access
content: Optional[str] = doc.content
title: Optional[str] = doc.title
```

## Type Hints Guidelines

- Use `typing` module for compatibility across Python versions
- Use `Optional[T]` for nullable types
- Use `Any` for unknown or dynamic types
- Use `TypeVar` for generic class methods
- Type annotate all method parameters and return values
- Prefer specific types over `Any` when possible

## Documentation Guidelines

### NumPy Style Docstrings
- Use NumPy documentation style for consistency
- Include Parameters, Returns, Raises, and Examples sections
- Document all public methods, functions, and classes

#### Class Example
```python
class Calculator:
    """
    A simple calculator for basic arithmetic operations.
    
    This class provides methods for addition, subtraction, multiplication,
    and division with input validation and error handling.
    
    Parameters
    ----------
    precision : int, optional
        Number of decimal places for results (default is 2)
    
    Attributes
    ----------
    precision : int
        The precision setting for calculations
    history : list
        List of previous calculations
    
    Examples
    --------
    >>> calc = Calculator(precision=3)
    >>> result = calc.add(2.5, 3.7)
    >>> print(result)
    6.200
    """
    
    def __init__(self, precision: int = 2) -> None:
        """
        Initialize the calculator.
        
        Parameters
        ----------
        precision : int, optional
            Number of decimal places (default is 2)
        """
        self.precision = precision
        self.history = []
```

#### Function Example
```python
def calculate_mean(data: List[float], exclude_outliers: bool = False) -> float:
    """
    Calculate the arithmetic mean of a list of numbers.
    
    Computes the average value of the input data with optional
    outlier exclusion using the IQR method.
    
    Parameters
    ----------
    data : list of float
        Input numerical data
    exclude_outliers : bool, optional
        Whether to exclude outliers before calculation (default is False)
        
    Returns
    -------
    float
        The arithmetic mean of the data
        
    Raises
    ------
    ValueError
        If data is empty or contains only NaN values
    TypeError
        If data contains non-numeric values
        
    Examples
    --------
    >>> data = [1.0, 2.0, 3.0, 4.0, 5.0]
    >>> calculate_mean(data)
    3.0
    
    >>> calculate_mean([1, 2, 100], exclude_outliers=True)
    1.5
    """
    # Implementation here
    pass
```

### Module Documentation Template
```python
"""
Brief description of the module.

Longer description explaining the module's purpose, main functionality,
and how it fits into the larger project. Include any important usage
notes or dependencies.

Classes
-------
ClassName1 : Brief description
ClassName2 : Brief description

Functions
---------
function_name1 : Brief description
function_name2 : Brief description

Examples
--------
Basic usage example:

>>> from mymodule import ClassName1
>>> obj = ClassName1()
>>> result = obj.some_method()

Notes
-----
Any important notes about the module, such as:
- Performance considerations
- Thread safety
- Dependencies
- Version compatibility
"""
```
