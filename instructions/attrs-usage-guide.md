# Conventions of using `attrs` python library

We use `attrs` to define data classes in our codebase. This guide explains the conventions we follow when using `attrs`.

## Rules
- use `define` and `field` to define data classes
- prefer to use `kw_only=True` in `define` to enforce keyword-only arguments, unless otherwise specified

## Example

```python
from attrs import define, field

@define(kw_only=True)
class MyDataClass:
    name: str = field(default="default_name", metadata={"help": "Name of the object"})
    value: int = field(default=0, metadata={"help": "Value of the object"})
``` 