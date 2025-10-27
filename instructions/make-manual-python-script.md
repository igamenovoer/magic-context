you are tasked to create a Python script that is intended to be manually executed by users, as such:
- DO NOT use testing frameworks like pytest or unittest
- DO NOT include `if __name__ == "__main__":` block
- Prefer scripting in global scope, using functions only for logical grouping, so that users can inspect and run parts of the code as needed, particularly in interactive environments like Jupyter notebooks
- Consider compatibility with jupyter notebooks, avoiding constructs that may not work well in such environments