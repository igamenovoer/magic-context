# Template: How to [Task Name] with [Service/Library Name]

> **Note:** Use this template to document how to use specific APIs or SDKs within the project. This ensures consistency across our documentation. Replace the bracketed text with actual content.

## Question
How do I [perform a specific action/solve a specific problem] using [Specific Provider/Library]?

## Prerequisites

[Provide a checklist of what is needed before running this tutorial. Note that this tutorial will not cover how to set these up in detail; the user is expected to have handled these beforehand.]

- [ ] **Service Status:** [e.g., Ensure the `classic-vision-serve` service is running]
- [ ] **Environment:** [e.g., `pixi` is installed and `pixi install` has been run]
- [ ] **Configuration:** [e.g., `SERVICE_API_KEY` is set in your `.env` or environment]
- [ ] **Data:** [e.g., Sample media files are available in `datasets/samples/`]

## Implementation Idea
[Provide a high-level overview of the solution. Explain the key concepts, endpoints, or logical flow involved.]

*   **Approach:**
    1.  [Step 1: e.g., Initialize the client with configuration]
    2.  [Step 2: e.g., Construct the request payload]
    3.  [Step 3: e.g., Parse the response]

## Critical Example Code

[Provide clear, self-contained, and copy-pasteable code snippets. **Write rich comments in the code examples to explain what each part does step-by-step**, as this is intended to be a tutorial for users who might be new to this specific API or SDK.]

### Python SDK Implementation

```python
import os
# Import necessary modules
from specific_library import Client, models

def run_example():
    """
    Demonstrates how to [perform task].
    """
    # 1. Configuration / Initialization
    api_key = os.getenv("SERVICE_API_KEY")
    if not api_key:
        raise ValueError("Please set SERVICE_API_KEY")
        
    client = Client(api_key=api_key)

    # 2. Prepare Data
    payload = {
        "input": "example data",
        "options": {"verbose": True}
    }

    # 3. Execute Request
    try:
        response = client.resource.create(**payload)
        
        # 4. Process Response
        print(f"Success! ID: {response.id}")
        return response
        
    except Exception as e:
        print(f"Error occurred: {e}")
        return None

if __name__ == "__main__":
    run_example()
```

### RESTful API Equivalent (cURL)

```bash
# Ensure API_KEY is set in your environment
curl -X POST "https://api.provider.com/v1/resource" \
     -H "Authorization: Bearer $API_KEY" \
     -H "Content-Type: application/json" \
     -d 
{
           "input": "example data",
           "options": {
             "verbose": true
           }
         }
```

## Input and Output

### Input
[Describe the input parameters, data types, or provide a sample JSON object representing the input.]

*   `input` (str): The main data to process.
*   `options` (dict): Optional configuration flags.

> **Image Handling (Input):**
> *   If **≤ 5 images**: Display them directly using `![alt text](path/to/image.png)`.
> *   If **> 5 images**: Provide an itemized list of image paths/descriptions.

### Output
[Show the expected output, console logs, or a sample JSON response body.]

> **Image Handling (Output):**
> *   If **≤ 5 images**: Display the resulting images directly.
> *   If **> 5 images**: Provide an itemized list of output image paths.

```json
{
  "id": "res_123456789",
  "status": "completed",
  "data": {
    "processed_result": "..."
  },
  "created_at": "2023-10-27T10:00:00Z"
}
```

