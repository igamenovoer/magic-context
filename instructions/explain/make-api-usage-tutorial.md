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

## Step-by-Step with Code

[Provide a step-by-step breakdown of the code, mapping directly to the steps outlined in the **Implementation Idea**. **Select the primary programming language (Python, TypeScript, C++, etc.) based on the project context and user request.** If the project supports multiple interfaces (e.g., Python SDK vs HTTP API), provide the most relevant one or both if requested.]

### Step 1: [Name of Step 1, e.g., Client Initialization]

[Elaborate on the concept. Explain strictly why this step is necessary, any configuration nuances, security considerations (like API key handling), or architectural choices.]

```[language]
# [Example code in the chosen primary language]
# [Initialize client / setup connection]
# [Handle authentication securely (e.g., env vars)]
```

### Step 2: [Name of Step 2, e.g., Request Construction]

[Discuss the data model or payload structure. Explain specific parameters, why certain options might be enabled or disabled (trade-offs), and data validation considerations.]

```[language]
# [Construct payload / request object]
# [Set options and parameters]
```

### Step 3: [Name of Step 3, e.g., Execution and Handling]

[Explain the execution flow. Discuss synchronous vs. asynchronous behavior, network timeouts, and the importance of robust error handling.]

```[language]
# [Execute request / call function]
# [Handle success and error cases]
```

### Complete Runnable Script

[Below is the consolidated, copy-pasteable script combining all steps. It is self-contained and ready to run.]

```[language]
# [Insert complete runnable script here]
```

### [Optional] Alternative Interface (e.g., REST API / CLI)

[If applicable to the project context (e.g., a web service), provide the equivalent using standard tools like cURL or a secondary language requested by the user.]

```bash
# [Example cURL command or alternative language snippet]
```

## Input and Output

### Input
[Describe the input parameters, data types, or provide a sample data structure (e.g., object, dictionary, or payload) representing the input.]

*   `input` (type): Description of the main data to process.
*   `options` (type): Description of optional configuration flags.

> **Image Handling (Input):**
> *   If **≤ 5 images**: Display them directly using `![alt text](path/to/image.png)`.
> *   If **> 5 images**: Provide an itemized list of image paths/descriptions.

### Output
[Show the expected output, console logs, or a sample response body/object.]

> **Image Handling (Output):**
> *   If **≤ 5 images**: Display the resulting images directly.
> *   If **> 5 images**: Provide an itemized list of output image paths.

```[format]
# [Insert sample output, console logs, or object representation here]
```

## References

### Relevant Source Code
[Include links to the most relevant files in the codebase, such as client definitions, model schemas, or existing integration tests that serve as a reference.]

*   `path/to/relevant_file.py`: [Brief description of what is in this file]
*   `path/to/api_definition.ts`: [Brief description]

### Online Resources
[Provide links to official documentation, third-party libraries, or external articles used or referenced in this tutorial.]

*   [Official API Documentation](https://example.com/docs)
*   [Provider SDK Repository](https://github.com/provider/sdk)