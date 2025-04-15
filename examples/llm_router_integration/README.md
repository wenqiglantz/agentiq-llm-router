<!--
SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->


# LLM Router Integration

## Table of Contents

- [LLM Router Integration](#llm-router-integration)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Key Features](#key-features)
  - [Installation and Setup](#installation-and-setup)
    - [Setup Virtual Environment and Install AgentIQ](#setup-virtual-environment-and-install-agentiq)
    - [Install this Workflow](#install-this-workflow)
    - [Set Up API Keys](#set-up-api-keys)
  - [Configuration](#configuration)
  - [Example Usage](#example-usage)
    - [Run the Workflow](#run-the-workflow)
  - [Deployment-Oriented Setup](#deployment-oriented-setup)
    - [Build the Docker Image](#build-the-docker-image)
    - [Run the Docker Container](#run-the-docker-container)
    - [Test the API](#test-the-api)

## Overview

The LLM Router Integration example demonstrates how to build an AgentIQ workflow that integrates with an LLM router service. This workflow allows for intelligent routing of user queries to appropriate language models based on the query content, leveraging NVIDIA's routing infrastructure to optimize model selection for different types of tasks.

## Key Features

- **LLM Routing**: Automatically routes queries to the most appropriate language model based on the query content.
- **Policy-Based Routing**: Supports different routing policies like task-based routing.
- **Triton Integration**: Uses Triton Inference Server for efficient model serving and routing.
- **Error Handling**: Robust error handling for API calls and request processing.
- **Agentic Workflows**: Fully configurable via YAML for flexibility and productivity.
- **Logging and Telemetry**: Comprehensive logging and tracing for monitoring and debugging.

## Installation and Setup

### Setup Virtual Environment and Install AgentIQ

If you have not already done so, follow the instructions in the [Install Guide](../../docs/source/intro/install.md) to create the development environment and install AgentIQ.

### Install this Workflow

From the root directory of the AgentIQ library, run the following command:

```bash
uv pip install -e examples/llm_router_integration
```

### Set Up API Keys

You need to set up your NVIDIA API key as an environment variable to access NVIDIA AI services. If you have not already done so, follow the [Obtaining API Keys](../../docs/source/intro/get-started.md#obtaining-api-keys) instructions.

```bash
export NVIDIA_API_KEY=<YOUR_API_KEY>
```

## Configuration

The LLM Router Integration is configured via a YAML configuration file. The main configuration parameters are:

- `llm_router`: Configuration for the LLM router, including API key, base URL, routing policy, and strategy.
- `llm_router_tool`: Configuration for the tool that interfaces with the LLM router.
- `workflow`: Defines the workflow type and the tools it uses.

You can customize these settings in the `examples/llm_router_integration/configs/config.yml` file.

## Example Usage

### Run the Workflow

Run the following command from the root of the AgentIQ repo to execute this workflow with a sample query:

```bash
aiq run --config_file examples/llm_router_integration/configs/config.yml --input "Tell me the difference between Model Context Protocol and Agent2Agent"
```

The call was classified as "Open QA" and handled by `llama-3.1-70b-instruct` model in LLM Router.

**expected output:**

```
2025-04-15 17:05:36,976 - aiq.runtime.loader - WARNING - Loading module 'aiq.tool.register' from entry point 'aiq_tools' took a long time (585.393429 ms). Ensure all imports are inside your registered functions.
2025-04-15 17:05:37,483 - aiq.runtime.loader - WARNING - Loading module 'aiq_automated_description_generation.register' from entry point 'aiq_automated_description_generation' took a long time (458.978176 ms). Ensure all imports are inside your registered functions.
2025-04-15 17:05:37,570 - aiq.cli.commands.start - INFO - Starting AgentIQ from config file: 'examples\llm_router_integration\configs\config.yml'
2025-04-15 17:05:37,570 - aiq.cli.commands.start - WARNING - The front end type in the config file (fastapi) does not match the command name (console). Overwriting the config file front end.
2025-04-15 17:05:37,687 - phoenix.config - INFO - 📋 Ensuring phoenix working directory: C:\Users\wglantz\.phoenix
2025-04-15 17:05:37,687 - phoenix.inferences.inferences - INFO - Dataset: phoenix_inferences_d10de5f6-c308-4142-9382-cf34f8c7e4ac initialized
2025-04-15 17:05:40,334 - aiq.profiler.utils - WARNING - Discovered frameworks: {<LLMFrameworkEnum.LANGCHAIN: 'langchain'>} in function llm_router_integration by inspecting source. It is recommended and more reliable to instead add the used LLMFrameworkEnum types in the framework_wrappers argument when calling @register_function.
Discovered frameworks: {<LLMFrameworkEnum.LANGCHAIN: 'langchain'>} in function llm_router_integration by inspecting source. It is recommended and more reliable to instead add the used LLMFrameworkEnum types in the framework_wrappers argument when calling @register_function.
2025-04-15 17:05:40,334 - aiq.profiler.decorators - INFO - Langchain callback handler registered

Configuration Summary:
--------------------
Workflow Type: llm_router_integration
Number of Functions: 1
Number of LLMs: 1
Number of Embedders: 0
Number of Memory: 0
Number of Retrievers: 0

2025-04-15 17:05:40,491 - aiq.front_ends.console.console_front_end_plugin - INFO - Processing input: ('Tell me the difference between Model Context Protocol and Agent2Agent',)
2025-04-15 17:05:40,491 - llm_router_integration.register - INFO - Sending query to LLM Router: Tell me the difference between Model Context Proto...
2025-04-15 17:05:40,839 - aiq.tool.llm_router_tool - INFO - Sending request to LLM Router...
2025-04-15 17:05:48,134 - httpx - INFO - HTTP Request: POST http://10.176.221.83:8084/v1/chat/completions "HTTP/1.1 200 OK"
2025-04-15 17:05:48,143 - aiq.tool.llm_router_tool - INFO - Prompt was classified as 'Open QA' & handled by model: meta/llama-3.1-70b-instruct
2025-04-15 17:05:48,143 - aiq.tool.llm_router_tool - INFO - Token usage: {'input_tokens': 22, 'output_tokens': 497, 'total_tokens': 519}
2025-04-15 17:05:48,154 - llm_router_integration.register - INFO - Successfully received response from LLM Router
2025-04-15 17:05:48,154 - aiq.observability.async_otel_listener - INFO - Intermediate step stream completed. No more events will arrive.
2025-04-15 17:05:48,154 - aiq.front_ends.console.console_front_end_plugin - INFO - --------------------------------------------------
Workflow Result:
["Two interesting topics in the realm of multi-agent systems!\n\nModel Context Protocol (MCP) and Agent2Agent (A2A) are both communication protocols used in multi-agent systems, but they serve different purposes and have different design goals.\n\n**Model Context Protocol (MCP)**\n\nMCP is a protocol designed to enable agents to share and reason about context models in a multi-agent system. A context model represents the agent's understanding of the environment, including entities, relationships, and actions. MCP allows agents to exchange and merge their context models, facilitating a shared understanding of the environment.\n\nThe main goals of MCP are:\n\n1. **Context sharing**: Enable agents to share their context models with each other.\n2. **Context merging**: Allow agents to merge their context models to create a more comprehensive and accurate understanding of the environment.\n\nMCP is designed to support applications where agents need to collaborate and reason about the environment, such as in smart spaces, ambient intelligence, or industrial automation.\n\n**Agent2Agent (A2A)**\n\nA2A, also known as the Agent-to-Agent Protocol, is a communication protocol designed to enable direct communication between agents in a multi-agent system. A2A allows agents to send and receive messages, enabling them to coordinate their actions, negotiate, and collaborate on tasks.\n\nThe main goals of A2A are:\n\n1. **Agent communication**: Enable direct communication between agents.\n2. **Action coordination**: Support coordination of actions among agents.\n\nA2A is designed to support applications where agents need to interact with each other, such as in distributed problem-solving, auctions, or negotiations.\n\n**Key differences**\n\nHere are the main differences between MCP and A2A:\n\n* **Purpose**: MCP focuses on sharing and merging context models, while A2A focuses on direct communication and action coordination between agents.\n* **Scope**: MCP is designed for context sharing and merging, whereas A2A is designed for general purpose agent communication.\n* **Message structure**: MCP messages typically contain context model information, while A2A messages can contain arbitrary data, such as text, images, or binary data.\n* **Complexity**: MCP is generally more complex than A2A, as it involves context model reasoning and merging.\n\nIn summary, if you need to enable agents to share and reason about context models, MCP is the way to go. If you need to enable direct communication and action coordination between agents, A2A is the more suitable choice."]
--------------------------------------------------
```

Let's try one more query:
```
aiq run --config_file examples/llm_router_integration/configs/config.yml --input "Write me a python function to compute the result of 345 multiplying 123?"
```
The call was classified as "code generation" and handled by `llama-3.3-nemotron-super-49b-v1`.

**expected output:**

```
2025-04-15 17:08:57,117 - aiq.runtime.loader - WARNING - Loading module 'aiq.tool.register' from entry point 'aiq_tools' took a long time (589.185715 ms). Ensure all imports are inside your registered functions.
2025-04-15 17:08:57,562 - aiq.runtime.loader - WARNING - Loading module 'aiq_automated_description_generation.register' from entry point 'aiq_automated_description_generation' took a long time (397.119284 ms). Ensure all imports are inside your registered functions.
2025-04-15 17:08:57,641 - aiq.cli.commands.start - INFO - Starting AgentIQ from config file: 'examples\llm_router_integration\configs\config.yml'
2025-04-15 17:08:57,641 - aiq.cli.commands.start - WARNING - The front end type in the config file (fastapi) does not match the command name (console). Overwriting the config file front end.
2025-04-15 17:08:57,730 - phoenix.config - INFO - 📋 Ensuring phoenix working directory: C:\Users\wglantz\.phoenix
2025-04-15 17:08:57,736 - phoenix.inferences.inferences - INFO - Dataset: phoenix_inferences_fdfa8092-23f7-4f74-9a23-29acb5eafe0d initialized
2025-04-15 17:08:59,985 - aiq.profiler.utils - WARNING - Discovered frameworks: {<LLMFrameworkEnum.LANGCHAIN: 'langchain'>} in function llm_router_integration by inspecting source. It is recommended and more reliable to instead add the used LLMFrameworkEnum types in the framework_wrappers argument when calling @register_function.
Discovered frameworks: {<LLMFrameworkEnum.LANGCHAIN: 'langchain'>} in function llm_router_integration by inspecting source. It is recommended and more reliable to instead add the used LLMFrameworkEnum types in the framework_wrappers argument when calling @register_function.
2025-04-15 17:08:59,987 - aiq.profiler.decorators - INFO - Langchain callback handler registered

Configuration Summary:
--------------------
Workflow Type: llm_router_integration
Number of Functions: 1
Number of LLMs: 1
Number of Embedders: 0
Number of Memory: 0
Number of Retrievers: 0

2025-04-15 17:09:00,153 - aiq.front_ends.console.console_front_end_plugin - INFO - Processing input: ('Write me a python function to compute the result of 345 multiplying 123?',)
2025-04-15 17:09:00,153 - llm_router_integration.register - INFO - Sending query to LLM Router: Write me a python function to compute the result o...
2025-04-15 17:09:00,490 - aiq.tool.llm_router_tool - INFO - Sending request to LLM Router...
2025-04-15 17:09:07,500 - httpx - INFO - HTTP Request: POST http://10.176.221.83:8084/v1/chat/completions "HTTP/1.1 200 OK"
2025-04-15 17:09:07,513 - aiq.tool.llm_router_tool - INFO - Prompt was classified as 'Code Generation' & handled by model: nvidia/llama-3.3-nemotron-super-49b-v1
2025-04-15 17:09:07,513 - aiq.tool.llm_router_tool - INFO - Token usage: {'input_tokens': 26, 'output_tokens': 470, 'total_tokens': 496}
2025-04-15 17:09:07,522 - llm_router_integration.register - INFO - Successfully received response from LLM Router
2025-04-15 17:09:07,522 - aiq.observability.async_otel_listener - INFO - Intermediate step stream completed. No more events will arrive.
2025-04-15 17:09:07,522 - aiq.front_ends.console.console_front_end_plugin - INFO - --------------------------------------------------
Workflow Result:
['Below is a simple Python function that computes the result of multiplying two numbers, specifically designed for your request with 345 and 123, but also adaptable for any two numbers by modifying the function parameters.\n\n### Simple Multiplication Function in Python\n\n```python\ndef multiply_numbers(a, b):\n    """\n    Returns the product of two numbers.\n    \n    Parameters:\n    - a (int or float): The first number.\n    - b (int or float): The second number.\n    \n    Returns:\n    - int or float: The product of a and b.\n    """\n    return a * b\n\n# Example Usage for your specific numbers\nif __name__ == "__main__":\n    num1 = 345\n    num2 = 123\n    result = multiply_numbers(num1, num2)\n    print(f"The result of {num1} multiplying {num2} is: {result}")\n```\n\n### How to Use This Function for Your Specific Query:\n\n1. **Copy the Code**: Copy the entire code block provided above.\n2. **Paste into a Python File**: Paste it into a file with a `.py` extension (e.g., `multiplication_example.py`).\n3. **Run the File**: Open a terminal or command prompt, navigate to the folder containing your file, and run it using Python (e.g., `python multiplication_example.py`).\n\n### Output for Your Specific Query:\n\nWhen you run this script with the provided numbers (345 and 123), the output will be:\n\n```\nThe result of 345 multiplying 123 is: 42435\n```\n\n### Making the Function More Interactive (Optional):\n\nIf you want to make the function interactive so users can input their own numbers without modifying the code, you can modify the `if __name__ == "__main__":` block like so:\n\n```python\n# Interactive Example Usage\nif __name__ == "__main__":\n    num1 = float(input("Enter the first number: "))\n    num2 = float(input("Enter the second number: "))\n    result = multiply_numbers(num1, num2)\n    print(f"The result of {num1} multiplying {num2} is: {result}")\n```\n\nWith this version, when you run the script, it will prompt you to enter two numbers before displaying their product.']
--------------------------------------------------
```

## Deployment-Oriented Setup

For a production deployment, you can use Docker:

### Build the Docker Image

Prior to building the Docker image, ensure that you have followed the steps in the [Installation and Setup](#installation-and-setup) section, and you are currently in the AgentIQ virtual environment.

From the root directory of the AgentIQ repository, build the Docker image:

```bash
docker build --build-arg AIQ_VERSION=$(python -m setuptools_scm) -t llm_router_integration -f examples/llm_router_integration/Dockerfile .
```

### Run the Docker Container

Deploy the container:

```bash
docker run -p 8000:8000 -e NVIDIA_API_KEY=<YOUR_API_KEY> llm_router_integration
```

### Test the API

Use the following curl command to test the deployed API:

```bash
curl -X 'POST' \
  'http://localhost:8000/generate' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"input_message": "What is the capital of France and what are some popular tourist attractions there?"}'
```

The API will respond with the output from the most appropriate language model based on your query content.
