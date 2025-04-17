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

# Multi Frameworks LLM Router 

We use the code of `multi_frameworks` as the foundation and modify it to build `multi_frameworks_llm_router`. We replaced the `langchain_research_tool` to call LLM Router.  Given the user input, the supervisor node classifies it as either being about `Retrieve`, `General`, or `Research` topic.  `Retrieve` topic gets routed to `llama_index_rag` tool, `General` topic gets routed to `haystack_chitchat_agent`, and `Research` tool gets routed to LLM Router to process. 

## Table of Contents

- [Multi Frameworks LLM Router](#multi-frameworks-llm-router)
  - [Table of Contents](#table-of-contents)
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

## Key Features

- **LLM Router Integration:** with integration with LLM Router, this workflow demostrates how we build workflows that can tap into the power of LLM Router. 
- **Custom-plug-in Tools:** with a basic llama-index RAG ingesting README from within this workflow
- **Custom Plugin System:** Developers can bring in new tools using plugins.
- **High-level API:** Enables defining functions that transform into asynchronous LangChain tools.
- **Agentic Workflows:** Fully configurable via YAML for flexibility and productivity.
- **Ease of Use:** Simplifies developer experience and deployment.


## Local Installation and Usage

If you have not already done so, follow the instructions in the [Install Guide](../../docs/source/intro/install.md) to create the development environment and install AgentIQ.

### Step 1: Set Your NVIDIA API Key Environment Variable
If you have not already done so, follow the [Obtaining API Keys](../../docs/source/intro/get-started.md#obtaining-api-keys) instructions to obtain an NVIDIA API key. You need to set your NVIDIA API key as an environment variable to access NVIDIA AI services.

```bash
export NVIDIA_API_KEY=<YOUR_API_KEY>
```

### Step 2: Running the `multi_frameworks_llm_router` Workflow

**Install the `multi_frameworks_llm_router` Workflow**

```bash
uv pip install -e examples/multi_frameworks_llm_router
```

**Run the `multi_frameworks_llm_router` Workflow**

note: the below is an example command to use and query this and trigger `rag_agent`

```bash
aiq run --config_file examples/multi_frameworks_llm_router/src/aiq_multi_frameworks_llm_router/configs/config.yml --input "What does this workflow demonstrate?"
```

**expected output:**

The output looks like the following. As you can see, the `chosen_worker_agent` was `Retriever`, which was routed to the rag tool to process.
```
2025-04-15 17:15:09,428 - aiq.runtime.loader - WARNING - Loading module 'aiq.tool.register' from entry point 'aiq_tools' took a long time (617.214680 ms). Ensure all imports are inside your registered functions.
2025-04-15 17:15:09,903 - aiq.runtime.loader - WARNING - Loading module 'aiq_automated_description_generation.register' from entry point 'aiq_automated_description_generation' took a long time (417.807341 ms). Ensure all imports are inside your registered functions.
2025-04-15 17:15:09,983 - aiq.cli.commands.start - INFO - Starting AgentIQ from config file: 'examples\multi_frameworks_llm_router\src\aiq_multi_frameworks_llm_router\configs\config.yml'
2025-04-15 17:15:09,994 - aiq.cli.commands.start - WARNING - The front end type in the config file (fastapi) does not match the command name (console). Overwriting the config file front end.
2025-04-15 17:15:10,063 - phoenix.config - INFO - 📋 Ensuring phoenix working directory: C:\Users\wglantz\.phoenix
2025-04-15 17:15:10,079 - phoenix.inferences.inferences - INFO - Dataset: phoenix_inferences_015f2115-f951-49f9-9061-0076491e8425 initialized
2025-04-15 17:15:36,318 - aiq.profiler.decorators - INFO - LlamaIndex callback handler registered
2025-04-15 17:15:37,320 - aiq_multi_frameworks_llm_router.llama_index_rag_tool - INFO - ##### processing data from ingesting files in this folder : ./examples/multi_frameworks_llm_router/README.md
None of PyTorch, TensorFlow >= 2.0, or Flax have been found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.
2025-04-15 17:16:26,562 - httpx - INFO - HTTP Request: POST https://integrate.api.nvidia.com/v1/embeddings "HTTP/1.1 200 OK"
2025-04-15 17:16:27,120 - haystack.tracing.tracer - INFO - Auto-enabled tracing for 'OpenTelemetryTracer'
2025-04-15 17:16:27,299 - aiq.profiler.decorators - INFO - Langchain callback handler registered
2025-04-15 17:16:27,299 - aiq_multi_frameworks_llm_router.register - INFO - workflow config = llm='nim_llm' llm_router_tool='llm_router_tool' data_dir='./examples/multi_frameworks_llm_router/README.md' rag_tool='llama_index_rag2' chitchat_agent='haystack_chitchat_agent2'

Configuration Summary:
--------------------
Workflow Type: multi_frameworks_llm_router
Number of Functions: 3
Number of LLMs: 2
Number of Embedders: 1
Number of Memory: 0
Number of Retrievers: 0

2025-04-15 17:16:27,435 - aiq.front_ends.console.console_front_end_plugin - INFO - Processing input: ('What does this workflow demonstrate?',)
2025-04-15 17:16:28,049 - aiq_multi_frameworks_llm_router.register - INFO - ========== inside **supervisor node**  current status =
 {'input': 'What does this workflow demonstrate?', 'chat_history': InMemoryChatMessageHistory(messages=[HumanMessage(content='What does this workflow demonstrate?', additional_kwargs={}, response_metadata={}), AIMessage(content='Retrieve', additional_kwargs={}, response_metadata={})])}
2025-04-15 17:16:28,049 - aiq_multi_frameworks_llm_router.register - INFO - ========== inside **router node**  current status =
 , ['input', 'chosen_worker_agent', 'chat_history']
2025-04-15 17:16:28,049 - aiq_multi_frameworks_llm_router.register - INFO -  ############# router to --> workers
2025-04-15 17:16:28,055 - aiq_multi_frameworks_llm_router.register - INFO - ========== inside **workers node**  current status =
 , {'input': 'What does this workflow demonstrate?', 'chat_history': InMemoryChatMessageHistory(messages=[HumanMessage(content='What does this workflow demonstrate?', additional_kwargs={}, response_metadata={}), AIMessage(content='Retrieve', additional_kwargs={}, response_metadata={})]), 'chosen_worker_agent': 'Retrieve'}
Added user message to memory: What does this workflow demonstrate?
2025-04-15 17:16:47,643 - httpx - INFO - HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"
=== Calling Function ===
Calling function: rag with args: {"input": "What does this workflow demonstrate?"}
2025-04-15 17:16:48,086 - httpx - INFO - HTTP Request: POST https://integrate.api.nvidia.com/v1/embeddings "HTTP/1.1 200 OK"
2025-04-15 17:16:50,914 - httpx - INFO - HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"
=== Function Output ===
This workflow demonstrates how to integrate multiple AI frameworks seamlessly using a set of LangChain / LangGraph agents, in AgentIQ, showcasing interoperability and flexibility by combining the strengths of different AI frameworks, such as Haystack Agent, LangChain Research Tool, and LlamaIndex RAG Tool, to build flexible AI pipelines that adapt to different use cases.
2025-04-15 17:16:56,781 - httpx - INFO - HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"
=== LLM Response ===
This workflow demonstrates how to integrate multiple AI frameworks seamlessly using a set of LangChain / LangGraph agents, in AgentIQ, showcasing interoperability and flexibility by combining the strengths of different AI frameworks, such as Haystack Agent, LangChain Research Tool, and LlamaIndex RAG Tool, to build flexible AI pipelines that adapt to different use cases.
2025-04-15 17:16:56,781 - aiq_multi_frameworks_llm_router.llama_index_rag_tool - INFO - response from llama-index Agent :
  This workflow demonstrates how to integrate multiple AI frameworks seamlessly using a set of LangChain / LangGraph agents, in AgentIQ, showcasing interoperability and flexibility by combining the strengths of different AI frameworks, such as Haystack Agent, LangChain Research Tool, and LlamaIndex RAG Tool, to build flexible AI pipelines that adapt to different use cases.
2025-04-15 17:16:56,781 - aiq_multi_frameworks_llm_router.register - INFO - **using rag_tool via llama_index_rag_agent >>> output:
 This workflow demonstrates how to integrate multiple AI frameworks seamlessly using a set of LangChain / LangGraph agents, in AgentIQ, showcasing interoperability and flexibility by combining the strengths of different AI frameworks, such as Haystack Agent, LangChain Research Tool, and LlamaIndex RAG Tool, to build flexible AI pipelines that adapt to different use cases.,
2025-04-15 17:16:56,781 - aiq_multi_frameworks_llm_router.register - INFO - final_output : This workflow demonstrates how to integrate multiple AI frameworks seamlessly using a set of LangChain / LangGraph agents, in AgentIQ, showcasing interoperability and flexibility by combining the strengths of different AI frameworks, such as Haystack Agent, LangChain Research Tool, and LlamaIndex RAG Tool, to build flexible AI pipelines that adapt to different use cases.
2025-04-15 17:16:56,781 - aiq.observability.async_otel_listener - INFO - Intermediate step stream completed. No more events will arrive.
2025-04-15 17:16:56,781 - aiq.front_ends.console.console_front_end_plugin - INFO - --------------------------------------------------
Workflow Result:
['This workflow demonstrates how to integrate multiple AI frameworks seamlessly using a set of LangChain / LangGraph agents, in AgentIQ, showcasing interoperability and flexibility by combining the strengths of different AI frameworks, such as Haystack Agent, LangChain Research Tool, and LlamaIndex RAG Tool, to build flexible AI pipelines that adapt to different use cases.']
--------------------------------------------------
```

Let's try one more query:
```
aiq run --config_file examples/multi_frameworks_llm_router/src/aiq_multi_frameworks_llm_router/configs/config.yml --input "What is Compound AI?"
```

Here is the output, the `chosen_worker_agent` was `Research`, and the call was sent to LLM Router, which classified the query as Open QA and handled it with `llama-3.1-70b-instruct`.

**expected output:**
```
2025-04-15 17:17:46,230 - aiq.runtime.loader - WARNING - Loading module 'aiq.tool.register' from entry point 'aiq_tools' took a long time (619.688034 ms). Ensure all imports are inside your registered functions.
2025-04-15 17:17:46,716 - aiq.runtime.loader - WARNING - Loading module 'aiq_automated_description_generation.register' from entry point 'aiq_automated_description_generation' took a long time (434.922934 ms). Ensure all imports are inside your registered functions.
2025-04-15 17:17:46,800 - aiq.cli.commands.start - INFO - Starting AgentIQ from config file: 'examples\multi_frameworks_llm_router\src\aiq_multi_frameworks_llm_router\configs\config.yml'
2025-04-15 17:17:46,809 - aiq.cli.commands.start - WARNING - The front end type in the config file (fastapi) does not match the command name (console). Overwriting the config file front end.
2025-04-15 17:17:46,885 - phoenix.config - INFO - 📋 Ensuring phoenix working directory: C:\Users\wglantz\.phoenix
2025-04-15 17:17:46,900 - phoenix.inferences.inferences - INFO - Dataset: phoenix_inferences_42f14c57-65a0-437b-9ef6-b3f85126d4b5 initialized
2025-04-15 17:17:52,689 - aiq.profiler.decorators - INFO - LlamaIndex callback handler registered
2025-04-15 17:17:52,976 - aiq_multi_frameworks_llm_router.llama_index_rag_tool - INFO - ##### processing data from ingesting files in this folder : ./examples/multi_frameworks_llm_router/README.md
None of PyTorch, TensorFlow >= 2.0, or Flax have been found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.
2025-04-15 17:18:01,745 - httpx - INFO - HTTP Request: POST https://integrate.api.nvidia.com/v1/embeddings "HTTP/1.1 200 OK"
2025-04-15 17:18:02,250 - haystack.tracing.tracer - INFO - Auto-enabled tracing for 'OpenTelemetryTracer'
2025-04-15 17:18:02,277 - aiq.profiler.decorators - INFO - Langchain callback handler registered
2025-04-15 17:18:02,277 - aiq_multi_frameworks_llm_router.register - INFO - workflow config = llm='nim_llm' llm_router_tool='llm_router_tool' data_dir='./examples/multi_frameworks_llm_router/README.md' rag_tool='llama_index_rag2' chitchat_agent='haystack_chitchat_agent2'

Configuration Summary:
--------------------
Workflow Type: multi_frameworks_llm_router
Number of Functions: 3
Number of LLMs: 2
Number of Embedders: 1
Number of Memory: 0
Number of Retrievers: 0

2025-04-15 17:18:02,363 - aiq.front_ends.console.console_front_end_plugin - INFO - Processing input: ('What is Compound AI?',)
2025-04-15 17:18:02,953 - aiq_multi_frameworks_llm_router.register - INFO - ========== inside **supervisor node**  current status =
 {'input': 'What is Compound AI?', 'chat_history': InMemoryChatMessageHistory(messages=[HumanMessage(content='What is Compound AI?', additional_kwargs={}, response_metadata={}), AIMessage(content='Research', additional_kwargs={}, response_metadata={})])}
2025-04-15 17:18:02,953 - aiq_multi_frameworks_llm_router.register - INFO - ========== inside **router node**  current status =
 , ['input', 'chosen_worker_agent', 'chat_history']
2025-04-15 17:18:02,953 - aiq_multi_frameworks_llm_router.register - INFO -  ############# router to --> workers
2025-04-15 17:18:02,953 - aiq_multi_frameworks_llm_router.register - INFO - ========== inside **workers node**  current status =
 , {'input': 'What is Compound AI?', 'chat_history': InMemoryChatMessageHistory(messages=[HumanMessage(content='What is Compound AI?', additional_kwargs={}, response_metadata={}), AIMessage(content='Research', additional_kwargs={}, response_metadata={})]), 'chosen_worker_agent': 'Research'}
2025-04-15 17:18:03,959 - aiq.tool.llm_router_tool - INFO - Sending request to LLM Router...
2025-04-15 17:18:12,806 - httpx - INFO - HTTP Request: POST http://10.176.221.83:8084/v1/chat/completions "HTTP/1.1 200 OK"
2025-04-15 17:18:12,819 - aiq.tool.llm_router_tool - INFO - Prompt was classified as 'Open QA' & handled by model: meta/llama-3.1-70b-instruct
2025-04-15 17:18:12,819 - aiq.tool.llm_router_tool - INFO - Token usage: {'input_tokens': 15, 'output_tokens': 513, 'total_tokens': 528}
2025-04-15 17:18:12,844 - aiq_multi_frameworks_llm_router.register - INFO - Successfully received response from LLM Router
2025-04-15 17:18:12,860 - aiq_multi_frameworks_llm_router.register - INFO - final_output : Compound AI is a relatively new term in the field of artificial intelligence (AI), and it refers to a more comprehensive approach to AI that incorporates multiple techniques and technologies to create a more robust and effective AI system.

The term "compound" in this context suggests a combination of different elements or techniques working together to produce a stronger, more powerful whole. In the case of Compound AI, this means integrating various AI techniques, such as machine learning, natural language processing, computer vision, and more, to create a more sophisticated and adaptable AI system.

There are a few key aspects that distinguish Compound AI from traditional AI approaches:

1. **Multimodal processing**: Compound AI systems can process and integrate multiple types of data, such as text, images, audio, and sensor data, to create a more comprehensive understanding of the world.
2. **Hybrid architectures**: Compound AI combines different AI techniques, such as symbolic AI (rule-based systems) and connectionist AI (deep learning models), to leverage the strengths of each approach.
3. **Cognitive architectures**: Compound AI systems are designed to mimic the structure and function of the human brain, incorporating cognitive frameworks that simulate human thought processes and decision-making.
4. **Continuous learning**: Compound AI systems are designed to learn continuously, adapting to new information, experiences, and contexts, and improving their performance over time.
5. **Explainability**: Compound AI systems aim to provide transparent and explainable decision-making processes, allowing humans to understand how the system arrives at its conclusions.

The benefits of Compound AI include:

1. **Improved accuracy**: By integrating multiple techniques and data sources, Compound AI can achieve higher accuracy and more robust performance.
2. **Increased adaptability**: Compound AI systems can adapt to changing environments, new data, and unexpected situations, making them more resilient and effective.
3. **Enhanced interpretability**: Compound AI's focus on explainability enables humans to understand and trust the decision-making processes, leading to greater confidence and adoption.

Some potential applications of Compound AI include:

1. **Healthcare**: Compound AI can be used for disease diagnosis, personalized medicine, and patient outcome prediction.
2. **Finance**: Compound AI can help with risk assessment, portfolio management, and financial forecasting.
3. **Security**: Compound AI can be applied to intrusion detection, threat analysis, and predictive policing.

In summary, Compound AI represents a new generation of AI systems that integrate multiple techniques and technologies to create more advanced, adaptable, and explainable AI models, with a wide range of potential applications and benefits.
2025-04-15 17:18:12,866 - aiq.observability.async_otel_listener - INFO - Intermediate step stream completed. No more events will arrive.
2025-04-15 17:18:12,866 - aiq.front_ends.console.console_front_end_plugin - INFO - --------------------------------------------------
Workflow Result:
['Compound AI is a relatively new term in the field of artificial intelligence (AI), and it refers to a more comprehensive approach to AI that incorporates multiple techniques and technologies to create a more robust and effective AI system.\n\nThe term "compound" in this context suggests a combination of different elements or techniques working together to produce a stronger, more powerful whole. In the case of Compound AI, this means integrating various AI techniques, such as machine learning, natural language processing, computer vision, and more, to create a more sophisticated and adaptable AI system.\n\nThere are a few key aspects that distinguish Compound AI from traditional AI approaches:\n\n1. **Multimodal processing**: Compound AI systems can process and integrate multiple types of data, such as text, images, audio, and sensor data, to create a more comprehensive understanding of the world.\n2. **Hybrid architectures**: Compound AI combines different AI techniques, such as symbolic AI (rule-based systems) and connectionist AI (deep learning models), to leverage the strengths of each approach.\n3. **Cognitive architectures**: Compound AI systems are designed to mimic the structure and function of the human brain, incorporating cognitive frameworks that simulate human thought processes and decision-making.\n4. **Continuous learning**: Compound AI systems are designed to learn continuously, adapting to new information, experiences, and contexts, and improving their performance over time.\n5. **Explainability**: Compound AI systems aim to provide transparent and explainable decision-making processes, allowing humans to understand how the system arrives at its conclusions.\n\nThe benefits of Compound AI include:\n\n1. **Improved accuracy**: By integrating multiple techniques and data sources, Compound AI can achieve higher accuracy and more robust performance.\n2. **Increased adaptability**: Compound AI systems can adapt to changing environments, new data, and unexpected situations, making them more resilient and effective.\n3. **Enhanced interpretability**: Compound AI\'s focus on explainability enables humans to understand and trust the decision-making processes, leading to greater confidence and adoption.\n\nSome potential applications of Compound AI include:\n\n1. **Healthcare**: Compound AI can be used for disease diagnosis, personalized medicine, and patient outcome prediction.\n2. **Finance**: Compound AI can help with risk assessment, portfolio management, and financial forecasting.\n3. **Security**: Compound AI can be applied to intrusion detection, threat analysis, and predictive policing.\n\nIn summary, Compound AI represents a new generation of AI systems that integrate multiple techniques and technologies to create more advanced, adaptable, and explainable AI models, with a wide range of potential applications and benefits.']
--------------------------------------------------
```
