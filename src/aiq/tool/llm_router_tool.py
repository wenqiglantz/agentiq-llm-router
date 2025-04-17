# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Tool implementation for routing LLM queries through the NVIDIA LLM Router.
Provides functionality to dynamically route queries to appropriate LLM backends
based on configured policies and routing strategies.
"""

import logging

from openai import OpenAI
from openai.types.chat import ChatCompletion

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.component_ref import LLMRef
from aiq.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class LLMRouterToolConfig(FunctionBaseConfig, name="llm_router_tool"):
    """
    Configuration for the LLM Router Tool.

    This tool calls the NVIDIA LLM Router to route queries to the appropriate LLM
    based on configured routing policies and strategies.

    Attributes:
        llm_name (LLMRef): Reference to the LLM configuration for the router
    """
    llm_name: LLMRef


@register_function(config_type=LLMRouterToolConfig)
async def llm_router_tool(config: LLMRouterToolConfig, builder: Builder):
    """
    Function implementation for the LLM Router Tool.

    Args:
        config (LLMRouterToolConfig): The tool configuration
        builder (Builder): The AIQ builder instance

    Yields:
        FunctionInfo: Function information for the LLM router call handler
    """
    llm_config = builder.get_llm_config(config.llm_name)

    async def _call_llm_router(query: str) -> ChatCompletion:
        """
        Make a request to an LLM through the LLM router.

        Args:
            query (str): The user query to be routed to the appropriate LLM

        Returns:
            ChatCompletion: The completion response from the selected LLM

        Raises:
            Exception: If the LLM router request fails
        """
        # Initialize OpenAI client with router configuration
        client = OpenAI(
            api_key=llm_config.api_key,
            max_retries=0,
            base_url=llm_config.base_url,  # URL for the router-controller
            default_headers={
                "Content-Type": "application/json", "Authorization": f"Bearer {llm_config.api_key}"
            })

        # Prepare message payload
        messages = [{"role": "user", "content": query}]

        # Set router-specific parameters
        extra_body = {
            "nim-llm-router": {
                "policy": llm_config.policy,
                "routing_strategy": llm_config.routing_strategy,
            }
        }

        logger.info("Sending request to LLM Router...")
        try:
            # Send request with raw response to access headers
            chat_completion = client.chat.completions.with_raw_response.create(
                messages=messages,
                model="",  # Model is determined by the router
                extra_body=extra_body)
        except Exception as e:
            logger.error("Error in LLM Router request:")
            if hasattr(e, 'response'):
                logger.error(f"Status Code: {e.response.status_code}")
                logger.error(f"Error Message: {e.response.text}")
            else:
                logger.error(f"Error: {str(e)}")
            raise e

        # Get the classifier information from headers
        chosen_classifier = chat_completion.headers.get('X-Chosen-Classifier', 'unknown')

        # Parse the completion
        completion = chat_completion.parse()

        logger.info(f"Prompt was classified as '{chosen_classifier}' & handled by model: {completion.model}")

        # Process token usage information
        _process_token_usage(completion)

        return completion

    def _process_token_usage(completion: ChatCompletion) -> None:
        """
        Process token usage from the completion response.

        This ensures the langchain callback handler can access token usage information.

        Args:
            completion (ChatCompletion): The completion response from the LLM
        """
        if hasattr(completion, 'usage') and completion.usage:
            # Map OpenAI usage field names to the names expected by the callback handler
            usage_metadata = {
                "input_tokens": completion.usage.prompt_tokens,
                "output_tokens": completion.usage.completion_tokens,
                "total_tokens": completion.usage.total_tokens
            }

            # Add these attributes to the message so LangChain callbacks can find them
            if hasattr(completion.choices[0].message, 'usage_metadata'):
                completion.choices[0].message.usage_metadata = usage_metadata
            else:
                # If usage_metadata doesn't exist, manually add it
                setattr(completion.choices[0].message, 'usage_metadata', usage_metadata)

            logger.info(f"Token usage: {usage_metadata}")
        else:
            logger.warning("No token usage information found in the response")

    # Return the function info
    yield FunctionInfo.from_fn(_call_llm_router,
                               description="Routes queries to the appropriate LLM using the NVIDIA LLM Router.")
