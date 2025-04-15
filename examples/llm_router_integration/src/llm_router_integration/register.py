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

import logging
import traceback
from typing import Any

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.cli.register_workflow import register_function
from aiq.data_models.component_ref import FunctionRef
from aiq.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class LLMRouterIntegrationConfig(FunctionBaseConfig, name="llm_router_integration"):
    """Configuration for the LLM Router Integration tool.
    
    Attributes:
        llm: Reference to the LLM model to use for routing
    """
    llm_router_tool: FunctionRef


@register_function(config_type=LLMRouterIntegrationConfig)
async def llm_router_integration(config: LLMRouterIntegrationConfig, builder: Builder) -> Any:
    """Register the LLM Router Integration tool.
    
    Args:
        config: Configuration for the LLM Router Integration
        builder: The AgentIQ builder instance
        
    Returns:
        FunctionInfo: The registered function information
        
    Yields:
        FunctionInfo: Tool registration information
    """

    llm_router_tool = builder.get_tool(fn_name=config.llm_router_tool, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    async def _llm_router_integration(text: str) -> str:
        """
        Integrates with LLM Router to process user queries.

        This function sends the user's text query to the LLM Router service 
        configured in the tool settings and returns the LLM's response.

        Args:
            text: The user query to be processed by the LLM Router

        Returns:
            str: Response from LLM Router, or error message if an error occurred
        """
        if not text or not text.strip():
            logger.warning("Empty query received, cannot process empty text")
            return "Error: Empty query received. Please provide a valid query."
            
        try:
            logger.info(f"Sending query to LLM Router: {text[:50]}{'...' if len(text) > 50 else ''}")
            completion = (await llm_router_tool.ainvoke(text))
            response = completion.choices[0].message.content
            logger.info("Successfully received response from LLM Router")
            return response if response is not None else "No response received from LLM Router."
                
        except Exception as e:
            error_msg = f"Error processing query through LLM Router: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Traceback: {traceback.format_exc()}")
            return f"An error occurred while processing your request: {str(e)}"

    # Create a Generic AgentIQ tool
    yield FunctionInfo.from_fn(
        _llm_router_integration,
        description="This tool integrates with LLM Router to process natural language queries"
    )
