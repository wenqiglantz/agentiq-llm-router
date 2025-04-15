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

from pydantic import ConfigDict
from pydantic import Field

from aiq.builder.builder import Builder
from aiq.builder.llm import LLMProviderInfo
from aiq.cli.register_workflow import register_llm_provider
from aiq.data_models.llm import LLMBaseConfig


class LLMRouterConfig(LLMBaseConfig, name="llm_router"):
    """
    An LLM Router provider to be used with an LLM client.
    
    This provider routes requests to different LLM backends based on configurable 
    policies and strategies.
    """

    model_config = ConfigDict(protected_namespaces=())

    api_key: str | None = Field(
        default=None, 
        description="API key to interact with the LLM router service."
    )
    base_url: str | None = Field(
        default=None, 
        description="Base URL of the LLM router service."
    )
    policy: str = Field(
        default="task_router", 
        description="Policy to use for LLM routing (e.g., 'task_router', 'cost_based')."
    )
    routing_strategy: str = Field(
        default="triton", 
        description="Backend routing strategy implementation (e.g., 'triton', 'direct')."
    )


@register_llm_provider(config_type=LLMRouterConfig)
async def llm_router(llm_config: LLMRouterConfig, builder: Builder):
    """
    Register an LLM Router provider with the given configuration.
    
    Args:
        llm_config: Configuration for the LLM Router
        builder: Builder instance for configuration
        
    Yields:
        LLMProviderInfo: Information about the registered LLM provider
    """
    
    yield LLMProviderInfo(config=llm_config, description="A LLM Router for use with an LLM client.")
