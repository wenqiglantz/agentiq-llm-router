# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import typing
from abc import ABC
from abc import abstractmethod

from pydantic_core import CoreSchema
from pydantic_core import core_schema

from aiq.data_models.common import HashableBaseModel
from aiq.data_models.component import ComponentGroup


def generate_instance_id(input_object: typing.Any) -> str:
    """Generates a unique identifier for a python object derived from its python unique id.

    Args:
        input_object (typing.Any): The input object to receive a unique identifier.

    Returns:
        str: Unique identifier.
    """

    return str(id(input_object))


class ComponentRefNode(HashableBaseModel):
    """A node type for component runtime instances reference names in a networkx digraph.

    Args:
        ref_name (ComponentRef): The name of the component runtime instance.
        component_group (ComponentGroup): The component group in an AgentIQ configuration object.
    """

    ref_name: "ComponentRef"
    component_group: ComponentGroup


class ComponentRef(str, ABC):
    """
    Abstract class used for the interface to derive ComponentRef objects.
    """

    def __new__(cls, value: "ComponentRef | str"):
        # Sublcassing str skips abstractmethod enforcement.
        if len(cls.__abstractmethods__ - set(cls.__dict__)):
            abstract_methods = ", ".join([f"'{method}'" for method in cls.__abstractmethods__])
            raise TypeError(f"Can't instantiate abstract class {cls.__name__} "
                            f"without an implementation for abstract method(s) {abstract_methods}")

        return super().__new__(cls, value)

    @property
    @abstractmethod
    def component_group(self) -> ComponentGroup:
        """Provides the component group this ComponentRef object represents.

        Returns:
            ComponentGroup: A component group of the AgentIQ configuration object
        """

        pass

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler, **kwargs) -> CoreSchema:
        return core_schema.no_info_plain_validator_function(cls)


class EmbedderRef(ComponentRef):
    """
    A reference to an embedder in an AgentIQ configuration object.
    """

    @property
    @typing.override
    def component_group(self):
        return ComponentGroup.EMBEDDERS


class FunctionRef(ComponentRef):
    """
    A reference to a function in an AgentIQ configuration object.
    """

    @property
    @typing.override
    def component_group(self):
        return ComponentGroup.FUNCTIONS


class LLMRef(ComponentRef):
    """
    A reference to an LLM in an AgentIQ configuration object.
    """

    @property
    @typing.override
    def component_group(self):
        return ComponentGroup.LLMS


class MemoryRef(ComponentRef):
    """
    A reference to a memory in an AgentIQ configuration object.
    """

    @property
    @typing.override
    def component_group(self):
        return ComponentGroup.MEMORY


class RetrieverRef(ComponentRef):
    """
    A reference to a retriever in an AgentIQ configuration object.
    """

    @property
    @typing.override
    def component_group(self):
        return ComponentGroup.RETRIEVERS


class LLMRouter(ComponentRef):
    """
    A reference to an LLM Router in an AgentIQ configuration object.
    This includes the necessary configuration for routing LLM requests.
    
    Args:
        value (str): The router name or identifier.
        api_key (str): The API key for the LLM router.
        base_url (str): The base URL for the LLM router.
        policy (str): The policy for routing LLM requests.
        routing_strategy (str): The strategy for routing LLM requests.
    """
    
    api_key: str
    base_url: str
    policy: str
    routing_strategy: str
    
    def __new__(cls, value: str, api_key: str, base_url: str, policy: str, routing_strategy: str):
        # Create the string instance
        instance = super().__new__(cls, value)
        
        # Set the attributes for the LLM router
        instance.api_key = api_key
        instance.base_url = base_url
        instance.policy = policy
        instance.routing_strategy = routing_strategy
        
        return instance
    
    @property
    @typing.override
    def component_group(self):
        return ComponentGroup.LLMS  # Using LLMS as the component group for now
