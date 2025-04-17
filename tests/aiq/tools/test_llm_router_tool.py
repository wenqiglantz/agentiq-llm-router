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
from unittest import mock

import openai  # Import openai for APIConnectionError
import pytest
from openai.types.chat import ChatCompletion
from openai.types.chat import ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.data_models.component_ref import LLMRef
from aiq.tool.llm_router_tool import LLMRouterToolConfig
from aiq.tool.llm_router_tool import llm_router_tool


class MockRawResponse:
    """Mock for the raw response from OpenAI."""

    def __init__(self, headers: dict, completion: ChatCompletion):
        self.headers = headers
        self._completion = completion

    def parse(self):
        return self._completion


@pytest.fixture
def mock_openai_client():
    """Fixture to mock the OpenAI client."""
    with mock.patch('aiq.tool.llm_router_tool.OpenAI') as mock_client:
        instance = mock_client.return_value

        # Set up the mock structure manually
        mock_completions = mock.MagicMock()
        mock_with_raw_response = mock.MagicMock()
        mock_create = mock.MagicMock()

        instance.chat = mock.MagicMock()
        instance.chat.completions = mock_completions
        instance.chat.completions.with_raw_response = mock_with_raw_response
        instance.chat.completions.with_raw_response.create = mock_create

        # Set the return value to None, it will be overridden in the tests
        instance.chat.completions.with_raw_response.create.return_value = None

        yield instance


@pytest.fixture
def mock_chat_completion():
    """Fixture to create a mock ChatCompletion object."""
    message = ChatCompletionMessage(content="This is a test response", role="assistant", usage_metadata=None)

    usage = CompletionUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)

    completion = ChatCompletion(id="test-id",
                                choices=[{
                                    "index": 0, "message": message, "finish_reason": "stop"
                                }],
                                created=1234567890,
                                model="test-model",
                                object="chat.completion",
                                usage=usage)

    return completion


@pytest.fixture
def mock_builder():
    """Fixture to mock the AIQ builder."""
    builder = mock.MagicMock(spec=Builder)
    llm_config = mock.MagicMock()
    llm_config.api_key = "test-api-key"
    llm_config.base_url = "https://test-router-url.com"
    llm_config.policy = "test-policy"
    llm_config.routing_strategy = "test-strategy"

    builder.get_llm_config.return_value = llm_config
    return builder


@pytest.mark.parametrize("config_values", [
    {
        "llm_name": "test_llm",
    },
], ids=[
    "basic_config",
])
def test_llm_router_tool_config(config_values: dict[str, typing.Any]):
    """
    Test the LLMRouterToolConfig class.
    """
    LLMRouterToolConfig.model_validate(config_values, strict=True)
    config = LLMRouterToolConfig(**config_values)

    model_dump = config.model_dump()
    model_dump.pop('type')

    LLMRouterToolConfig.model_validate(model_dump, strict=True)

    # Validate the config has the expected field
    assert isinstance(config.llm_name, LLMRef)
    assert config.llm_name == "test_llm"


@pytest.mark.asyncio
async def test_llm_router_tool_successful_call(mock_openai_client, mock_chat_completion, mock_builder):
    """
    Test that the LLM router tool correctly calls the OpenAI API with the expected parameters
    and processes the response appropriately.
    """
    # Setup the mock response
    headers = {"X-Chosen-Classifier": "test-classifier"}
    mock_raw_response = MockRawResponse(headers=headers, completion=mock_chat_completion)
    mock_openai_client.chat.completions.with_raw_response.create.return_value = mock_raw_response

    # Create the tool config
    config = LLMRouterToolConfig(llm_name="test_llm")

    # Get the tool function
    async with llm_router_tool(config, mock_builder) as tool_info:
        assert isinstance(tool_info, FunctionInfo)

        # Call the tool function
        router_fn = tool_info.single_fn
        result = await router_fn("What is the meaning of life?")

        # Verify the OpenAI client was called with the correct parameters
        mock_openai_client.chat.completions.with_raw_response.create.assert_called_once()
        call_args = mock_openai_client.chat.completions.with_raw_response.create.call_args[1]

        assert call_args["messages"] == [{"role": "user", "content": "What is the meaning of life?"}]
        assert call_args["model"] == ""  # Model is determined by the router
        assert call_args["extra_body"] == {
            "nim-llm-router": {
                "policy": "test-policy",
                "routing_strategy": "test-strategy",
            }
        }

        # Verify the result is as expected
        assert result == mock_chat_completion

        # Verify the usage metadata was processed
        assert hasattr(result.choices[0].message, "usage_metadata")
        assert result.choices[0].message.usage_metadata == {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}


@pytest.mark.skip(reason="API compatibility issue with openai 1.66.2: APIConnectionError constructor has changed")
@pytest.mark.asyncio
async def test_llm_router_tool_error_handling(mock_openai_client, mock_builder):
    """
    Test that the LLM router tool correctly handles errors from the OpenAI API.
    """
    # Setup the mock to raise an exception
    mock_error = openai.APIConnectionError("Connection error", request=mock.MagicMock())
    mock_openai_client.chat.completions.with_raw_response.create.side_effect = mock_error

    # Create the tool config
    config = LLMRouterToolConfig(llm_name="test_llm")

    # Get the tool function
    async with llm_router_tool(config, mock_builder) as tool_info:
        assert isinstance(tool_info, FunctionInfo)

        # Call the tool function and expect it to raise the same exception
        router_fn = tool_info.single_fn
        with pytest.raises(openai.APIConnectionError) as excinfo:
            await router_fn("What is the meaning of life?")

        assert str(excinfo.value) == "Connection error"


@pytest.mark.asyncio
async def test_llm_router_tool_no_usage_info(mock_openai_client, mock_chat_completion, mock_builder):
    """
    Test that the LLM router tool correctly handles responses with no usage information.
    """
    # Create a completion with no usage info
    completion_no_usage = mock_chat_completion
    completion_no_usage.usage = None

    # Setup the mock response
    headers = {"X-Chosen-Classifier": "test-classifier"}
    mock_raw_response = MockRawResponse(headers=headers, completion=completion_no_usage)
    mock_openai_client.chat.completions.with_raw_response.create.return_value = mock_raw_response

    # Create the tool config
    config = LLMRouterToolConfig(llm_name="test_llm")

    # Get the tool function
    async with llm_router_tool(config, mock_builder) as tool_info:
        assert isinstance(tool_info, FunctionInfo)

        # Call the tool function
        router_fn = tool_info.single_fn
        result = await router_fn("What is the meaning of life?")

        # Verify the result is as expected
        assert result == completion_no_usage

        # Since there's no usage info, we expect a warning but no error
        # (This is logged, not returned, so we can't easily check it)
