# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
MCP Integration Library for Medical Report Processing

This module provides reusable components for integrating Model Context Protocol (MCP)
with pydantic-ai agents.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s %(message)s")
LOGGER = logging.getLogger(__name__)

# Maximum characters for tool results to prevent context explosion
MAX_TOOL_RESULT_CHARS = 50000  # ~12.5k tokens


class MCPToolProvider:
    """
    Manages connections to MCP servers and provides tool wrappers for pydantic-ai.

    Thread-safety: Uses an asyncio.Lock to protect concurrent access to MCP sessions.
    """

    def __init__(self, max_result_chars: int = MAX_TOOL_RESULT_CHARS):
        self.servers: Dict[str, ClientSession] = {}
        self.contexts = {}
        self._lock = asyncio.Lock()
        self._tool_wrappers_cache: Optional[List] = None
        self.max_result_chars = max_result_chars

    async def connect_server(self, name: str, server_params: StdioServerParameters):
        """Connect to an MCP server."""
        LOGGER.info(f"Connecting to MCP server: {name}")

        # stdio_client returns an async context manager
        # We need to enter it and keep it alive
        context_manager = stdio_client(server_params)
        read_stream, write_stream = await context_manager.__aenter__()

        # Store the context manager to close it later
        self.contexts[name] = context_manager

        session = ClientSession(read_stream, write_stream)
        await session.__aenter__()

        # Initialize the session
        await session.initialize()

        self.servers[name] = session
        LOGGER.info(f"Connected to {name}")

    async def call_tool(
        self, server_name: str, tool_name: str, arguments: Dict[str, Any]
    ) -> Any:
        """Call a tool on a specific MCP server.

        Uses a lock to prevent concurrent access issues with the MCP session.
        """
        if server_name not in self.servers:
            raise ValueError(f"Server {server_name} not connected")

        async with self._lock:
            session = self.servers[server_name]
            result = await session.call_tool(tool_name, arguments)
            return result

    def _truncate_result(self, result_text: str, tool_name: str) -> str:
        """Truncate tool result if it exceeds the maximum size."""
        if len(result_text) <= self.max_result_chars:
            return result_text

        truncated = result_text[: self.max_result_chars]
        truncated += f"\n\n[... RESULT TRUNCATED: {len(result_text)} chars -> {self.max_result_chars} chars ...]"
        LOGGER.warning(
            f"Tool {tool_name} result truncated: {len(result_text)} -> {len(truncated)} chars"
        )
        return truncated

    def create_tool_wrapper(self, server_name: str, tool_info: Any):
        """Create a pydantic-ai compatible tool wrapper for an MCP tool.

        The wrapper includes result size limiting to prevent context explosion.
        """
        tool_name = tool_info.name

        async def tool_func(**kwargs) -> str:
            """Wrapped MCP tool function."""
            try:
                result = await self.call_tool(server_name, tool_name, kwargs)

                # Extract content from MCP result
                result_text = ""
                if hasattr(result, "content"):
                    content_items = result.content
                    if isinstance(content_items, list):
                        text_parts = []
                        for item in content_items:
                            if hasattr(item, "text"):
                                text_parts.append(item.text)
                        result_text = (
                            "\n".join(text_parts) if text_parts else str(result)
                        )
                    else:
                        result_text = str(content_items)
                else:
                    result_text = str(result)

                # Truncate if too large
                return self._truncate_result(result_text, tool_name)

            except Exception as e:
                LOGGER.exception(f"Error calling tool {tool_name}")
                return f"Error: {str(e)}"

        # Set function metadata for pydantic-ai
        tool_func.__name__ = tool_name
        if hasattr(tool_info, "description") and tool_info.description:
            tool_func.__doc__ = tool_info.description

        return tool_func

    async def get_tool_wrappers(self, force_refresh: bool = False) -> List:
        """Get all available tools as pydantic-ai compatible wrappers.

        Tool wrappers are cached after first creation to avoid recreating them
        for each request. Use force_refresh=True to recreate the cache.
        """
        if self._tool_wrappers_cache is not None and not force_refresh:
            return self._tool_wrappers_cache

        tools = []

        async with self._lock:
            for server_name, session in self.servers.items():
                try:
                    # List tools from the server using the MCP protocol
                    tools_result = await session.list_tools()

                    if hasattr(tools_result, "tools"):
                        server_tools = tools_result.tools
                        for tool_info in server_tools:
                            wrapper = self.create_tool_wrapper(server_name, tool_info)
                            tools.append(wrapper)
                            LOGGER.info(
                                f"Registered tool: {tool_info.name} from {server_name}"
                            )
                except Exception:
                    LOGGER.exception(f"Error getting tools from {server_name}")

        self._tool_wrappers_cache = tools
        return tools

    async def close(self):
        """Close all MCP server connections."""
        # Close sessions first
        for name, session in self.servers.items():
            try:
                await session.__aexit__(None, None, None)
                LOGGER.info(f"Closed session for {name}")
            except Exception:
                LOGGER.exception(f"Error closing session {name}")

        # Close stdio context managers
        for name, context_manager in self.contexts.items():
            try:
                await context_manager.__aexit__(None, None, None)
                LOGGER.info(f"Closed connection to {name}")
            except Exception:
                LOGGER.exception(f"Error closing context for {name}")


def load_system_prompt(prompt_file: Path) -> str:
    """Load system prompt from file."""
    if not prompt_file.exists():
        raise FileNotFoundError(f"System prompt file not found: {prompt_file}")
    return prompt_file.read_text(encoding="utf-8").strip()


def load_medical_dictionary(dict_file: Path) -> str:
    """Load medical dictionary from file."""
    if not dict_file.exists():
        raise FileNotFoundError(f"Medical dictionary file not found: {dict_file}")
    return dict_file.read_text(encoding="utf-8").strip()


def load_system_prompt_with_dictionary(prompt_file: Path, dict_file: Path) -> str:
    """Load system prompt and merge medical dictionary into it."""
    prompt = load_system_prompt(prompt_file)
    dictionary = load_medical_dictionary(dict_file)
    return prompt.replace("{{MEDICAL_DICTIONARY}}", dictionary)


def load_mcp_config(config_path: Path) -> Dict[str, Any]:
    """Load MCP server configuration from JSON file."""
    if not config_path.exists():
        raise FileNotFoundError(f"MCP config not found: {config_path}")

    return json.loads(config_path.read_text(encoding="utf-8"))
