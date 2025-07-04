"""
MCP (Model Context Protocol) Manager for LLM Kernel

This module handles MCP server connections and tool execution,
allowing the LLM to interact with external tools and resources.
"""

import os
import json
import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

try:
    from fastmcp import Client
except ImportError:
    Client = None


class MCPManager:
    """Manages MCP server connections and tool execution."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize MCP Manager."""
        self.logger = logger or logging.getLogger(__name__)
        self.servers: Dict[str, Client] = {}
        self.available_tools: Dict[str, Dict[str, Any]] = {}
        self.server_configs: Dict[str, Dict[str, Any]] = {}
        
        if Client is None:
            self.logger.warning("fastmcp not installed. MCP features will be unavailable.")
    
    def load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load MCP server configuration from file or environment."""
        config = {}
        
        # Try to load from file
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            # Try common config locations
            config_locations = [
                Path.home() / '.llm-kernel' / 'mcp-config.json',
                Path.cwd() / 'mcp-config.json',
                Path.cwd() / '.mcp' / 'config.json',
            ]
            
            for location in config_locations:
                if location.exists():
                    with open(location, 'r') as f:
                        config = json.load(f)
                    self.logger.info(f"Loaded MCP config from {location}")
                    break
        
        # Override with environment variables
        env_servers = os.getenv('LLM_KERNEL_MCP_SERVERS')
        if env_servers:
            try:
                env_config = json.loads(env_servers)
                config['mcpServers'] = {**config.get('mcpServers', {}), **env_config}
            except json.JSONDecodeError:
                self.logger.error("Invalid JSON in LLM_KERNEL_MCP_SERVERS environment variable")
        
        self.server_configs = config.get('mcpServers', {})
        return config
    
    async def connect_server(self, name: str, config: Union[str, Dict[str, Any]]) -> bool:
        """Connect to an MCP server."""
        if Client is None:
            self.logger.error("fastmcp not installed. Cannot connect to MCP servers.")
            return False
        
        try:
            # Create client based on config type
            if isinstance(config, str):
                # Simple URL or path
                client = Client(config)
            elif isinstance(config, dict):
                # Complex configuration
                if 'command' in config:
                    # Stdio server (like filesystem)
                    import subprocess
                    cmd = [config['command']] + config.get('args', [])
                    env = {**os.environ, **config.get('env', {})}
                    
                    # For now, we'll use the URL if provided
                    # TODO: Implement proper stdio server support
                    if 'url' in config:
                        client = Client(config['url'])
                    else:
                        self.logger.warning(f"Stdio servers not yet fully supported. Skipping {name}")
                        return False
                elif 'url' in config:
                    # HTTP server
                    client = Client(config['url'])
                else:
                    self.logger.error(f"Invalid config for server {name}")
                    return False
            else:
                self.logger.error(f"Invalid config type for server {name}")
                return False
            
            # Connect to the server
            await client.__aenter__()
            self.servers[name] = client
            
            # Discover available tools
            tools = await client.list_tools()
            for tool in tools:
                tool_name = f"{name}.{tool['name']}"
                self.available_tools[tool_name] = {
                    'server': name,
                    'original_name': tool['name'],
                    'description': tool.get('description', ''),
                    'inputSchema': tool.get('inputSchema', {})
                }
            
            self.logger.info(f"Connected to MCP server '{name}' with {len(tools)} tools")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to MCP server '{name}': {e}")
            return False
    
    async def disconnect_server(self, name: str) -> bool:
        """Disconnect from an MCP server."""
        if name not in self.servers:
            return False
        
        try:
            client = self.servers[name]
            await client.__aexit__(None, None, None)
            del self.servers[name]
            
            # Remove tools from this server
            self.available_tools = {
                k: v for k, v in self.available_tools.items()
                if v['server'] != name
            }
            
            self.logger.info(f"Disconnected from MCP server '{name}'")
            return True
            
        except Exception as e:
            self.logger.error(f"Error disconnecting from server '{name}': {e}")
            return False
    
    async def connect_all_servers(self) -> Dict[str, bool]:
        """Connect to all configured MCP servers."""
        results = {}
        
        for name, config in self.server_configs.items():
            results[name] = await self.connect_server(name, config)
        
        return results
    
    async def disconnect_all_servers(self):
        """Disconnect from all MCP servers."""
        server_names = list(self.servers.keys())
        for name in server_names:
            await self.disconnect_server(name)
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call an MCP tool."""
        if tool_name not in self.available_tools:
            raise ValueError(f"Tool '{tool_name}' not found")
        
        tool_info = self.available_tools[tool_name]
        server_name = tool_info['server']
        original_name = tool_info['original_name']
        
        if server_name not in self.servers:
            raise RuntimeError(f"Server '{server_name}' not connected")
        
        client = self.servers[server_name]
        
        try:
            result = await client.call_tool(original_name, arguments)
            return result
        except Exception as e:
            self.logger.error(f"Error calling tool '{tool_name}': {e}")
            raise
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools from connected servers."""
        tools = []
        for tool_name, tool_info in self.available_tools.items():
            tools.append({
                'name': tool_name,
                'server': tool_info['server'],
                'description': tool_info['description'],
                'inputSchema': tool_info['inputSchema']
            })
        return tools
    
    def get_tool_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get the schema for a specific tool."""
        if tool_name in self.available_tools:
            return self.available_tools[tool_name]['inputSchema']
        return None
    
    def is_connected(self) -> bool:
        """Check if any MCP servers are connected."""
        return len(self.servers) > 0
    
    def get_connected_servers(self) -> List[str]:
        """Get list of connected server names."""
        return list(self.servers.keys())