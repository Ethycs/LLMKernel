"""
MCP (Model Context Protocol) Magic Commands for LLM Kernel

Contains magic commands for MCP server management and tool execution.
"""

import json
import asyncio
import concurrent.futures
from IPython.core.magic import Magics, line_magic, cell_magic, magics_class
from IPython.display import display, HTML

# Check if MCP is available
try:
    from ..mcp_manager import MCPManager
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False


@magics_class
class MCPMagics(Magics):
    """MCP-related magic commands."""
    
    def __init__(self, shell, kernel_instance):
        super().__init__(shell)
        self.kernel = kernel_instance
    
    @line_magic
    def llm_mcp_connect(self, line):
        """Connect to MCP servers.
        
        Usage:
            %llm_mcp_connect                    # Connect to all configured servers
            %llm_mcp_connect filesystem         # Connect to specific server
            %llm_mcp_connect https://example.com/mcp  # Connect to URL
        """
        if not MCP_AVAILABLE or not hasattr(self.kernel, 'mcp_manager') or self.kernel.mcp_manager is None:
            print("❌ MCP features are not available. Install fastmcp: pip install fastmcp")
            return
            
        async def connect():
            if not line.strip():
                # Connect to all configured servers
                results = await self.kernel.mcp_manager.connect_all_servers()
                
                if not results:
                    print("❌ No MCP servers configured")
                    print("💡 Create a config file or use: %llm_mcp_connect <server_url>")
                    return
                
                print("🔌 MCP Server Connections:")
                for name, success in results.items():
                    status = "✅ Connected" if success else "❌ Failed"
                    print(f"   {name}: {status}")
                
                # Show available tools
                tools = self.kernel.mcp_manager.list_tools()
                if tools:
                    print(f"\n🛠️  {len(tools)} tools available")
                    print("   Use %llm_mcp_tools to list them")
            else:
                # Connect to specific server
                arg = line.strip()
                
                # Check if it's a configured server name
                if arg in self.kernel.mcp_manager.server_configs:
                    success = await self.kernel.mcp_manager.connect_server(
                        arg, self.kernel.mcp_manager.server_configs[arg]
                    )
                else:
                    # Treat as URL or path
                    server_name = arg.split('/')[-1].replace('.', '_')
                    success = await self.kernel.mcp_manager.connect_server(server_name, arg)
                
                if success:
                    print(f"✅ Connected to {arg}")
                    tools = [t for t in self.kernel.mcp_manager.list_tools() 
                            if t['server'] == (arg if arg in self.kernel.mcp_manager.server_configs else server_name)]
                    print(f"🛠️  {len(tools)} tools available")
                else:
                    print(f"❌ Failed to connect to {arg}")
        
        # Run async function
        self._run_async(connect())
    
    @line_magic
    def llm_mcp_disconnect(self, line):
        """Disconnect from MCP servers.
        
        Usage:
            %llm_mcp_disconnect              # Disconnect all
            %llm_mcp_disconnect filesystem   # Disconnect specific server
        """
        async def disconnect():
            if not line.strip():
                # Disconnect all
                await self.kernel.mcp_manager.disconnect_all_servers()
                print("🔌 Disconnected from all MCP servers")
            else:
                # Disconnect specific server
                server_name = line.strip()
                success = await self.kernel.mcp_manager.disconnect_server(server_name)
                if success:
                    print(f"🔌 Disconnected from {server_name}")
                else:
                    print(f"❌ Server '{server_name}' not connected")
        
        self._run_async(disconnect())
    
    @line_magic
    def llm_mcp_tools(self, line):
        """List available MCP tools.
        
        Usage:
            %llm_mcp_tools              # List all tools
            %llm_mcp_tools filesystem   # List tools from specific server
            %llm_mcp_tools --json       # Output as JSON
        """
        tools = self.kernel.mcp_manager.list_tools()
        
        if not tools:
            print("❌ No MCP tools available")
            print("💡 Connect to servers first: %llm_mcp_connect")
            return
        
        # Filter by server if specified
        server_filter = None
        output_json = False
        
        args = line.strip().split()
        for arg in args:
            if arg == '--json':
                output_json = True
            elif not arg.startswith('--'):
                server_filter = arg
        
        if server_filter:
            tools = [t for t in tools if t['server'] == server_filter]
        
        if output_json:
            print(json.dumps(tools, indent=2))
        else:
            # Group by server
            by_server = {}
            for tool in tools:
                server = tool['server']
                if server not in by_server:
                    by_server[server] = []
                by_server[server].append(tool)
            
            print(f"🛠️  Available MCP Tools ({len(tools)} total):")
            for server, server_tools in by_server.items():
                print(f"\n📦 {server} ({len(server_tools)} tools):")
                for tool in server_tools:
                    print(f"   • {tool['name']}")
                    if tool['description']:
                        print(f"     {tool['description']}")
    
    @line_magic
    def llm_mcp_call(self, line):
        """Call an MCP tool directly.
        
        Usage:
            %llm_mcp_call tool_name {"arg": "value"}
            %llm_mcp_call filesystem.read_file {"path": "/etc/hosts"}
        """
        parts = line.strip().split(None, 1)
        if len(parts) < 2:
            print("Usage: %llm_mcp_call <tool_name> <json_arguments>")
            return
        
        tool_name = parts[0]
        
        try:
            arguments = json.loads(parts[1])
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON arguments: {e}")
            return
        
        async def call_tool():
            try:
                result = await self.kernel.mcp_manager.call_tool(tool_name, arguments)
                print(f"✅ Tool '{tool_name}' executed successfully:")
                
                # Pretty print the result
                if isinstance(result, dict) or isinstance(result, list):
                    print(json.dumps(result, indent=2))
                else:
                    print(result)
                    
            except ValueError as e:
                print(f"❌ {e}")
            except Exception as e:
                print(f"❌ Error calling tool: {e}")
        
        self._run_async(call_tool())
    
    @line_magic
    def llm_mcp_config(self, line):
        """Show or set MCP configuration.
        
        Usage:
            %llm_mcp_config                    # Show current config
            %llm_mcp_config reload             # Reload config from file
            %llm_mcp_config path/to/config.json # Load specific config file
        """
        arg = line.strip()
        
        if not arg:
            # Show current configuration
            if not self.kernel.mcp_manager.server_configs:
                print("❌ No MCP servers configured")
                print("\n💡 Create a config file at one of these locations:")
                print("   - ~/.llm-kernel/mcp-config.json")
                print("   - ./mcp-config.json")
                print("   - ./.mcp/config.json")
                print("\nExample config:")
                print(json.dumps({
                    "mcpServers": {
                        "filesystem": {
                            "command": "npx",
                            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/allowed/dir"]
                        },
                        "github": {
                            "url": "https://api.github.com/mcp"
                        }
                    }
                }, indent=2))
            else:
                print("📋 MCP Server Configuration:")
                print(json.dumps({"mcpServers": self.kernel.mcp_manager.server_configs}, indent=2))
                
                # Show connected servers
                connected = self.kernel.mcp_manager.get_connected_servers()
                if connected:
                    print(f"\n✅ Connected servers: {', '.join(connected)}")
                else:
                    print("\n⚠️  No servers connected. Use %llm_mcp_connect")
                    
        elif arg == 'reload':
            # Reload configuration
            old_config = self.kernel.mcp_manager.server_configs.copy()
            self.kernel.mcp_manager.load_config()
            
            if self.kernel.mcp_manager.server_configs != old_config:
                print("✅ Configuration reloaded")
                print("💡 Use %llm_mcp_connect to connect to new servers")
            else:
                print("ℹ️  Configuration unchanged")
                
        else:
            # Load specific config file
            try:
                self.kernel.mcp_manager.load_config(arg)
                print(f"✅ Loaded configuration from {arg}")
                print("💡 Use %llm_mcp_connect to connect to servers")
            except Exception as e:
                print(f"❌ Error loading config: {e}")
    
    @cell_magic
    def llm_mcp(self, line, cell):
        """Query LLM with MCP tools available.
        
        The LLM will have access to all connected MCP tools and can use them
        to help answer your query.
        
        Usage:
            %%llm_mcp
            Can you read the README.md file and summarize it?
            
            %%llm_mcp --model=gpt-4
            Check the current git status and explain any uncommitted changes.
        """
        if not self.kernel.mcp_manager.is_connected():
            print("⚠️  No MCP servers connected. The LLM won't have access to external tools.")
            print("💡 Use %llm_mcp_connect to connect to MCP servers first")
        
        # Parse model from line
        import shlex
        try:
            args = shlex.split(line)
        except ValueError:
            args = line.split()
        
        model = None
        for i, arg in enumerate(args):
            if arg == '--model' and i + 1 < len(args):
                model = args[i + 1]
            elif arg.startswith('--model='):
                model = arg.split('=', 1)[1]
        
        query = cell.strip()
        if not query:
            print("❌ Please provide a query")
            return
        
        # Query with MCP tools
        try:
            result = self._run_async(
                self.kernel.query_llm_with_mcp_async(query, model)
            )
            
            # Display result based on mode
            if self.kernel.display_mode == 'chat':
                self._display_chat_response(result, model or self.kernel.active_model)
            else:
                from IPython.display import Markdown
                display(Markdown(result))
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def _run_async(self, coro):
        """Run async function in sync context."""
        loop = asyncio.get_event_loop()
        if loop.is_running():
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        else:
            return loop.run_until_complete(coro)
    
    def _display_chat_response(self, response: str, model: str):
        """Display response in chat format."""
        html = f'''
        <div style="margin: 10px 0 20px 40px; padding: 10px; background: #f5f5f5; 
                    border-radius: 10px; border-left: 3px solid #2196F3;">
            <strong>🤖 {model}:</strong><br>
            <div style="margin-top: 8px; white-space: pre-wrap;">{response}</div>
        </div>
        '''
        display(HTML(html))