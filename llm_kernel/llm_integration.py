"""
LLM Integration Module for LLM Kernel

Handles core LLM querying functionality including standard queries
and MCP-enhanced queries with tool calling.
"""

import json
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

try:
    import litellm
except ImportError:
    litellm = None


class LLMIntegration:
    """Handles LLM queries and tool interactions."""
    
    def __init__(self, kernel_instance):
        self.kernel = kernel_instance
        self.logger = kernel_instance.log
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def query_llm_async(self, query: str, model: str = None, **kwargs) -> str:
        """Asynchronously query an LLM model."""
        if model is None:
            model = self.kernel.active_model
            
        if model not in self.kernel.llm_clients:
            raise ValueError(f"Model {model} not available. Available: {list(self.kernel.llm_clients.keys())}")
            
        model_name = self.kernel.llm_clients[model]
        
        # Always use notebook cells as context when in a notebook environment
        messages = self.kernel.get_notebook_cells_as_context()
        messages.append({"role": "user", "content": query})
        
        try:
            # Use LiteLLM to query the model
            response = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: litellm.completion(
                    model=model_name,
                    messages=messages,
                    **kwargs
                )
            )
            
            result = response.choices[0].message.content
            
            # Track costs if available
            if hasattr(response, '_hidden_params') and 'response_cost' in response._hidden_params:
                self._track_cost(model, response._hidden_params['response_cost'])
            
            # Track the exchange
            self.kernel.track_exchange(model, query, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error querying {model}: {e}")
            return f"Error: {str(e)}"
    
    async def query_llm_with_mcp_async(self, query: str, model: str = None, **kwargs) -> str:
        """Query LLM with MCP tools available for function calling."""
        if model is None:
            model = self.kernel.active_model
            
        if model not in self.kernel.llm_clients:
            raise ValueError(f"Model {model} not available")
            
        model_name = self.kernel.llm_clients[model]
        
        # Get available tools from MCP
        mcp_tools = []
        if hasattr(self.kernel, 'mcp_manager') and self.kernel.mcp_manager:
            mcp_tools = self._get_mcp_tools_for_llm()
        
        # Always use notebook cells as context
        messages = self.kernel.get_notebook_cells_as_context()
        messages.append({"role": "user", "content": query})
        
        # Add system message about available tools
        if mcp_tools:
            tools_desc = self._create_tools_description(mcp_tools)
            system_msg = {
                "role": "system",
                "content": f"You have access to the following tools:\n\n{tools_desc}\n\nYou can use these tools by responding with a special format:\n<tool_use>\n<tool_name>tool_name</tool_name>\n<parameters>{{\"param\": \"value\"}}</parameters>\n</tool_use>\n\nAfter using a tool, you'll receive the result and can continue your response."
            }
            messages.insert(0, system_msg)
        
        try:
            # Initial query
            response = await self._query_with_tools(model_name, messages, mcp_tools, **kwargs)
            
            # Handle tool calls if any
            final_response = await self._handle_tool_calls(model_name, messages, response, mcp_tools, **kwargs)
            
            # Track the exchange
            self.kernel.track_exchange(model, query, final_response)
            
            return final_response
            
        except Exception as e:
            self.logger.error(f"Error in MCP-enhanced query: {e}")
            return f"Error: {str(e)}"
    
    async def _query_with_tools(self, model_name: str, messages: List[Dict], tools: List[Dict], **kwargs) -> str:
        """Query LLM with tools available."""
        # For now, use standard completion
        # TODO: When LiteLLM supports function calling for more models,
        # we can use the tools parameter directly
        
        response = await asyncio.get_event_loop().run_in_executor(
            self.executor,
            lambda: litellm.completion(
                model=model_name,
                messages=messages,
                **kwargs
            )
        )
        
        result = response.choices[0].message.content
        
        # Track costs
        if hasattr(response, '_hidden_params') and 'response_cost' in response._hidden_params:
            self._track_cost(model_name, response._hidden_params['response_cost'])
        
        return result
    
    async def _handle_tool_calls(self, model_name: str, messages: List[Dict], response: str, tools: List[Dict], **kwargs) -> str:
        """Handle tool calls in the response."""
        # Check if response contains tool calls
        if '<tool_use>' not in response:
            return response
        
        # Parse tool calls
        import re
        tool_calls = re.findall(r'<tool_use>(.*?)</tool_use>', response, re.DOTALL)
        
        if not tool_calls:
            return response
        
        # Process each tool call
        tool_results = []
        for tool_call in tool_calls:
            # Extract tool name and parameters
            name_match = re.search(r'<tool_name>(.*?)</tool_name>', tool_call)
            params_match = re.search(r'<parameters>(.*?)</parameters>', tool_call, re.DOTALL)
            
            if name_match and params_match:
                tool_name = name_match.group(1).strip()
                try:
                    params = json.loads(params_match.group(1).strip())
                except json.JSONDecodeError:
                    tool_results.append(f"Error: Invalid JSON parameters for {tool_name}")
                    continue
                
                # Call the tool
                try:
                    result = await self.kernel.mcp_manager.call_tool(tool_name, params)
                    tool_results.append(f"Tool '{tool_name}' result:\n{json.dumps(result, indent=2)}")
                except Exception as e:
                    tool_results.append(f"Error calling {tool_name}: {str(e)}")
        
        # If we have tool results, query again with the results
        if tool_results:
            # Add tool results to conversation
            tool_msg = "\n\n".join(tool_results)
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": f"Here are the tool results:\n\n{tool_msg}\n\nPlease continue with your response based on these results."})
            
            # Query again
            final_response = await self._query_with_tools(model_name, messages, tools, **kwargs)
            return final_response
        
        return response
    
    def _get_mcp_tools_for_llm(self) -> List[Dict[str, Any]]:
        """Get MCP tools formatted for LLM function calling."""
        if not hasattr(self.kernel, 'mcp_manager') or not self.kernel.mcp_manager:
            return []
            
        if not self.kernel.mcp_manager.is_connected():
            return []
        
        tools = []
        for tool in self.kernel.mcp_manager.list_tools():
            # Convert to OpenAI function calling format
            tool_def = {
                "type": "function",
                "function": {
                    "name": tool['name'],
                    "description": tool.get('description', ''),
                    "parameters": tool.get('inputSchema', {})
                }
            }
            tools.append(tool_def)
        
        return tools
    
    def _create_tools_description(self, tools: List[Dict]) -> str:
        """Create a human-readable description of available tools."""
        descriptions = []
        for tool in tools:
            func = tool.get('function', {})
            name = func.get('name', 'unknown')
            desc = func.get('description', 'No description')
            params = func.get('parameters', {})
            
            # Format parameters
            param_desc = []
            if 'properties' in params:
                for param_name, param_info in params['properties'].items():
                    param_type = param_info.get('type', 'any')
                    param_desc.append(f"  - {param_name} ({param_type}): {param_info.get('description', '')}")
            
            tool_desc = f"• {name}: {desc}"
            if param_desc:
                tool_desc += "\n" + "\n".join(param_desc)
            
            descriptions.append(tool_desc)
        
        return "\n\n".join(descriptions)
    
    def _track_cost(self, model: str, cost: float):
        """Track cost for the session."""
        if not hasattr(self.kernel, 'session_costs'):
            self.kernel.session_costs = {'total': 0.0, 'by_model': {}}
        
        self.kernel.session_costs['total'] += cost
        if model not in self.kernel.session_costs['by_model']:
            self.kernel.session_costs['by_model'][model] = 0.0
        self.kernel.session_costs['by_model'][model] += cost