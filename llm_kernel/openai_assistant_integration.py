"""
OpenAI Assistant Integration for PDF and File Support

This module provides integration with OpenAI's Assistants API which supports
native file uploads including PDFs.
"""

import os
import json
import time
from typing import Dict, List, Any, Optional
from pathlib import Path

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    openai = None


class OpenAIAssistantIntegration:
    """Handles OpenAI Assistant API for file-based queries."""
    
    def __init__(self, logger=None):
        self.log = logger
        self._assistants = {}  # model -> assistant_id
        self._threads = {}     # conversation_id -> thread_id
        self._client = None
        
        if HAS_OPENAI and os.getenv('OPENAI_API_KEY'):
            self._client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    def get_or_create_assistant(self, model: str = "gpt-4o") -> Optional[str]:
        """Get or create an assistant for the given model."""
        if not self._client:
            return None
            
        if model in self._assistants:
            return self._assistants[model]
        
        try:
            # Create a new assistant
            assistant = self._client.beta.assistants.create(
                name=f"LLM Kernel Assistant ({model})",
                instructions="You are a helpful AI assistant that can read and analyze documents including PDFs. Answer questions based on the uploaded files and conversation context.",
                tools=[{"type": "code_interpreter"}],
                model=model
            )
            
            self._assistants[model] = assistant.id
            if self.log:
                self.log.info(f"Created OpenAI assistant: {assistant.id}")
            
            return assistant.id
            
        except Exception as e:
            if self.log:
                self.log.error(f"Failed to create assistant: {e}")
            return None
    
    def get_or_create_thread(self, conversation_id: str = "default") -> Optional[str]:
        """Get or create a thread for the conversation."""
        if not self._client:
            return None
            
        if conversation_id in self._threads:
            return self._threads[conversation_id]
        
        try:
            thread = self._client.beta.threads.create()
            self._threads[conversation_id] = thread.id
            
            if self.log:
                self.log.info(f"Created thread: {thread.id}")
            
            return thread.id
            
        except Exception as e:
            if self.log:
                self.log.error(f"Failed to create thread: {e}")
            return None
    
    def query_with_files(self, query: str, file_ids: List[str], model: str = "gpt-4o", 
                        conversation_id: str = "default") -> Optional[str]:
        """Query the assistant with attached files."""
        if not self._client:
            return None
        
        # Get or create assistant and thread
        assistant_id = self.get_or_create_assistant(model)
        thread_id = self.get_or_create_thread(conversation_id)
        
        if not assistant_id or not thread_id:
            return None
        
        try:
            # Add file attachments
            attachments = []
            for file_id in file_ids:
                attachments.append({
                    "file_id": file_id,
                    "tools": [{"type": "code_interpreter"}]
                })
            
            # Create message with file attachments
            message = self._client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=query,
                attachments=attachments if attachments else None
            )
            
            # Run the assistant
            run = self._client.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id=assistant_id
            )
            
            # Wait for completion
            while run.status in ['queued', 'in_progress']:
                time.sleep(0.5)
                run = self._client.beta.threads.runs.retrieve(
                    thread_id=thread_id,
                    run_id=run.id
                )
            
            if run.status == 'completed':
                # Get the response
                messages = self._client.beta.threads.messages.list(
                    thread_id=thread_id,
                    order='desc',
                    limit=1
                )
                
                if messages.data:
                    # Extract text from the response
                    response_text = ""
                    for content in messages.data[0].content:
                        if content.type == 'text':
                            response_text += content.text.value
                    
                    return response_text
                    
            elif run.status == 'failed':
                error_msg = f"Assistant run failed: {run.last_error}"
                if self.log:
                    self.log.error(error_msg)
                return error_msg
                
        except Exception as e:
            if self.log:
                self.log.error(f"Error querying assistant: {e}")
            return f"Error: {str(e)}"
        
        return None
    
    def add_message_to_thread(self, content: str, thread_id: str, role: str = "user") -> bool:
        """Add a message to an existing thread."""
        if not self._client:
            return False
            
        try:
            self._client.beta.threads.messages.create(
                thread_id=thread_id,
                role=role,
                content=content
            )
            return True
        except Exception as e:
            if self.log:
                self.log.error(f"Failed to add message to thread: {e}")
            return False
    
    def delete_assistant(self, assistant_id: str) -> bool:
        """Delete an assistant."""
        if not self._client:
            return False
            
        try:
            self._client.beta.assistants.delete(assistant_id)
            # Remove from cache
            self._assistants = {k: v for k, v in self._assistants.items() if v != assistant_id}
            return True
        except Exception as e:
            if self.log:
                self.log.error(f"Failed to delete assistant: {e}")
            return False
    
    def cleanup(self):
        """Clean up all assistants and threads."""
        # Delete all assistants
        for model, assistant_id in list(self._assistants.items()):
            self.delete_assistant(assistant_id)
        
        # Clear thread cache
        self._threads.clear()