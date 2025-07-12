#!/usr/bin/env python3
"""
Test PDF with retry logic for handling temporary OpenAI server errors.
"""

import asyncio
import time
from typing import Optional

async def query_with_retry(query_func, max_retries=3, initial_delay=2):
    """
    Retry a query function with exponential backoff.
    
    Args:
        query_func: Async function to call
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds between retries
    """
    last_error = None
    delay = initial_delay
    
    for attempt in range(max_retries + 1):
        try:
            result = await query_func()
            return result
        except Exception as e:
            last_error = e
            error_str = str(e)
            
            # Check if it's a retryable error
            if any(err in error_str for err in ["InternalServerError", "500", "502", "503", "504"]):
                if attempt < max_retries:
                    print(f"⚠️  Attempt {attempt + 1} failed with server error. Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                    delay *= 2  # Exponential backoff
                    continue
            
            # Non-retryable error or out of retries
            raise
    
    # If we get here, we've exhausted retries
    raise Exception(f"Failed after {max_retries + 1} attempts. Last error: {last_error}")


# Example implementation for the kernel
def add_retry_to_llm_query():
    """
    Example of how to add retry logic to the LLM query method.
    """
    code = '''
    async def query_llm_async_with_retry(self, query: str, model: str = None, **kwargs) -> str:
        """Query LLM with automatic retry for server errors."""
        
        async def _do_query():
            return await self.query_llm_async(query, model, **kwargs)
        
        max_retries = kwargs.pop('max_retries', 3)
        retry_delay = kwargs.pop('retry_delay', 2)
        
        for attempt in range(max_retries + 1):
            try:
                result = await _do_query()
                return result
            except Exception as e:
                error_str = str(e)
                
                # Check if it's a retryable error
                if ("InternalServerError" in error_str or 
                    "server had an error" in error_str.lower() or
                    any(code in error_str for code in ["500", "502", "503", "504"])):
                    
                    if attempt < max_retries:
                        wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                        self.logger.info(f"Server error on attempt {attempt + 1}, retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        continue
                
                # Non-retryable error or exhausted retries
                raise
        
        # Should never reach here
        raise Exception("Unexpected error in retry logic")
    '''
    
    print("Add this method to LLMIntegration class for automatic retry:")
    print(code)


if __name__ == "__main__":
    print("PDF Query Retry Strategy\n")
    print("="*50)
    
    print("\n1. When to retry:")
    print("   • InternalServerError (500)")
    print("   • Bad Gateway (502)")
    print("   • Service Unavailable (503)")
    print("   • Gateway Timeout (504)")
    
    print("\n2. Retry strategy:")
    print("   • Max attempts: 3")
    print("   • Initial delay: 2 seconds")
    print("   • Exponential backoff: 2s, 4s, 8s")
    
    print("\n3. Non-retryable errors:")
    print("   • Authentication errors (401, 403)")
    print("   • Invalid request errors (400)")
    print("   • Not found errors (404)")
    print("   • Rate limit errors (429) - handle separately")
    
    print("\n" + "="*50)
    add_retry_to_llm_query()
    
    print("\n✅ Retry logic can help handle temporary OpenAI server issues automatically!")