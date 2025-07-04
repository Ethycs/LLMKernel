# Debugging the LLM Kernel

This guide covers canonical methods for debugging Jupyter kernels, specifically the LLM Kernel.

## 1. Enable Kernel Debug Logging

### Method 1: Environment Variable
```bash
export JUPYTER_ENABLE_DEBUG=1
export LLM_KERNEL_DEBUG=true
jupyter lab --debug
```

### Method 2: Direct Kernel Launch
```bash
# Find the kernel connection file
jupyter kernel --debug

# In another terminal, connect with:
jupyter console --existing kernel-xxxxx.json --debug
```

## 2. Attach a Debugger

### Using VS Code (Recommended)

1. Add this to your kernel code (e.g., in `do_execute`):
```python
import debugpy
debugpy.listen(5678)
debugpy.wait_for_client()  # Kernel will pause here
```

2. Create `.vscode/launch.json`:
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Attach to Kernel",
            "type": "python",
            "request": "attach",
            "connect": {
                "host": "localhost",
                "port": 5678
            },
            "justMyCode": false
        }
    ]
}
```

3. Start Jupyter, run a cell, then attach debugger in VS Code

### Using pdb (Command Line)

Add breakpoints in your kernel code:
```python
import pdb; pdb.set_trace()
```

## 3. Kernel Logging

### Add Strategic Logging

```python
# In kernel.py
import logging

class LLMKernel(IPythonKernel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Setup file logging
        file_handler = logging.FileHandler('llm_kernel_debug.log')
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        self.log.addHandler(file_handler)
        self.log.setLevel(logging.DEBUG)
    
    def do_execute(self, code, silent, store_history=True, 
                   user_expressions=None, allow_stdin=False):
        self.log.debug(f"Executing code: {code[:100]}...")
        self.log.debug(f"Parent header: {self._parent_header}")
        # ... rest of method
```

Then tail the log:
```bash
tail -f llm_kernel_debug.log
```

## 4. Test Kernel Outside Jupyter

### Create a Test Script

```python
# test_kernel_direct.py
import asyncio
from llm_kernel.kernel import LLMKernel

async def test_kernel():
    kernel = LLMKernel()
    
    # Test query
    result = await kernel.query_llm_async("Hello, world!")
    print(f"Result: {result}")
    
    # Test interrupt
    future = asyncio.create_task(
        kernel.query_llm_async("Tell me a long story")
    )
    await asyncio.sleep(1)
    future.cancel()
    
    try:
        await future
    except asyncio.CancelledError:
        print("Successfully cancelled!")

asyncio.run(test_kernel())
```

## 5. Jupyter Debug Commands

### Inspect Kernel State

```python
# In a notebook cell
import IPython
kernel = IPython.get_ipython().kernel

# Inspect kernel attributes
print(f"Execution count: {kernel.execution_count}")
print(f"Active model: {kernel.active_model}")
print(f"Context strategy: {kernel.context_manager.context_strategy}")

# Check parent message
print(f"Parent header: {kernel._parent_header}")
```

### Monitor Messages

```python
# Create a debug cell magic
from IPython.core.magic import cell_magic, Magics, magics_class

@magics_class
class DebugMagics(Magics):
    @cell_magic
    def debug_msg(self, line, cell):
        kernel = self.shell.kernel
        print(f"Parent: {kernel._parent_header}")
        print(f"Cell ID: {kernel._current_cell_id}")
        return eval(cell)

# Register it
ip = get_ipython()
ip.register_magics(DebugMagics)
```

## 6. Common Debug Scenarios

### Debug Interrupt Issues

```python
# Add to query_llm_sync
self.log.debug(f"Starting query, future: {self._current_execution}")

# In interrupt handler
def do_interrupt(self):
    self.log.debug(f"Interrupt received, current execution: {self._current_execution}")
    if hasattr(self, '_current_execution') and self._current_execution:
        self.log.debug(f"Cancelling future: {self._current_execution}")
        self._current_execution.cancel()
    return super().do_interrupt()
```

### Debug Context Issues

```python
# Add to do_execute
self.log.debug(f"Cell metadata: {metadata}")
self.log.debug(f"Resolved cell ID: {cell_id}")
self.log.debug(f"Execution history: {len(self.execution_tracker.execution_history)} cells")
```

### Debug Async Issues

```python
# Check event loop state
import asyncio

try:
    loop = asyncio.get_event_loop()
    self.log.debug(f"Event loop running: {loop.is_running()}")
    self.log.debug(f"Event loop: {loop}")
except RuntimeError as e:
    self.log.debug(f"No event loop: {e}")
```

## 7. Performance Profiling

```python
import cProfile
import pstats
from io import StringIO

# Add profiling to slow methods
def profile_method(func):
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        result = func(*args, **kwargs)
        pr.disable()
        
        s = StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(10)  # Top 10 functions
        
        with open('kernel_profile.log', 'a') as f:
            f.write(f"\n\nProfile for {func.__name__}:\n")
            f.write(s.getvalue())
        
        return result
    return wrapper

# Use it
@profile_method
def query_llm_sync(self, query, model=None, **kwargs):
    # ... method implementation
```

## 8. Debug Tools

### Install Debug Dependencies

```bash
pixi add --feature dev debugpy ipdb memory-profiler
```

### Memory Debugging

```python
from memory_profiler import profile

@profile
def do_execute(self, code, silent, store_history=True, 
               user_expressions=None, allow_stdin=False):
    # ... method implementation
```

Run with:
```bash
python -m memory_profiler llm_kernel/kernel.py
```

## 9. Kernel Connection Debugging

### Find Kernel Connection Info

```bash
# List running kernels
jupyter kernel list

# Get connection info
cat $(jupyter --runtime-dir)/kernel-*.json
```

### Connect Manually

```python
import jupyter_client

# Connect to running kernel
cf = jupyter_client.find_connection_file()
km = jupyter_client.BlockingKernelClient(connection_file=cf)
km.load_connection_file()
km.start_channels()

# Send execute request
msg_id = km.execute("print('Hello from external client')")

# Get reply
reply = km.get_shell_msg()
print(reply)
```

## 10. Common Issues and Solutions

### Issue: Kernel Dies Silently

Add exception handling with logging:
```python
def do_execute(self, code, silent, store_history=True, 
               user_expressions=None, allow_stdin=False):
    try:
        # ... your code
    except Exception as e:
        self.log.error(f"Execution error: {e}", exc_info=True)
        return {
            'status': 'error',
            'ename': type(e).__name__,
            'evalue': str(e),
            'traceback': []
        }
```

### Issue: Can't Find Log Output

Check these locations:
- `~/.local/share/jupyter/runtime/` (Linux/Mac)
- `%APPDATA%\jupyter\runtime\` (Windows)
- Current working directory
- Jupyter console output

### Issue: Debugger Won't Attach

1. Check firewall settings for port 5678
2. Ensure `debugpy` is installed in kernel environment
3. Try `debugpy.listen(("0.0.0.0", 5678))` for all interfaces

## Quick Debug Checklist

1. ✅ Enable debug logging: `export JUPYTER_ENABLE_DEBUG=1`
2. ✅ Add file logging to kernel
3. ✅ Use `self.log.debug()` liberally
4. ✅ Attach debugger for complex issues
5. ✅ Test outside Jupyter for isolation
6. ✅ Check kernel connection files
7. ✅ Profile performance bottlenecks

Remember: The kernel runs in a separate process, so print statements won't show in the notebook!