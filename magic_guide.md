# LLM Kernel Magic Commands Guide

This guide covers all magic commands available in the LLM Kernel, organized by category.

## Table of Contents

1. [Chat Mode & Basic Interaction](#chat-mode--basic-interaction)
2. [Model Management](#model-management)
3. [Context Management](#context-management)
4. [Cell Visibility (Hide/Show)](#cell-visibility-hideshow)
5. [Context Persistence](#context-persistence)
6. [Conversation Management](#conversation-management)
7. [Configuration & Settings](#configuration--settings)
8. [Debugging & Development](#debugging--development)
9. [Cell Magic Commands](#cell-magic-commands)

---

## Chat Mode & Basic Interaction

### `%llm_chat`
Toggle chat mode on/off. When enabled, you can type naturally in cells without using `%%llm`.

```python
%llm_chat          # Toggle chat mode
%llm_chat on       # Enable chat mode
%llm_chat off      # Disable chat mode
%llm_chat status   # Check current status
```

**Example:**
```python
%llm_chat on
# Now you can just type in cells:
What is machine learning?
```

---

## Model Management

### `%llm_models`
List all available LLM models configured in your environment.

```python
%llm_models
```

**Output example:**
```
🤖 Available LLM Models:
  ✅ (active) gpt-4o
  ⚪ claude-3-sonnet
  ⚪ gemini-pro
  ⚪ ollama/llama3
```

### `%llm_model`
Switch between different LLM models or check the current model.

```python
%llm_model                  # Show current model
%llm_model gpt-4o          # Switch to GPT-4
%llm_model claude-3-sonnet  # Switch to Claude
```

---

## Context Management

### `%llm_context`
Display the current context that will be sent to the LLM. Shows all messages in the context window.

```python
%llm_context
```

**Output includes:**
- List of all messages (truncated for display)
- Total message count
- Estimated token usage
- Hidden cells (if any)

### `%llm_notebook_context`
Toggle notebook context mode. When enabled (default with chat mode), notebook cells literally become the LLM's context window.

```python
%llm_notebook_context          # Toggle mode
%llm_notebook_context on       # Enable
%llm_notebook_context off      # Disable
%llm_notebook_context status   # Check status
```

### `%llm_clear`
Clear the conversation history.

```python
%llm_clear
```

---

## Cell Visibility (Hide/Show)

### `%%hide` (Cell Magic)
Hide a cell from the LLM context. The cell still executes normally but won't be included in the context sent to the LLM.

```python
%%hide
# This cell is hidden from the LLM
api_key = "sk-secret-key"
password = "super-secret"
```

### `%llm_unhide`
Unhide previously hidden cells.

```python
%llm_unhide 5      # Unhide cell number 5
%llm_unhide all    # Unhide all cells
```

### `%llm_hidden`
Show which cells are currently hidden from the context.

```python
%llm_hidden
```

**Output example:**
```
🙈 Hidden cells: 3, 7, 12
```

---

## Context Persistence

### `%llm_context_save`
Save the current context to a JSON file for later use.

```python
%llm_context_save                      # Auto-named file
%llm_context_save my_session.json      # Specific filename
```

### `%llm_context_load`
Load a previously saved context from a file.

```python
%llm_context_load my_session.json
```

### `%llm_context_reset`
Reset context to a clean state.

```python
%llm_context_reset                # Clear everything
%llm_context_reset --keep-hidden  # Clear but keep hidden cells
```

### `%llm_context_persist`
Control whether context persists across kernel restarts (default: on).

```python
%llm_context_persist          # Toggle
%llm_context_persist on       # Enable persistence
%llm_context_persist off      # Disable persistence
%llm_context_persist status   # Check current setting
```

---

## Conversation Management

### `%llm_status`
Show comprehensive kernel status including model, context size, and conversation history.

```python
%llm_status
```

**Output includes:**
- Active model
- Available models count
- Conversation history length
- Context window usage
- Token estimates

### `%llm_history`
Display conversation history in a formatted view.

```python
%llm_history              # Show recent history
%llm_history --all        # Show complete history
%llm_history --last=10    # Show last 10 exchanges
```

### `%llm_prune`
Intelligently prune conversation history to reduce context size.

```python
%llm_prune                           # Use default strategy
%llm_prune --strategy=semantic      # Semantic similarity pruning
%llm_prune --strategy=recency       # Keep only recent exchanges
%llm_prune --threshold=0.7          # Set relevance threshold
```

### `%llm_pin_cell`
Pin important cells to always include them in context.

```python
%llm_pin_cell 5    # Pin cell number 5
```

---

## Configuration & Settings

### `%llm_config`
Show interactive configuration panel with widgets for settings.

```python
%llm_config
```

### `%llm_display`
Set the display mode for LLM responses.

```python
%llm_display           # Show current mode
%llm_display inline    # Traditional inline display
%llm_display chat      # Chat-style display
```

---

## Debugging & Development

### `%llm_debug`
Enable debugger for VS Code integration.

```python
%llm_debug           # Start on default port (5678)
%llm_debug 5679      # Start on specific port
%llm_debug --wait    # Wait for debugger to attach
```

---

## Cell Magic Commands

### `%%llm`
Query the LLM with the cell content. Can specify model and other options.

```python
%%llm
What is the meaning of life?
```

```python
%%llm --model=gpt-4o
Explain quantum computing in simple terms.
```

### `%%llm_gpt4`
Shortcut to query GPT-4 specifically.

```python
%%llm_gpt4
Write a Python function to calculate fibonacci numbers.
```

### `%%llm_claude`
Shortcut to query Claude specifically.

```python
%%llm_claude
Analyze this code for potential improvements.
```

### `%%llm_compare`
Compare responses from multiple models side-by-side.

```python
%%llm_compare gpt-4o claude-3-sonnet
What's the best way to handle errors in Python?
```

## Context Reranking & Custom Processing

### `%llm_rerank`
Use LLM to intelligently reorder cells by relevance to a query.

```python
%llm_rerank "machine learning"          # Rerank by relevance
%llm_rerank --show "web development"    # Show ranking without reordering
%llm_rerank --top=10 "Python basics"    # Keep only top 10 most relevant
%llm_rerank                              # Rerank based on last query
```

### `%llm_rerank_clear`
Clear reranking and restore original cell order.

```python
%llm_rerank_clear
```

### `%%meta` (Cell Magic)
Define custom context processing functions for filtering, ranking, or transforming.

```python
%%meta filter
def filter_cells(messages):
    """Custom filter logic"""
    filtered = []
    for msg in messages:
        if len(msg['content']) > 50:  # Keep only longer messages
            filtered.append(msg)
    return filtered
```

```python
%%meta ranking
def rank_cells(messages, query):
    """Custom ranking logic"""
    # Your ranking algorithm here
    return reordered_messages
```

```python
%%meta transform
def transform_context(messages):
    """Custom transformation logic"""
    # Add metadata, modify content, etc.
    return transformed_messages
```

### `%llm_apply_meta`
Apply custom meta functions to the context.

```python
%llm_apply_meta filter                  # Apply filter function
%llm_apply_meta ranking "query text"    # Apply ranking with query
%llm_apply_meta transform               # Apply transformation
%llm_apply_meta all "query"            # Apply all functions in sequence
```

### `%llm_meta_list`
List all defined meta functions.

```python
%llm_meta_list
```

---

## Common Workflows

### Starting a New Session
```python
%llm_chat on              # Enable chat mode
%llm_model gpt-4o         # Select your preferred model
# Start typing naturally in cells!
```

### Working with Sensitive Data
```python
%%hide
api_key = "secret-key"
password = "secret-pass"

# In next cell (not hidden):
# Use the credentials without exposing them to LLM
connect_to_api(api_key)
```

### Saving and Resuming Work
```python
# Before closing notebook:
%llm_context_save project_session.json

# When returning:
%llm_context_load project_session.json
%llm_chat on
# Continue where you left off!
```

### Managing Large Contexts
```python
%llm_context              # Check current size
%llm_prune                # Reduce if needed
%llm_pin_cell 3           # Keep important cells
%llm_hidden               # Check hidden cells
```

### Comparing Models
```python
%%llm_compare gpt-4o claude-3-sonnet gemini-pro
Explain the concept of recursion with an example.
```

### Smart Context Management with Reranking
```python
# When working on a specific topic
%llm_rerank "machine learning algorithms"

# Check what's most relevant
%llm_rerank --show "neural networks"

# Keep only the most relevant context
%llm_rerank --top=10 "deep learning"

# Define custom filtering
%%meta filter
def filter_cells(messages):
    # Remove short or irrelevant cells
    return [m for m in messages if len(m['content']) > 100]

# Apply the filter
%llm_apply_meta filter

# When done, restore original order
%llm_rerank_clear
```

---

## Tips and Best Practices

1. **Use `%%hide` for sensitive data** - API keys, passwords, and personal information should be in hidden cells

2. **Pin important cells** - Use `%llm_pin_cell` for crucial context like function definitions or project requirements

3. **Monitor context size** - Use `%llm_context` regularly to ensure you're not exceeding token limits

4. **Save sessions** - Use `%llm_context_save` before closing important work sessions

5. **Chat mode for natural flow** - Enable `%llm_chat` for conversational interactions without magic commands

6. **Compare models for best results** - Use `%%llm_compare` to find which model works best for your use case

---

## Quick Reference

| Command | Description |
|---------|-------------|
| **Chat Mode** |
| `%llm_chat` | Toggle chat mode |
| **Models** |
| `%llm_models` | List available models |
| `%llm_model` | Switch active model |
| **Context** |
| `%llm_context` | Show current context |
| `%llm_notebook_context` | Toggle notebook context mode |
| `%llm_clear` | Clear conversation |
| **Hide/Show** |
| `%%hide` | Hide cell from LLM |
| `%llm_unhide` | Unhide cells |
| `%llm_hidden` | Show hidden cells |
| **Persistence** |
| `%llm_context_save` | Save context |
| `%llm_context_load` | Load context |
| `%llm_context_reset` | Reset context |
| `%llm_context_persist` | Toggle persistence |
| **Management** |
| `%llm_status` | Show kernel status |
| `%llm_history` | Show conversation history |
| `%llm_prune` | Prune conversation |
| `%llm_pin_cell` | Pin important cells |
| **Reranking** |
| `%llm_rerank` | Reorder cells by relevance |
| `%llm_rerank_clear` | Clear reranking |
| `%llm_apply_meta` | Apply custom functions |
| `%llm_meta_list` | List meta functions |
| **Config** |
| `%llm_config` | Configuration panel |
| `%llm_display` | Set display mode |
| `%llm_debug` | Enable debugging |
| **Cell Magics** |
| `%%llm` | Query LLM |
| `%%llm_gpt4` | Query GPT-4 |
| `%%llm_claude` | Query Claude |
| `%%llm_compare` | Compare models |
| `%%hide` | Hide cell from context |
| `%%meta` | Define custom functions |

---

*For more examples and demos, check out the demo notebooks in the repository.*