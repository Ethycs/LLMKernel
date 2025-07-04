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
10. [Multimodal Content](#multimodal-content)

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

**New in v0.2.0:**
- Context automatically rescans when you add new cells
- Works with notebook file reading - all cells above are included

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

**New in v0.2.0:** This command now reads directly from the notebook file and automatically rescans for changes.

```python
%llm_context                  # Show context and rescan for changes
%llm_context --no-rescan      # Show context without rescanning
```

**Features:**
- **Automatic rescanning**: Detects and includes new/edited cells above the current position
- **Notebook file reading**: Shows ALL cells in the notebook, not just executed ones
- **Smart tracking**: Auto-rescans when you've added cells since the last scan

**Output includes:**
- Notebook file path being read
- List of all messages (truncated for display)
- Total message count
- Estimated token usage
- Hidden cells (if any)
- Auto-rescan indicator when new cells are detected

### `%llm_notebook_context` (Deprecated)
*Note: This command has been removed in v0.2.0. The kernel now always uses notebook cells as context.*

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

## Context Window Management

### `%llm_context_window`
Display context window information and usage for LLM models.

```python
%llm_context_window                # Show current model's context window
%llm_context_window all            # Show all models' context windows  
%llm_context_window gpt-4          # Show specific model's context window
```

**Output example:**
```
📊 Context Window for gpt-4o:
   Max tokens: 128,000
   Current usage: 2,450 tokens (1.9%)
   [██░░░░░░░░░░░░░░░░░░░░░░░░░░░░]
```

### `%llm_token_count`
Count tokens in current context or provided text using model-specific tokenizers.

```python
%llm_token_count                           # Count tokens in current context
%llm_token_count "Hello world!"            # Count tokens in text
%llm_token_count --model=claude-3 "text"   # Use specific model tokenizer
```

### `%llm_cost`
Track and estimate costs for LLM usage.

```python
%llm_cost                    # Show session costs
%llm_cost estimate           # Estimate cost for current context
%llm_cost --model=gpt-4      # Show costs for specific model
```

**Output example:**
```
💰 Session Costs:
   Total: $0.002450
   
   By Model:
   - gpt-4o: $0.001800
   - claude-3-sonnet: $0.000650
```

---

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

### `%llm_rerank_apply`
Apply the current reranking by physically reorganizing cells in the notebook file.

⚠️ **WARNING**: This command modifies your notebook file! Always backup first.

```python
%llm_rerank_apply              # Apply ranking (modifies notebook!)
%llm_rerank_apply --backup     # Create backup first (recommended)
```

**Workflow:**
```python
# 1. First rerank cells by relevance
%llm_rerank "data analysis and visualization"

# 2. Check the new order
%llm_context

# 3. If happy with the order, apply it permanently
%llm_rerank_apply --backup

# 4. Reload notebook in Jupyter to see changes
```

**Use cases:**
- Organizing research notes by topic
- Grouping related code cells together
- Creating a narrative flow for presentations
- Cleaning up exploratory notebooks

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

## MCP (Model Context Protocol) Integration

### `%llm_mcp_connect`
Connect to MCP servers for external tool access.

```python
%llm_mcp_connect                    # Connect all configured servers
%llm_mcp_connect filesystem         # Connect specific server
%llm_mcp_connect https://api.example.com/mcp  # Connect by URL
```

### `%llm_mcp_disconnect`
Disconnect from MCP servers.

```python
%llm_mcp_disconnect                 # Disconnect all
%llm_mcp_disconnect filesystem      # Disconnect specific server
```

### `%llm_mcp_tools`
List available tools from connected MCP servers.

```python
%llm_mcp_tools                      # List all tools
%llm_mcp_tools filesystem           # List tools from specific server
%llm_mcp_tools --json              # Output as JSON
```

**Output example:**
```
🛠️  Available MCP Tools (15 total):

📦 filesystem (5 tools):
   • filesystem.read_file
     Read contents of a file
   • filesystem.write_file
     Write content to a file
   • filesystem.list_directory
     List directory contents
```

### `%llm_mcp_call`
Call an MCP tool directly.

```python
%llm_mcp_call filesystem.read_file {"path": "README.md"}
%llm_mcp_call github.create_issue {"title": "Bug report", "body": "..."}
```

### `%llm_mcp_config`
Manage MCP server configuration.

```python
%llm_mcp_config                     # Show current config
%llm_mcp_config reload              # Reload from file
%llm_mcp_config ./my-config.json    # Load specific config
```

### `%%llm_mcp` (Cell Magic)
Query LLM with MCP tool access (coming soon).

```python
%%llm_mcp
Can you read the README.md and summarize the project?
```

---

## Context Auto-Rescanning (New in v0.2.0)

The kernel now intelligently manages context by reading directly from the notebook file and automatically rescanning when needed.

### How It Works

1. **Notebook File Reading**: The kernel finds and reads your `.ipynb` file directly
2. **Automatic Detection**: Tracks when you've added or edited cells
3. **Smart Rescanning**: Automatically rescans when:
   - You run `%llm_context`
   - You make an LLM query and new cells were added
   - You toggle chat mode

### Key Benefits

- **See ALL cells**: No need to execute cells first - they appear in context immediately
- **Edit freely**: Make changes to cells above and they're automatically picked up
- **No restarts needed**: Add new context without restarting the kernel

### Example Workflow

```python
# Cell 1: Define some data
data = [1, 2, 3, 4, 5]

# Cell 2: Enable chat mode
%llm_chat on

# Cell 3: Check context - sees all cells above
%llm_context

# Cell 4: Add a NEW cell above (between 1 and 2)
# Go back and insert: mean_value = 3.0

# Cell 5: Ask about it - auto-rescans and includes the new cell!
What is the mean_value I defined?

# Cell 6: Check what happened
%llm_context  # Shows "✨ Auto-rescanned (1 new cells detected)"
```

### Manual Control

```python
# Force a rescan
%llm_context

# Skip rescanning (faster if no changes)
%llm_context --no-rescan

# The rescan indicator shows when it happens
# ✨ Auto-rescanned (3 new cells detected)
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

### Organizing Notebooks with Reranking
```python
# Reorganize a messy notebook by topic
%llm_rerank "data preprocessing and cleaning"

# Review the suggested order
%llm_context

# If you like it, permanently reorganize the notebook
%llm_rerank_apply --backup

# Or try a different organization
%llm_rerank_clear
%llm_rerank "model training and evaluation"
%llm_context

# Apply when satisfied
%llm_rerank_apply --backup
```

---

## Tips and Best Practices

1. **Use `%%hide` for sensitive data** - API keys, passwords, and personal information should be in hidden cells

2. **Pin important cells** - Use `%llm_pin_cell` for crucial context like function definitions or project requirements

3. **Monitor context size** - Use `%llm_context` regularly to ensure you're not exceeding token limits

4. **Save sessions** - Use `%llm_context_save` before closing important work sessions

5. **Chat mode for natural flow** - Enable `%llm_chat` for conversational interactions without magic commands

6. **Compare models for best results** - Use `%%llm_compare` to find which model works best for your use case

7. **Let auto-rescan work for you** - Just add cells naturally; the kernel will detect and include them automatically

8. **All cells are visible** - With notebook file reading, you don't need to execute cells for them to appear in context

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
| `%llm_context` | Show current context (auto-rescans) |
| `%llm_context --no-rescan` | Show context without rescanning |
| `%llm_clear` | Clear conversation |
| `%llm_context_window` | Show context window info |
| `%llm_token_count` | Count tokens |
| `%llm_cost` | Track/estimate costs |
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
| `%llm_rerank_apply` | Apply reranking to notebook file |
| `%llm_apply_meta` | Apply custom functions |
| `%llm_meta_list` | List meta functions |
| **MCP Integration** |
| `%llm_mcp_connect` | Connect to MCP servers |
| `%llm_mcp_disconnect` | Disconnect servers |
| `%llm_mcp_tools` | List available tools |
| `%llm_mcp_call` | Call MCP tool |
| `%llm_mcp_config` | Manage MCP config |
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
| `%%llm_mcp` | Query with MCP tools |
| `%%llm_vision` | Query with multimodal content |
| **Multimodal** |
| `%llm_paste` | Paste clipboard content |
| `%llm_image` | Include image file/URL |
| `%llm_pdf` | Include PDF as images |
| `%llm_pdf_native` | Upload PDF directly |
| `%llm_media_clear` | Clear attached media |
| `%llm_media_list` | List attached media |
| `%llm_files_list` | List uploaded files |
| `%llm_files_clear` | Clear uploaded files |

---

## Multimodal Content

The LLM Kernel supports multimodal content including images, PDFs, and clipboard content for vision-capable models.

**Important:** Images and PDFs are now added to the conversation context and persist across cells, just like in regular LLM chat interfaces!

### Two Ways to Handle PDFs

1. **%llm_pdf** - Converts PDF pages to images (works with any vision model)
2. **%llm_pdf_native** - Uploads the PDF file directly (for APIs that support native PDF like OpenAI GPT-4.1+, Claude)

### `%llm_paste`
Paste and include clipboard content (image, PDF, or text) in the conversation context.

```python
%llm_paste                    # Paste clipboard content
%llm_paste --show            # Show what's in clipboard
%llm_paste --as-image       # Force treat as image
%llm_paste --as-text        # Force treat as text
```

**Supported clipboard content:**
- **Files (Ctrl+C)** - Copy files in File Explorer with Ctrl+C and paste directly!
- **Images** - Copy an image (screenshot, from web, etc.) and paste
- **PDF paths** - Copy a PDF file path as text and paste
- **Text** - Any other text content

**Example:**
```python
# Cell 1: Copy an image to clipboard, then paste it
%llm_paste
✅ Pasted image (800x600) - added to conversation context

# Cell 2: Ask about it in a different cell!
What's in the image I just pasted?

# Or copy a PDF file with Ctrl+C in File Explorer:
%llm_paste
✅ Pasted PDF 'report.pdf' (2.3 MB) - added to conversation context

# Cell 3: Ask about the PDF
Summarize this PDF for me

# The LLM can see both the image and PDF from previous cells!

# Platform-specific file copying:
# Windows: Select file → Ctrl+C → %llm_paste
# macOS: Select file → Cmd+C → %llm_paste  
# Linux: Select file → Ctrl+C → %llm_paste (requires xclip)
```

### `%llm_image`
Include an image file or URL in the next LLM query.

```python
%llm_image path/to/image.png              # Include local image
%llm_image https://example.com/image.jpg  # Include image from URL
%llm_image --show path/to/image.png       # Preview without including
```

**Example:**
```python
%llm_image diagram.png
Can you explain this diagram?

# Or from URL:
%llm_image https://example.com/chart.png
What trends do you see in this chart?
```

### `%llm_pdf`
Include PDF content in the conversation context as images or text.

```python
%llm_pdf path/to/document.pdf           # Include all pages as images
%llm_pdf --pages 1,3,5 document.pdf     # Include specific pages
%llm_pdf --text document.pdf            # Extract text instead of images
%llm_pdf --show document.pdf            # Preview first page
```

**Example:**
```python
# Cell 1: Add PDF pages as images
%llm_pdf report.pdf --pages 1,2,3
✅ Added 3 page images from PDF

# Cell 2: Ask about them in a different cell!
Summarize the key findings from these pages

# Or extract as text (also persists across cells):
%llm_pdf article.pdf --text
✅ Extracted text from 10 pages

# Later cells can reference the PDF content
What are the main arguments in this paper?
```

### `%llm_pdf_native`
Upload PDF files directly to the conversation (for APIs that support native PDF).

This is the modern approach supported by OpenAI GPT-4.1+, Claude, and other APIs that accept PDF files directly without conversion to images.

```python
%llm_pdf_native document.pdf              # Upload entire PDF
%llm_pdf_native --preview document.pdf    # Preview file info without uploading
```

**Example:**
```python
# Cell 1: Upload a PDF directly
%llm_pdf_native research_paper.pdf
✅ Uploaded PDF 'research_paper.pdf' (2.3 MB) to conversation

# Cell 2: Ask about it in any subsequent cell
What are the key findings in this paper?

# Cell 3: The LLM has full access to the PDF content
Can you extract all the citations from this paper?
```

### `%llm_files_list`
List all files uploaded to the conversation.

```python
%llm_files_list    # Show all uploaded files
```

**Output example:**
```
📁 Uploaded files in conversation:
  1. research_paper.pdf (pdf, 2.3 MB)
      Path: /home/user/documents/research_paper.pdf
  2. data_analysis.pdf (pdf, 1.5 MB)
      Path: /home/user/documents/data_analysis.pdf
```

### `%llm_files_clear`
Clear uploaded files from conversation history.

```python
%llm_files_clear    # Remove all file uploads from conversation
```

### `%llm_media_clear`
Clear multimodal content from cells or conversation history.

```python
%llm_media_clear              # Clear current cell's media
%llm_media_clear all          # Clear all cells' media
%llm_media_clear history      # Clear images from conversation history
```

### `%llm_media_list`
List multimodal content attached to cells.

```python
%llm_media_list               # List all media
%llm_media_list current       # List current cell's media
```

**Output example:**
```
📎 Media in current cell (cell_5):
  1. image - diagram.png
     Size: (1024, 768)
  2. image - clipboard
     Size: (800, 600)
```

### `%%llm_vision` (Cell Magic)
Query a vision-capable LLM with attached images and text.

```python
%%llm_vision
What do you see in these images?

%%llm_vision --model=gpt-4o
Compare and contrast these visualizations
```

**Complete workflow example:**
```python
# Cell 1: Attach multiple images
%llm_image screenshot1.png
%llm_image screenshot2.png
%llm_paste  # Add from clipboard

# Cell 2: Check what's attached
%llm_media_list current

# Cell 3: Query with vision model
%%llm_vision
Can you analyze these UI screenshots and suggest improvements?

# Cell 4: Clear media when done
%llm_media_clear
```

### Supported Vision Models

The following models have vision capabilities:
- **OpenAI**: `gpt-4-vision`, `gpt-4o`, `gpt-4o-mini`
- **Anthropic**: `claude-3-opus`, `claude-3-sonnet`, `claude-3-haiku`
- **Google**: `gemini-pro-vision`, `gemini-1.5-pro`
- **Local**: `llava`, `bakllava` (via Ollama)

### Requirements

For full multimodal support, install optional dependencies:

```bash
# For clipboard support
pip install pyperclip pillow

# For PDF support
pip install pymupdf

# Or install all at once
pip install llm-kernel[multimodal]
```

### Tips for Multimodal Usage

1. **Check model compatibility** - Ensure your active model supports vision before attaching images
2. **Image size** - Large images are automatically resized to fit model limits (max 2048x2048)
3. **Multiple images** - You can attach multiple images to a single query
4. **Context persistence** - Attached media is tied to specific cells
5. **Memory usage** - Clear media after use to free memory with `%llm_media_clear`

---

*For more examples and demos, check out the demo notebooks in the repository.*