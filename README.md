# LLM Kernel

A Jupyter kernel with LiteLLM integration and intelligent context management for seamless multi-LLM workflows.

## Features

🤖 **Multi-LLM Support**: Access 100+ LLM providers through LiteLLM (OpenAI, Anthropic, Google, Ollama, and more)

🧠 **Intelligent Context Management**: 
- Cell dependency tracking
- Execution order awareness
- Automatic context optimization
- Smart dialogue pruning

📊 **Rich Jupyter Integration**:
- Native magic commands
- Interactive widgets
- HTML visualizations
- Model comparison tools

🔧 **Flexible Configuration**:
- Environment variable support (.env files)
- Hierarchical configuration (user/project/notebook)
- Runtime parameter overrides

⚡ **Performance Optimized**:
- Parallel model queries
- Async processing
- Token budget management
- Context caching

## Installation

### Quick Install (Pip)

```bash
pip install llm-kernel
llm-kernel-install install
```

### Quick Install (Pixi - Recommended)

```bash
git clone https://github.com/your-org/llm-kernel.git
cd llm-kernel
pixi install
pixi run install-kernel
```

### Development Install (Pip)

```bash
git clone https://github.com/your-org/llm-kernel.git
cd llm-kernel
pip install -e .
python -m llm_kernel.install install
```

### Development Install (Pixi - Recommended)

```bash
git clone https://github.com/your-org/llm-kernel.git
cd llm-kernel
pixi install --environment dev
pixi run install-kernel
```

## Setup

### 1. Configure API Keys

Create a `.env` file in your project directory:

```bash
# OpenAI
OPENAI_API_KEY=sk-your-openai-key

# Anthropic
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key

# Google
GOOGLE_API_KEY=your-google-key

# Optional: Default settings
DEFAULT_LLM_MODEL=gpt-4o-mini
LLM_KERNEL_MAX_TOKENS=4000
LLM_KERNEL_CONTEXT_STRATEGY=smart
```

### 2. Start Jupyter

```bash
jupyter notebook
# or
jupyter lab
```

### 3. Select LLM Kernel

Create a new notebook and select "LLM Kernel" from the kernel dropdown.

## Usage

### Basic LLM Queries

```python
# Query the active model
%%llm
What is the capital of France?
```

```python
# Query a specific model
%%llm_gpt4
Explain quantum computing in simple terms
```

```python
# Compare multiple models
%%llm_compare gpt-4o claude-3-sonnet
What are the pros and cons of renewable energy?
```

### Model Management

```python
# List available models
%llm_models

# Switch active model
%llm_model claude-3-sonnet

# Show current status
%llm_status
```

### Context Management

```python
# Pin important cells for context
%llm_pin_cell 5

# Set context strategy
%llm_context smart  # or: chronological, dependency, manual

# Show interactive configuration
%llm_config
```

### Advanced Features

```python
# Clear conversation history
%llm_clear

# Show context visualization
%llm_graph

# Export/import context
%llm_export_context my_session.json
%llm_import_context my_session.json
```

## Magic Commands Reference

### Line Magics

| Command | Description |
|---------|-------------|
| `%llm_models` | List available models |
| `%llm_model <name>` | Switch active model |
| `%llm_status` | Show kernel status |
| `%llm_clear` | Clear conversation history |
| `%llm_config` | Show configuration panel |
| `%llm_context <strategy>` | Set context strategy |
| `%llm_pin_cell <id>` | Pin cell for context |
| `%llm_unpin_cell <id>` | Unpin cell |

### Cell Magics

| Command | Description |
|---------|-------------|
| `%%llm` | Query active model |
| `%%llm_gpt4` | Query GPT-4 |
| `%%llm_claude` | Query Claude |
| `%%llm_compare <models>` | Compare multiple models |

## Context Management Strategies

### Smart (Default)
Combines multiple factors:
- Cell dependencies
- Recency
- User importance (pinned cells)
- Content relevance

### Chronological
Uses most recent exchanges in execution order.

### Dependency
Includes cells that define variables/functions used in current context.

### Manual
Only includes explicitly pinned cells.

## Configuration

### Environment Variables

```bash
LLM_KERNEL_DEFAULT_MODEL=gpt-4o-mini
LLM_KERNEL_CONTEXT_STRATEGY=smart
LLM_KERNEL_MAX_TOKENS=4000
LLM_KERNEL_MAX_CELLS=20
LLM_KERNEL_AUTO_PRUNE=true
LLM_KERNEL_LOG_LEVEL=INFO
```

### Configuration Files

**User Config** (`~/.llm-kernel/config.json`):
```json
{
  "default_model": "gpt-4o-mini",
  "context_strategy": "smart",
  "max_context_tokens": 4000,
  "auto_prune": true
}
```

**Project Config** (`.llm-kernel.json`):
```json
{
  "default_model": "claude-3-sonnet",
  "context_strategy": "dependency",
  "max_context_tokens": 6000,
  "project_specific_setting": "value"
}
```

## VS Code Integration

The kernel works seamlessly with VS Code's Jupyter extension:

1. Install the kernel: `llm-kernel-install install`
2. Open a `.ipynb` file in VS Code
3. Select "LLM Kernel" from the kernel picker
4. All magic commands and widgets work natively

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Jupyter UI   │◄──►│   LLM Kernel     │◄──►│    LiteLLM      │
│                 │    │                  │    │                 │
│ • Magic Commands│    │ • Context Mgmt   │    │ • OpenAI        │
│ • Widgets       │    │ • Dep Tracking   │    │ • Anthropic     │
│ • Visualizations│    │ • Dialogue Prune │    │ • Google        │
└─────────────────┘    │ • Config Mgmt    │    │ • Ollama        │
                       └──────────────────┘    │ • 100+ more     │
                                               └─────────────────┘
```

## Development

### Setup Development Environment

```bash
git clone https://github.com/your-org/llm-kernel.git
cd llm-kernel
pip install -e ".[dev]"
```

### Run Tests

```bash
# Run all tests
pytest test_kernel.py -v

# Run specific test classes
pytest test_kernel.py::TestContextManager -v

# Run with coverage (if installed)
pytest test_kernel.py --cov=llm_kernel

# Run only unit tests (skip integration tests)
pytest test_kernel.py -m "not integration" -v
```

### Code Formatting

```bash
black llm_kernel/
flake8 llm_kernel/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Troubleshooting

### Common Issues

**Kernel not appearing in Jupyter:**
```bash
llm-kernel-install list
llm-kernel-install install --user
```

**API key errors:**
- Check your `.env` file is in the correct location
- Verify API keys are valid
- Check environment variable names

**Import errors:**
```bash
llm-kernel-install check
pip install -r requirements.txt
```

### Debug Mode

Enable debug logging:
```bash
export LLM_KERNEL_DEBUG=true
jupyter notebook
```

## Roadmap

- [ ] Enhanced semantic similarity with embeddings
- [ ] Visual context dependency graphs
- [ ] Notebook-wide context search
- [ ] Custom model fine-tuning integration
- [ ] Collaborative context sharing
- [ ] Plugin system for custom providers

## Support

- 📖 [Documentation](https://github.com/your-org/llm-kernel/wiki)
- 🐛 [Issue Tracker](https://github.com/your-org/llm-kernel/issues)
- 💬 [Discussions](https://github.com/your-org/llm-kernel/discussions)
