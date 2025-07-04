# LLM Kernel Quick Start Guide

Get up and running with the LLM Kernel in 5 minutes!

## 🎯 NEW: Copy & Paste Multimodal Content!

The LLM Kernel now supports seamless file pasting with native uploads:

```python
# Copy any file (Ctrl+C) then:
%llm_paste

# Works with:
# - PDFs (uploaded natively via OpenAI Assistants API)
# - Images (PNG, JPG, etc.)
# - Screenshots
# - Text

# Ask questions in any cell:
What's in this document?
```

**Smart Upload Handling:**
- **OpenAI**: Uses Assistants API for native PDF reading - no conversion needed!
- **Claude**: Embeds PDFs directly in messages
- **Files are cached**: Avoids re-uploading the same file
- **Automatic fallback**: Converts to images if native upload isn't supported
- **Machine-readable output**: LLMs can read and analyze PDF content directly

## 🚀 Installation

### Quick Start for VS Code Users

```bash
# One-line setup
pixi run -e notebook vscode
```

This will:
1. Install the LLM Kernel
2. Open VS Code in the project directory
3. The kernel will be available in any `.ipynb` file

### Option 1: Pixi (Recommended)

```bash
# Clone and install with Pixi
git clone https://github.com/your-org/llm-kernel.git
cd llm-kernel
pixi install
pixi run install-kernel

# Verify installation
pixi run list-kernels
```

### Option 2: Pip

```bash
# Install from PyPI (when published)
pip install llm-kernel

# OR install from source
git clone https://github.com/your-org/llm-kernel.git
cd llm-kernel
pip install -e .

# Install the kernel for Jupyter
llm-kernel-install install

# Verify installation
llm-kernel-install list
```

### Step 3: Set Up API Keys

Create a `.env` file in your project directory:

```bash
# Copy the example file
cp .env.example .env

# Edit with your API keys
nano .env  # or use your favorite editor
```

Add your API keys:
```bash
OPENAI_API_KEY=sk-your-openai-key
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key
DEFAULT_LLM_MODEL=gpt-4o-mini
```

## 🎯 First Steps

### 1. Start Jupyter

```bash
jupyter notebook
# or
jupyter lab
```

### 2. Create a New Notebook

- Click "New" → "LLM Kernel"
- Or change kernel in existing notebook: Kernel → Change Kernel → LLM Kernel

### 3. Try Basic Commands

**List available models:**
```python
%llm_models
```

**Query an LLM:**
```python
%%llm
What is the capital of France?
```

**Compare models:**
```python
%%llm_compare gpt-4o-mini claude-3-haiku
Explain quantum computing in simple terms
```

## 🧠 Context Management

**Pin important cells:**
```python
# In cell 1
import pandas as pd
data = pd.read_csv('my_data.csv')

# Pin this cell so it's always in context
%llm_pin_cell 1
```

**Check context status:**
```python
%llm_status
```

**Configure context strategy:**
```python
%llm_config
```

## 🔧 Configuration

**Interactive configuration panel:**
```python
%llm_config
```

**Switch models:**
```python
%llm_model claude-3-sonnet
```

**Set context strategy:**
```python
%llm_context smart  # or: chronological, dependency, manual
```

## 📊 Advanced Features

**Model-specific queries:**
```python
%%llm_gpt4
Write a Python function to calculate fibonacci numbers

%%llm_claude  
Review the above code for potential improvements
```

**Clear conversation history:**
```python
%llm_clear
```

**Show context visualization:**
```python
%llm_graph
```

## 🛠️ Troubleshooting

**Check dependencies:**
```bash
llm-kernel-install check
```

**Test the installation:**
```bash
python test_kernel.py
```

**Enable debug mode:**
```bash
export LLM_KERNEL_DEBUG=true
jupyter notebook
```

**Common issues:**

1. **Kernel not appearing:** Run `llm-kernel-install install --user`
2. **API key errors:** Check your `.env` file location and format
3. **Import errors:** Install missing dependencies with `pip install -r requirements.txt`

## 📚 Example Workflow

Here's a complete example workflow:

```python
# Cell 1: Setup
import pandas as pd
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('sales_data.csv')
%llm_pin_cell 1  # Pin this setup cell

# Cell 2: Explore data
%%llm
Analyze this sales dataset and suggest interesting patterns to explore:
{data.head()}

# Cell 3: Create visualization based on LLM suggestion
%%llm_gpt4
Create a Python script to visualize the monthly sales trends from this data

# Cell 4: Compare analysis approaches
%%llm_compare claude-3-sonnet gpt-4o
What are the key insights from this sales data analysis?

# Cell 5: Check context and clean up
%llm_status
%llm_clear  # Clear history when done
```

## 🎉 You're Ready!

You now have a powerful LLM-enabled Jupyter environment with:

- ✅ Multi-model access (OpenAI, Anthropic, Google, Ollama)
- ✅ Intelligent context management
- ✅ Cell dependency tracking
- ✅ Interactive configuration
- ✅ Model comparison tools

## 📖 Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Explore advanced magic commands
- Set up project-specific configurations
- Try different context management strategies

Happy coding! 🚀
