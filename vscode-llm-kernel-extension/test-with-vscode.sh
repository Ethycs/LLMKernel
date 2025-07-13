#!/bin/bash

echo "🧪 Testing VS Code Extension with Extension Development Host"
echo "=========================================================="

# Compile the extension
echo "📦 Compiling TypeScript..."
npm run compile

if [ $? -ne 0 ]; then
    echo "❌ Compilation failed"
    exit 1
fi

echo "✅ Compilation successful"

# Get the current directory (extension root)
EXTENSION_DIR=$(pwd)

echo "📁 Extension directory: $EXTENSION_DIR"

# Create a test workspace
mkdir -p test-workspace
cd test-workspace

# Create a test notebook for testing
cat > test-notebook.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print('Hello, this is a test notebook for LLM Kernel extension!')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Test LLM Extension\n",
    "\n",
    "This notebook is for testing the LLM Kernel extension features:\n",
    "\n",
    "1. Chat Mode Toggle\n",
    "2. LLM Overlay\n",
    "3. Natural Language Detection"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

cd ..

echo "🚀 Launching VS Code Extension Development Host..."
echo ""
echo "VS Code will open with your extension loaded for testing."
echo "The extension will run in Development Mode."
echo ""

# Launch VS Code with extension development
code --extensionDevelopmentPath="$EXTENSION_DIR" test-workspace/test-notebook.ipynb

echo ""
echo "📋 Testing Checklist:"
echo "==================="
echo ""
echo "✅ Extension Loading:"
echo "   1. Check VS Code Output panel for any errors"
echo "   2. Look for 'LLM Kernel' in the Output dropdown"
echo "   3. No red error messages should appear"
echo ""
echo "✅ Commands Available:"
echo "   1. Open Command Palette (Ctrl+Shift+P)"
echo "   2. Type 'LLM Kernel' - you should see these commands:"
echo "      - LLM Kernel: Toggle Chat Mode"
echo "      - LLM Kernel: Enable Chat Mode"
echo "      - LLM Kernel: Disable Chat Mode"
echo "      - LLM Kernel: Toggle Overlay"
echo "      - LLM Kernel: Add LLM Cell"
echo "      - LLM Kernel: Create New LLM Notebook"
echo ""
echo "✅ Chat Mode Testing:"
echo "   1. Run 'LLM Kernel: Toggle Chat Mode'"
echo "   2. Look for status bar indicator showing chat mode is active"
echo "   3. Create a new cell and type: 'What is machine learning?'"
echo "   4. The extension should detect this as natural language"
echo ""
echo "✅ Overlay Testing:"
echo "   1. Run 'LLM Kernel: Toggle Overlay'"
echo "   2. Look for overlay status in status bar"
echo "   3. Create a cell with: '%%llm' magic command"
echo ""
echo "✅ LLM Cell Testing:"
echo "   1. Run 'LLM Kernel: Add LLM Cell'"
echo "   2. New cell should be created with 'Ask me anything!' text"
echo "   3. Cell should have LLM metadata"
echo ""
echo "🐛 Debugging:"
echo "   - If errors occur, check Help > Toggle Developer Tools"
echo "   - Look at Console tab for JavaScript errors"
echo "   - Check VS Code Output panel for extension logs"
echo ""
echo "✨ Extension is ready for testing!"
echo "   Close VS Code when done testing."