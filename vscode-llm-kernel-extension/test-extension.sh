#!/bin/bash

echo "🧪 Testing VS Code Extension in WSL"
echo "=================================="

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

# Launch VS Code with the extension loaded for development
echo "🚀 Launching VS Code with extension in development mode..."
echo "   This will open VS Code on Windows desktop through WSL"
echo "   The extension will be loaded and you can manually test it"

# Open VS Code with extension development
code --extensionDevelopmentPath="$EXTENSION_DIR" --new-window

echo ""
echo "🎯 Manual Testing Instructions:"
echo "================================"
echo "1. VS Code should now be open with your extension loaded"
echo "2. Open Command Palette (Ctrl+Shift+P)"
echo "3. Look for commands starting with 'LLM Kernel'"
echo "4. Create a new Jupyter notebook to test overlay features"
echo "5. Try these commands:"
echo "   - 'LLM Kernel: Toggle Chat Mode'"
echo "   - 'LLM Kernel: Add LLM Cell'"
echo "   - 'LLM Kernel: Toggle Overlay'"
echo ""
echo "🐛 If you see errors:"
echo "- Check the Output panel (View > Output)"
echo "- Look at the Developer Console (Help > Toggle Developer Tools)"
echo ""
echo "✨ Extension testing environment is ready!"