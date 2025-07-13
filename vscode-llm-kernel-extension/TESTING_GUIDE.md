# 🧪 VS Code Extension Testing Guide

## ✅ Fixed Issues

Based on the canonical VS Code extension development process, I've fixed several common issues:

1. **✅ Added missing `tasks.json`** - Required for the build process
2. **✅ Updated `launch.json`** - Added `runtimeExecutable` and proper task references  
3. **✅ Fixed activation events** - Changed to `"*"` for immediate activation
4. **✅ All TypeScript compiles successfully**

## 🚀 Canonical Testing Process

### Step 1: Open Extension in VS Code
```bash
# Make sure you're in the extension directory
cd /mnt/f/Keytone/Documents/GitHub/LLMKernel/vscode-llm-kernel-extension

# Open VS Code in this directory (IMPORTANT!)
code .
```

### Step 2: Compile and Debug
1. **Press F5** or go to **Run and Debug** panel (Ctrl+Shift+D)
2. Select **"Run Extension"** from the dropdown
3. Click the **▶️ Play button** or press **F5**

This will:
- Automatically compile TypeScript (`npm run compile`)
- Open a new **Extension Development Host** window
- Load your extension in the new window

### Step 3: Verify Extension Loaded
In the **Extension Development Host** window:

1. **Check Output Panel:**
   - View → Output
   - Select "LLM Kernel" from dropdown
   - Look for activation logs

2. **Check Command Palette:**
   - Press `Ctrl+Shift+P`
   - Type "LLM Kernel"
   - You should see all these commands:
     - ✅ LLM Kernel: Toggle Chat Mode
     - ✅ LLM Kernel: Enable Chat Mode  
     - ✅ LLM Kernel: Disable Chat Mode
     - ✅ LLM Kernel: Add LLM Cell
     - ✅ LLM Kernel: Create New LLM Notebook
     - ✅ LLM Kernel: Switch Model
     - ✅ LLM Kernel: Show Dashboard

### Step 4: Test Core Features

#### 🗣️ Chat Mode Testing
1. Open a Jupyter notebook in the Extension Development Host
2. Run command: **"LLM Kernel: Toggle Chat Mode"**
3. Check status bar - should show "💬 Chat Mode" indicator
4. Create a new cell and type: `What is machine learning?`
5. The extension should detect this as natural language

#### 🔄 LLM Overlay Testing  
1. Run command: **"LLM Kernel: Enable Overlay"**
2. Check status bar - should show "🤖 LLM Overlay" indicator
3. Create a cell with: `%%llm` magic command
4. Extension should intercept this for LLM processing

#### 📝 LLM Cell Testing
1. Run command: **"LLM Kernel: Add LLM Cell"**
2. New cell should appear with "Ask me anything!" text
3. Cell should have LLM metadata

### Step 5: Making Changes and Reloading

**🔄 IMPORTANT: After making code changes:**

1. **Save your changes** in the main VS Code window
2. **Recompile** (automatic if using watch mode, or run `npm run compile`)
3. **Reload Extension Development Host:** Press `Ctrl+Shift+F5` in the Extension Development Host window
4. **Test your changes**

## 🐛 Debugging Issues

### Extension Not Loading
1. Check **Developer Tools:** Help → Toggle Developer Tools
2. Look at **Console** tab for errors
3. Check **Output panel** for compilation errors

### Commands Not Appearing
1. Verify `activationEvents` is set to `"*"` in package.json
2. Check that `contributes.commands` lists all commands
3. Ensure TypeScript compilation succeeded

### Changes Not Reflected
1. **Always reload** with `Ctrl+Shift+F5` after changes
2. Check if TypeScript compiled successfully
3. Verify you're testing in the Extension Development Host, not main VS Code

## 📂 Project Structure Verification

```
vscode-llm-kernel-extension/
├── .vscode/
│   ├── launch.json        ✅ Fixed with runtimeExecutable
│   ├── tasks.json         ✅ Added for build process
│   └── settings.json      ✅ WSL-specific settings
├── src/
│   ├── extension.ts       ✅ Main extension entry point
│   ├── services/
│   │   ├── chatModeManager.ts     ✅ Chat mode functionality
│   │   └── llmOverlayManager.ts   ✅ Overlay system
│   └── test/              ✅ Complete test suite
├── out/                   ✅ Compiled JavaScript (after npm run compile)
└── package.json           ✅ Fixed activationEvents
```

## 🎯 Expected Results

When working correctly, you should see:

1. **Status Bar Indicators:**
   - 💬 Chat Mode (when enabled)
   - 🤖 LLM Overlay (when enabled)

2. **Command Palette:**
   - All "LLM Kernel" commands available

3. **Natural Language Detection:**
   - Questions like "What is X?" trigger LLM processing
   - Code like `import pandas` stays as code

4. **Console Output:**
   - No red error messages
   - Extension activation logs

## 🚨 Common Canonical Issues

1. **Not opening VS Code in extension root** → Commands: `cd extension-dir && code .`
2. **Missing build task** → Fixed: Added tasks.json
3. **Wrong activation events** → Fixed: Changed to `"*"`
4. **Forgetting to reload** → Always use `Ctrl+Shift+F5`
5. **Testing in wrong window** → Use Extension Development Host window

## 💡 Pro Tips

- Use **watch mode**: `npm run watch` for automatic compilation
- Use **Developer Tools** for debugging JavaScript
- **Reload frequently** when making changes
- Check **Output panel** for extension logs
- Test in **clean Extension Development Host** environment

---

✨ **The extension is now properly configured for canonical VS Code extension development!**