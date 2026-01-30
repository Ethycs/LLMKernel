"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || function (mod) {
    if (mod && mod.__esModule) return mod;
    var result = {};
    if (mod != null) for (var k in mod) if (k !== "default" && Object.prototype.hasOwnProperty.call(mod, k)) __createBinding(result, mod, k);
    __setModuleDefault(result, mod);
    return result;
};
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.ChatModeManager = void 0;
const vscode = __importStar(require("vscode"));
class ChatModeManager {
    constructor() {
        this.activeChatModes = new Set();
        this.chatModeIndicators = new Map();
        this.naturalLanguageDetector = new NaturalLanguageDetector();
        this.setupEventListeners();
    }
    setupEventListeners() {
        // Listen for notebook changes
        vscode.window.onDidChangeActiveNotebookEditor((editor) => {
            this.updateChatModeDisplay(editor);
        });
        // Listen for notebook document changes
        vscode.workspace.onDidChangeNotebookDocument((event) => {
            this.handleNotebookChange(event.notebook);
        });
    }
    toggleChatMode(notebook) {
        var _a;
        return __awaiter(this, void 0, void 0, function* () {
            const targetNotebook = notebook || ((_a = vscode.window.activeNotebookEditor) === null || _a === void 0 ? void 0 : _a.notebook);
            if (!targetNotebook) {
                vscode.window.showWarningMessage('No active notebook found');
                return;
            }
            const notebookUri = targetNotebook.uri.toString();
            const isActive = this.activeChatModes.has(notebookUri);
            if (isActive) {
                yield this.disableChatMode(targetNotebook);
            }
            else {
                yield this.enableChatMode(targetNotebook);
            }
        });
    }
    enableChatMode(notebook) {
        return __awaiter(this, void 0, void 0, function* () {
            const notebookUri = notebook.uri.toString();
            // Update notebook metadata
            const edit = new vscode.WorkspaceEdit();
            const metadataEdit = vscode.NotebookEdit.updateNotebookMetadata(Object.assign(Object.assign({}, notebook.metadata), { llm_chat_mode: {
                    enabled: true,
                    enabledAt: new Date().toISOString(),
                    version: '1.0.0'
                } }));
            edit.set(notebook.uri, [metadataEdit]);
            yield vscode.workspace.applyEdit(edit);
            // Track active chat mode
            this.activeChatModes.add(notebookUri);
            // Update visual indicators
            this.showChatModeIndicator(notebook);
            vscode.window.showInformationMessage('💬 Chat mode enabled! Type naturally in cells - no %%llm needed.', 'Show Tips', 'Disable').then(choice => {
                if (choice === 'Show Tips') {
                    this.showChatModeTips();
                }
                else if (choice === 'Disable') {
                    this.disableChatMode(notebook);
                }
            });
        });
    }
    disableChatMode(notebook) {
        return __awaiter(this, void 0, void 0, function* () {
            const notebookUri = notebook.uri.toString();
            // Update notebook metadata
            const edit = new vscode.WorkspaceEdit();
            const metadata = Object.assign({}, notebook.metadata);
            if (metadata.llm_chat_mode) {
                metadata.llm_chat_mode.enabled = false;
                metadata.llm_chat_mode.disabledAt = new Date().toISOString();
            }
            const metadataEdit = vscode.NotebookEdit.updateNotebookMetadata(metadata);
            edit.set(notebook.uri, [metadataEdit]);
            yield vscode.workspace.applyEdit(edit);
            // Remove from active chat modes
            this.activeChatModes.delete(notebookUri);
            // Hide visual indicators
            this.hideChatModeIndicator(notebook);
            vscode.window.showInformationMessage('Chat mode disabled - use %%llm for LLM queries');
        });
    }
    isChatModeActive(notebook) {
        const notebookUri = notebook.uri.toString();
        return this.activeChatModes.has(notebookUri);
    }
    isNaturalLanguageQuery(source, notebook) {
        // Only apply natural language detection if chat mode is active
        if (!this.isChatModeActive(notebook)) {
            return false;
        }
        return this.naturalLanguageDetector.isNaturalLanguage(source);
    }
    showChatModeIndicator(notebook) {
        const notebookUri = notebook.uri.toString();
        let statusBarItem = this.chatModeIndicators.get(notebookUri);
        if (!statusBarItem) {
            statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 150);
            this.chatModeIndicators.set(notebookUri, statusBarItem);
        }
        statusBarItem.text = '$(comment-discussion) Chat Mode';
        statusBarItem.tooltip = new vscode.MarkdownString(`
**💬 Chat Mode Active**

Type naturally in cells without %%llm magic commands.

**Examples:**
- \`What is machine learning?\`
- \`Explain this code above\`
- \`Help me optimize this function\`

**Tips:**
- Still works with code cells
- Use Ctrl+Shift+T to toggle
- Compatible with any kernel

[Disable Chat Mode](command:llm-kernel.disableChatMode)
        `);
        statusBarItem.backgroundColor = new vscode.ThemeColor('statusBarItem.prominentBackground');
        statusBarItem.command = 'llm-kernel.toggleChatMode';
        statusBarItem.show();
    }
    hideChatModeIndicator(notebook) {
        const notebookUri = notebook.uri.toString();
        const statusBarItem = this.chatModeIndicators.get(notebookUri);
        if (statusBarItem) {
            statusBarItem.hide();
            statusBarItem.dispose();
            this.chatModeIndicators.delete(notebookUri);
        }
    }
    updateChatModeDisplay(editor) {
        var _a, _b;
        if (!editor) {
            // Hide all indicators when no editor is active
            this.chatModeIndicators.forEach(item => item.hide());
            return;
        }
        const isChatEnabled = ((_b = (_a = editor.notebook.metadata) === null || _a === void 0 ? void 0 : _a.llm_chat_mode) === null || _b === void 0 ? void 0 : _b.enabled) === true;
        const notebookUri = editor.notebook.uri.toString();
        if (isChatEnabled && !this.activeChatModes.has(notebookUri)) {
            // Restore chat mode state
            this.activeChatModes.add(notebookUri);
            this.showChatModeIndicator(editor.notebook);
        }
        else if (!isChatEnabled && this.activeChatModes.has(notebookUri)) {
            // Clean up chat mode state
            this.activeChatModes.delete(notebookUri);
            this.hideChatModeIndicator(editor.notebook);
        }
        // Show/hide indicators based on active editor
        this.chatModeIndicators.forEach((item, uri) => {
            if (uri === notebookUri) {
                item.show();
            }
            else {
                item.hide();
            }
        });
    }
    handleNotebookChange(notebook) {
        var _a, _b;
        const isChatEnabled = ((_b = (_a = notebook.metadata) === null || _a === void 0 ? void 0 : _a.llm_chat_mode) === null || _b === void 0 ? void 0 : _b.enabled) === true;
        const notebookUri = notebook.uri.toString();
        if (isChatEnabled !== this.activeChatModes.has(notebookUri)) {
            this.updateChatModeDisplay(vscode.window.activeNotebookEditor);
        }
    }
    showChatModeTips() {
        const panel = vscode.window.createWebviewPanel('chatModeTips', 'Chat Mode Tips', vscode.ViewColumn.Beside, {});
        panel.webview.html = `
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body { 
                        font-family: var(--vscode-font-family); 
                        padding: 20px; 
                        line-height: 1.6;
                    }
                    h1 { color: var(--vscode-foreground); }
                    .tip { 
                        background: var(--vscode-textBlockQuote-background); 
                        padding: 15px; 
                        margin: 15px 0; 
                        border-radius: 4px; 
                        border-left: 4px solid var(--vscode-textLink-foreground);
                    }
                    .example { 
                        background: var(--vscode-textCodeBlock-background); 
                        padding: 10px; 
                        margin: 10px 0; 
                        border-radius: 4px; 
                        font-family: var(--vscode-editor-font-family);
                    }
                    .shortcut { 
                        background: var(--vscode-badge-background); 
                        color: var(--vscode-badge-foreground); 
                        padding: 2px 6px; 
                        border-radius: 3px; 
                        font-family: monospace;
                    }
                </style>
            </head>
            <body>
                <h1>💬 Chat Mode Tips</h1>
                
                <div class="tip">
                    <h3>🗣️ Natural Conversation</h3>
                    <p>With chat mode enabled, just type questions naturally:</p>
                    <div class="example">What is machine learning?</div>
                    <div class="example">Explain the code in the cell above</div>
                    <div class="example">How can I optimize this function?</div>
                </div>
                
                <div class="tip">
                    <h3>🔄 Seamless Integration</h3>
                    <p>Chat mode works alongside regular code:</p>
                    <div class="example">
                        # Regular Python code still works<br>
                        import pandas as pd<br>
                        df = pd.read_csv('data.csv')<br><br>
                        # Natural language in next cell<br>
                        Analyze this dataset and suggest visualizations
                    </div>
                </div>
                
                <div class="tip">
                    <h3>⌨️ Quick Controls</h3>
                    <ul>
                        <li><span class="shortcut">Ctrl+Shift+T</span> - Toggle chat mode</li>
                        <li><span class="shortcut">Ctrl+Shift+L</span> - Add LLM query (works in any mode)</li>
                        <li><span class="shortcut">Ctrl+Shift+M</span> - Switch LLM model</li>
                    </ul>
                </div>
                
                <div class="tip">
                    <h3>🎯 Smart Detection</h3>
                    <p>The extension automatically detects:</p>
                    <ul>
                        <li>Natural language questions vs code</li>
                        <li>Context from surrounding cells</li>
                        <li>When to use LLM vs execute code</li>
                    </ul>
                </div>
                
                <div class="tip">
                    <h3>🔒 Privacy & Control</h3>
                    <ul>
                        <li>Chat mode is per-notebook</li>
                        <li>Hidden cells (%%hide) stay hidden</li>
                        <li>Toggle off anytime with no impact</li>
                        <li>Works with any kernel (Python, R, Julia, etc.)</li>
                    </ul>
                </div>
                
                <h2>🚀 Getting Started</h2>
                <ol>
                    <li>Open any notebook</li>
                    <li>Click the chat mode toggle or press <span class="shortcut">Ctrl+Shift+T</span></li>
                    <li>Start typing questions naturally in cells</li>
                    <li>Mix with regular code as needed</li>
                </ol>
                
                <p><strong>Tip:</strong> Try asking "Explain the code above" after running some Python - the LLM will see your code and variables!</p>
            </body>
            </html>
        `;
    }
    dispose() {
        this.chatModeIndicators.forEach(item => item.dispose());
        this.chatModeIndicators.clear();
        this.activeChatModes.clear();
    }
}
exports.ChatModeManager = ChatModeManager;
class NaturalLanguageDetector {
    constructor() {
        this.codePatterns = [
            // Common programming constructs
            /^(import|from|def|class|if|for|while|try|catch|function|var|let|const)\s/i,
            /^(library|require|using|#include|<%|%>)\s*\(/i,
            // Assignment patterns
            /^\s*\w+\s*[=<-]\s*/,
            // Function calls
            /^\s*\w+\s*\(/,
            // Magic commands (should be handled separately)
            /^%%?\w+/,
            // Comments only
            /^\s*[#\/\*]/,
            // Brackets/braces suggesting code structure
            /^\s*[{\[]/,
            // Shell commands
            /^!\s*\w+/,
            // Mathematical expressions
            /^\s*[\d\w\s+\-*/=()]+\s*$/
        ];
        this.naturalLanguageIndicators = [
            // Question words
            /^(what|how|why|when|where|who|which|can|could|would|should|is|are|do|does|did)\s+/i,
            // Conversational starters
            /^(please|help|explain|show|tell|describe|analyze|compare|summarize)\s+/i,
            // Complete sentences with subjects and verbs
            /\b(i|you|we|they|it|this|that)\s+(am|is|are|was|were|have|has|had|will|would|could|should|can|do|does|did)\b/i,
            // Common natural language patterns
            /\b(the|a|an)\s+\w+\s+(is|are|was|were)\b/i,
            // Requests and questions
            /\?$|^(could you|can you|would you|please)/i
        ];
    }
    isNaturalLanguage(source) {
        const trimmed = source.trim();
        // Empty or very short text
        if (trimmed.length < 3) {
            return false;
        }
        // Definitely code if matches programming patterns
        if (this.codePatterns.some(pattern => pattern.test(trimmed))) {
            return false;
        }
        // Likely natural language if matches conversational patterns
        if (this.naturalLanguageIndicators.some(pattern => pattern.test(trimmed))) {
            return true;
        }
        // Heuristic: natural language tends to have more words and less punctuation
        return this.scoreAsNaturalLanguage(trimmed) > 0.6;
    }
    scoreAsNaturalLanguage(text) {
        let score = 0;
        // Length factor (longer text more likely to be natural language)
        if (text.length > 20)
            score += 0.2;
        if (text.length > 50)
            score += 0.2;
        // Word count factor
        const words = text.split(/\s+/).length;
        if (words > 3)
            score += 0.2;
        if (words > 6)
            score += 0.2;
        // Punctuation ratio (natural language has more spaces, less symbols)
        const spaceRatio = (text.match(/\s/g) || []).length / text.length;
        if (spaceRatio > 0.15)
            score += 0.3;
        // Symbol density (code has more symbols)
        const symbolRatio = (text.match(/[{}[\]()=<>!&|+\-*/]/g) || []).length / text.length;
        if (symbolRatio < 0.1)
            score += 0.3;
        // Contains question mark or common words
        if (text.includes('?'))
            score += 0.4;
        if (/\b(what|how|why|please|help|explain)\b/i.test(text))
            score += 0.4;
        return Math.min(score, 1.0);
    }
}
