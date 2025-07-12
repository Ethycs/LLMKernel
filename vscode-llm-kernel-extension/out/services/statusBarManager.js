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
Object.defineProperty(exports, "__esModule", { value: true });
exports.StatusBarManager = void 0;
const vscode = __importStar(require("vscode"));
class StatusBarManager {
    constructor() {
        this.currentModel = 'gpt-4o-mini';
        this.sessionCost = 0;
        this.contextSize = 0;
        this.isKernelActive = false;
        this.statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
        this.statusBarItem.command = 'llm-kernel.showDashboard';
    }
    initialize(context) {
        context.subscriptions.push(this.statusBarItem);
        // Register command for status bar updates
        context.subscriptions.push(vscode.commands.registerCommand('llm-kernel.updateStatusBar', () => {
            this.updateFromConfig();
        }));
        // Listen for notebook changes
        context.subscriptions.push(vscode.window.onDidChangeActiveNotebookEditor((editor) => {
            this.onNotebookChanged(editor);
        }));
        // Listen for configuration changes
        context.subscriptions.push(vscode.workspace.onDidChangeConfiguration((e) => {
            if (e.affectsConfiguration('llm-kernel')) {
                this.updateFromConfig();
            }
        }));
        this.updateFromConfig();
        this.show();
    }
    updateFromConfig() {
        const config = vscode.workspace.getConfiguration('llm-kernel');
        this.currentModel = config.get('defaultModel', 'gpt-4o-mini');
        this.updateDisplay();
    }
    updateModel(model) {
        this.currentModel = model;
        this.updateDisplay();
    }
    updateCost(cost) {
        this.sessionCost += cost;
        this.updateDisplay();
    }
    updateContextSize(size) {
        this.contextSize = size;
        this.updateDisplay();
    }
    setKernelStatus(isActive) {
        this.isKernelActive = isActive;
        this.updateDisplay();
    }
    updateDisplay() {
        const modelIcon = this.getModelIcon(this.currentModel);
        const statusIcon = this.isKernelActive ? '$(pulse)' : '$(circle-outline)';
        const costDisplay = this.sessionCost > 0 ? ` | 💰 $${this.sessionCost.toFixed(3)}` : '';
        const contextDisplay = this.contextSize > 0 ? ` | 🎯 ${this.contextSize} cells` : '';
        this.statusBarItem.text = `${statusIcon} ${modelIcon} ${this.getModelDisplayName()}${costDisplay}${contextDisplay}`;
        // Update tooltip
        this.statusBarItem.tooltip = this.createTooltip();
        // Update color based on cost
        if (this.sessionCost > 1.0) {
            this.statusBarItem.color = new vscode.ThemeColor('errorForeground');
        }
        else if (this.sessionCost > 0.1) {
            this.statusBarItem.color = new vscode.ThemeColor('warningForeground');
        }
        else {
            this.statusBarItem.color = undefined;
        }
    }
    getModelIcon(model) {
        const modelLower = model.toLowerCase();
        if (modelLower.includes('gpt-4')) {
            return '🧠';
        }
        else if (modelLower.includes('claude')) {
            return '🔮';
        }
        else if (modelLower.includes('local') || modelLower.includes('llama')) {
            return '🏠';
        }
        else if (modelLower.includes('gemini')) {
            return '💎';
        }
        return '🤖';
    }
    getModelDisplayName() {
        const modelMap = {
            'gpt-4o': 'GPT-4o',
            'gpt-4o-mini': 'GPT-4o Mini',
            'claude-3-sonnet': 'Claude Sonnet',
            'claude-3-haiku': 'Claude Haiku',
            'local-llama': 'Local Llama',
            'gemini-pro': 'Gemini Pro'
        };
        return modelMap[this.currentModel] || this.currentModel;
    }
    createTooltip() {
        const tooltip = new vscode.MarkdownString();
        tooltip.isTrusted = true;
        tooltip.appendMarkdown(`### LLM Kernel Status\n\n`);
        tooltip.appendMarkdown(`**Model:** ${this.getModelDisplayName()}\n\n`);
        tooltip.appendMarkdown(`**Status:** ${this.isKernelActive ? '🟢 Active' : '🔴 Inactive'}\n\n`);
        if (this.sessionCost > 0) {
            tooltip.appendMarkdown(`**Session Cost:** $${this.sessionCost.toFixed(4)}\n\n`);
        }
        if (this.contextSize > 0) {
            tooltip.appendMarkdown(`**Context Size:** ${this.contextSize} cells\n\n`);
        }
        tooltip.appendMarkdown(`---\n\n`);
        tooltip.appendMarkdown(`**Shortcuts:**\n`);
        tooltip.appendMarkdown(`- \`Ctrl+Shift+L\` - Add LLM query\n`);
        tooltip.appendMarkdown(`- \`Ctrl+Shift+M\` - Switch model\n`);
        tooltip.appendMarkdown(`- \`Ctrl+Shift+E\` - Explain cell\n`);
        tooltip.appendMarkdown(`- \`Ctrl+Shift+R\` - Refactor cell\n\n`);
        tooltip.appendMarkdown(`Click to open dashboard`);
        return tooltip;
    }
    onNotebookChanged(editor) {
        var _a, _b, _c;
        if (editor) {
            // Check if this notebook has LLM kernel
            const kernelSpec = (_b = (_a = editor.notebook.metadata) === null || _a === void 0 ? void 0 : _a.kernelspec) === null || _b === void 0 ? void 0 : _b.name;
            const hasLLMKernel = (kernelSpec === null || kernelSpec === void 0 ? void 0 : kernelSpec.includes('llm')) || false;
            this.setKernelStatus(hasLLMKernel);
            // Count context cells (cells with LLM metadata)
            let contextCount = 0;
            for (let i = 0; i < editor.notebook.cellCount; i++) {
                const cell = editor.notebook.cellAt(i);
                if (((_c = cell.metadata) === null || _c === void 0 ? void 0 : _c.llm_context) || cell.document.getText().includes('%%llm')) {
                    contextCount++;
                }
            }
            this.updateContextSize(contextCount);
        }
        else {
            this.setKernelStatus(false);
            this.updateContextSize(0);
        }
    }
    show() {
        this.statusBarItem.show();
    }
    hide() {
        this.statusBarItem.hide();
    }
    dispose() {
        this.statusBarItem.dispose();
    }
    // Additional methods for real-time updates
    addTokensUsed(tokens, cost) {
        this.updateCost(cost);
        // Could track tokens separately if needed
        this.updateDisplay();
        // Show warning if cost threshold exceeded
        const config = vscode.workspace.getConfiguration('llm-kernel');
        const threshold = config.get('costThreshold', 0.10);
        const showWarnings = config.get('showCostWarnings', true);
        if (showWarnings && this.sessionCost > threshold && this.sessionCost - cost <= threshold) {
            vscode.window.showWarningMessage(`💰 Session cost exceeded $${threshold.toFixed(2)} threshold (current: $${this.sessionCost.toFixed(3)})`, 'View Dashboard', 'Dismiss').then(choice => {
                if (choice === 'View Dashboard') {
                    vscode.commands.executeCommand('llm-kernel.showDashboard');
                }
            });
        }
    }
    resetSession() {
        this.sessionCost = 0;
        this.contextSize = 0;
        this.updateDisplay();
        vscode.window.showInformationMessage('Session metrics reset');
    }
    // Method to be called when a cell execution completes
    onCellExecutionComplete(cellUri, success, tokens, cost) {
        if (success && tokens && cost) {
            this.addTokensUsed(tokens, cost);
        }
        // Update context size by counting current LLM cells
        const editor = vscode.window.activeNotebookEditor;
        if (editor) {
            this.onNotebookChanged(editor);
        }
    }
    // Animation for active queries
    showActiveQuery() {
        const originalText = this.statusBarItem.text;
        let dotCount = 0;
        const animate = () => {
            const dots = '.'.repeat((dotCount % 3) + 1);
            const spaces = ' '.repeat(3 - dots.length);
            this.statusBarItem.text = originalText + ` ${dots}${spaces}`;
            dotCount++;
        };
        const interval = setInterval(animate, 500);
        // Stop animation after reasonable timeout or when query completes
        setTimeout(() => {
            clearInterval(interval);
            this.updateDisplay();
        }, 10000); // 10 second timeout
        return () => {
            clearInterval(interval);
            this.updateDisplay();
        };
    }
}
exports.StatusBarManager = StatusBarManager;
