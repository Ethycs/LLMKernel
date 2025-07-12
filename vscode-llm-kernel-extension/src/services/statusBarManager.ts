import * as vscode from 'vscode';

export class StatusBarManager {
    private statusBarItem: vscode.StatusBarItem;
    private currentModel: string = 'gpt-4o-mini';
    private sessionCost: number = 0;
    private contextSize: number = 0;
    private isKernelActive: boolean = false;

    constructor() {
        this.statusBarItem = vscode.window.createStatusBarItem(
            vscode.StatusBarAlignment.Right,
            100
        );
        this.statusBarItem.command = 'llm-kernel.showDashboard';
    }

    initialize(context: vscode.ExtensionContext): void {
        context.subscriptions.push(this.statusBarItem);
        
        // Register command for status bar updates
        context.subscriptions.push(
            vscode.commands.registerCommand('llm-kernel.updateStatusBar', () => {
                this.updateFromConfig();
            })
        );

        // Listen for notebook changes
        context.subscriptions.push(
            vscode.window.onDidChangeActiveNotebookEditor((editor) => {
                this.onNotebookChanged(editor);
            })
        );

        // Listen for configuration changes
        context.subscriptions.push(
            vscode.workspace.onDidChangeConfiguration((e) => {
                if (e.affectsConfiguration('llm-kernel')) {
                    this.updateFromConfig();
                }
            })
        );

        this.updateFromConfig();
        this.show();
    }

    updateFromConfig(): void {
        const config = vscode.workspace.getConfiguration('llm-kernel');
        this.currentModel = config.get<string>('defaultModel', 'gpt-4o-mini');
        this.updateDisplay();
    }

    updateModel(model: string): void {
        this.currentModel = model;
        this.updateDisplay();
    }

    updateCost(cost: number): void {
        this.sessionCost += cost;
        this.updateDisplay();
    }

    updateContextSize(size: number): void {
        this.contextSize = size;
        this.updateDisplay();
    }

    setKernelStatus(isActive: boolean): void {
        this.isKernelActive = isActive;
        this.updateDisplay();
    }

    private updateDisplay(): void {
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
        } else if (this.sessionCost > 0.1) {
            this.statusBarItem.color = new vscode.ThemeColor('warningForeground');
        } else {
            this.statusBarItem.color = undefined;
        }
    }

    private getModelIcon(model: string): string {
        const modelLower = model.toLowerCase();
        if (modelLower.includes('gpt-4')) {
            return '🧠';
        } else if (modelLower.includes('claude')) {
            return '🔮';
        } else if (modelLower.includes('local') || modelLower.includes('llama')) {
            return '🏠';
        } else if (modelLower.includes('gemini')) {
            return '💎';
        }
        return '🤖';
    }

    private getModelDisplayName(): string {
        const modelMap: Record<string, string> = {
            'gpt-4o': 'GPT-4o',
            'gpt-4o-mini': 'GPT-4o Mini',
            'claude-3-sonnet': 'Claude Sonnet',
            'claude-3-haiku': 'Claude Haiku',
            'local-llama': 'Local Llama',
            'gemini-pro': 'Gemini Pro'
        };
        return modelMap[this.currentModel] || this.currentModel;
    }

    private createTooltip(): vscode.MarkdownString {
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

    private onNotebookChanged(editor: vscode.NotebookEditor | undefined): void {
        if (editor) {
            // Check if this notebook has LLM kernel
            const kernelSpec = editor.notebook.metadata?.kernelspec?.name;
            const hasLLMKernel = kernelSpec?.includes('llm') || false;
            
            this.setKernelStatus(hasLLMKernel);
            
            // Count context cells (cells with LLM metadata)
            let contextCount = 0;
            for (let i = 0; i < editor.notebook.cellCount; i++) {
                const cell = editor.notebook.cellAt(i);
                if (cell.metadata?.llm_context || cell.document.getText().includes('%%llm')) {
                    contextCount++;
                }
            }
            this.updateContextSize(contextCount);
        } else {
            this.setKernelStatus(false);
            this.updateContextSize(0);
        }
    }

    show(): void {
        this.statusBarItem.show();
    }

    hide(): void {
        this.statusBarItem.hide();
    }

    dispose(): void {
        this.statusBarItem.dispose();
    }

    // Additional methods for real-time updates
    addTokensUsed(tokens: number, cost: number): void {
        this.updateCost(cost);
        // Could track tokens separately if needed
        this.updateDisplay();
        
        // Show warning if cost threshold exceeded
        const config = vscode.workspace.getConfiguration('llm-kernel');
        const threshold = config.get<number>('costThreshold', 0.10);
        const showWarnings = config.get<boolean>('showCostWarnings', true);
        
        if (showWarnings && this.sessionCost > threshold && this.sessionCost - cost <= threshold) {
            vscode.window.showWarningMessage(
                `💰 Session cost exceeded $${threshold.toFixed(2)} threshold (current: $${this.sessionCost.toFixed(3)})`,
                'View Dashboard',
                'Dismiss'
            ).then(choice => {
                if (choice === 'View Dashboard') {
                    vscode.commands.executeCommand('llm-kernel.showDashboard');
                }
            });
        }
    }

    resetSession(): void {
        this.sessionCost = 0;
        this.contextSize = 0;
        this.updateDisplay();
        vscode.window.showInformationMessage('Session metrics reset');
    }

    // Method to be called when a cell execution completes
    onCellExecutionComplete(cellUri: vscode.Uri, success: boolean, tokens?: number, cost?: number): void {
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
    showActiveQuery(): () => void {
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