import * as vscode from 'vscode';

export class StatusBarManager {
    private statusBarItem: vscode.StatusBarItem;
    private currentModel: string = 'gpt-4o-mini';
    private sessionCost: number = 0;
    private contextSize: number = 0;
    private isKernelActive: boolean = false;
    private queryCount: number = 0;
    private chatModeActive: boolean = false;

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

        // Listen for cell execution completions and parse outputs
        context.subscriptions.push(
            vscode.workspace.onDidChangeNotebookDocument((e) => {
                for (const cellChange of e.cellChanges) {
                    if (cellChange.executionSummary === undefined) continue;
                    if (!cellChange.executionSummary.success) continue;

                    const cellText = cellChange.cell.document.getText().trim();
                    this.parseCellExecution(cellText, cellChange.cell);
                }
            })
        );

        this.updateFromConfig();
        this.show();
    }

    /**
     * Parse cell source and output after execution to sync status bar with kernel state.
     */
    private parseCellExecution(cellSource: string, cell: vscode.NotebookCell): void {
        // Count LLM queries
        if (cellSource.startsWith('%%llm')) {
            this.queryCount++;
        }

        // Parse output text for status data
        const outputText = this.getCellOutputText(cell);
        if (!outputText) return;

        // %llm_status output parsing
        if (cellSource.startsWith('%llm_status')) {
            const modelMatch = outputText.match(/Active Model:\s*(.+)/);
            if (modelMatch) {
                this.currentModel = modelMatch[1].trim();
            }
            const contextMatch = outputText.match(/Context Window Usage:\s*([\d.]+)%/);
            if (contextMatch) {
                // Use percentage for display
            }
            const chatMatch = outputText.match(/Chat Mode:\s*(ON|OFF)/i);
            if (chatMatch) {
                this.chatModeActive = chatMatch[1].toUpperCase() === 'ON';
            }
        }

        // %llm_model output parsing
        if (cellSource.startsWith('%llm_model')) {
            const switchMatch = outputText.match(/Switched to model:\s*(.+)/i) ||
                               outputText.match(/Active model:\s*(.+)/i);
            if (switchMatch) {
                this.currentModel = switchMatch[1].trim();
            }
        }

        // %llm_cost output parsing
        if (cellSource.startsWith('%llm_cost')) {
            const costMatch = outputText.match(/Total.*?\$([\d.]+)/);
            if (costMatch) {
                this.sessionCost = parseFloat(costMatch[1]);
            }
        }

        // Chat mode state
        if (cellSource.startsWith('%llm_chat')) {
            if (cellSource.includes('off')) {
                this.chatModeActive = false;
            } else {
                this.chatModeActive = true;
            }
        }

        this.updateDisplay();
    }

    private getCellOutputText(cell: vscode.NotebookCell): string {
        const outputs = cell.outputs;
        if (!outputs || outputs.length === 0) return '';

        const textParts: string[] = [];
        for (const output of outputs) {
            for (const item of output.items) {
                if (item.mime === 'text/plain' || item.mime === 'text/html' || item.mime === 'text/markdown') {
                    textParts.push(new TextDecoder().decode(item.data));
                }
            }
        }
        return textParts.join('\n');
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
        const costDisplay = this.sessionCost > 0 ? ` | $${this.sessionCost.toFixed(3)}` : '';
        const contextDisplay = this.contextSize > 0 ? ` | ${this.contextSize} cells` : '';
        const chatDisplay = this.chatModeActive ? ' | $(comment-discussion)' : '';

        this.statusBarItem.text = `${statusIcon} ${modelIcon} ${this.getModelDisplayName()}${costDisplay}${contextDisplay}${chatDisplay}`;

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
        if (modelLower.includes('gpt') || modelLower.includes('o3')) {
            return '$(hubot)';
        } else if (modelLower.includes('claude')) {
            return '$(sparkle)';
        } else if (modelLower.includes('llama') || modelLower.includes('ollama')) {
            return '$(home)';
        } else if (modelLower.includes('gemini')) {
            return '$(star)';
        }
        return '$(hubot)';
    }

    private getModelDisplayName(): string {
        const modelMap: Record<string, string> = {
            'gpt-4o': 'GPT-4o',
            'gpt-4o-mini': 'GPT-4o Mini',
            'gpt-4.1': 'GPT-4.1',
            'o3-mini': 'o3-mini',
            'claude-3-sonnet': 'Claude Sonnet',
            'claude-3-haiku': 'Claude Haiku',
            'claude-3-5-sonnet': 'Claude 3.5 Sonnet',
            'gemini-2.5-pro': 'Gemini 2.5 Pro',
            'ollama/llama3': 'Local Llama',
        };
        return modelMap[this.currentModel] || this.currentModel;
    }

    private createTooltip(): vscode.MarkdownString {
        const tooltip = new vscode.MarkdownString();
        tooltip.isTrusted = true;

        tooltip.appendMarkdown(`### LLM Kernel Status\n\n`);
        tooltip.appendMarkdown(`**Model:** ${this.getModelDisplayName()}\n\n`);
        tooltip.appendMarkdown(`**Status:** ${this.isKernelActive ? '$(circle-filled) Active' : '$(circle-outline) Inactive'}\n\n`);

        if (this.queryCount > 0) {
            tooltip.appendMarkdown(`**Queries:** ${this.queryCount}\n\n`);
        }

        if (this.sessionCost > 0) {
            tooltip.appendMarkdown(`**Session Cost:** $${this.sessionCost.toFixed(4)}\n\n`);
        }

        if (this.contextSize > 0) {
            tooltip.appendMarkdown(`**Context Size:** ${this.contextSize} cells\n\n`);
        }

        if (this.chatModeActive) {
            tooltip.appendMarkdown(`**Chat Mode:** ON\n\n`);
        }

        tooltip.appendMarkdown(`---\n\n`);
        tooltip.appendMarkdown(`**Shortcuts:**\n`);
        tooltip.appendMarkdown(`- \`Ctrl+Shift+L\` - Add LLM query\n`);
        tooltip.appendMarkdown(`- \`Ctrl+Shift+M\` - Switch model\n`);
        tooltip.appendMarkdown(`- \`Ctrl+Shift+E\` - Explain cell\n`);
        tooltip.appendMarkdown(`- \`Ctrl+Shift+R\` - Refactor cell\n`);
        tooltip.appendMarkdown(`- \`Ctrl+Shift+T\` - Toggle chat mode\n\n`);
        tooltip.appendMarkdown(`Click to open dashboard`);

        return tooltip;
    }

    private onNotebookChanged(editor: vscode.NotebookEditor | undefined): void {
        if (editor) {
            // Check if this notebook has LLM kernel
            const kernelSpec = editor.notebook.metadata?.kernelspec?.name;
            const hasLLMKernel = kernelSpec?.includes('llm') || false;

            this.setKernelStatus(hasLLMKernel);

            // Count context cells (cells with LLM magic)
            let contextCount = 0;
            for (let i = 0; i < editor.notebook.cellCount; i++) {
                const cell = editor.notebook.cellAt(i);
                if (cell.document.getText().includes('%%llm')) {
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

    resetSession(): void {
        this.sessionCost = 0;
        this.contextSize = 0;
        this.queryCount = 0;
        this.chatModeActive = false;
        this.updateDisplay();
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
        }, 10000);

        return () => {
            clearInterval(interval);
            this.updateDisplay();
        };
    }
}
