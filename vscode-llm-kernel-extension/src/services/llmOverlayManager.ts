import * as vscode from 'vscode';
import { ChatModeManager } from './chatModeManager';

export class LLMOverlayManager {
    private activeOverlays = new Set<string>();
    private executionInterceptor: ExecutionInterceptor;
    private overlayIndicator: OverlayIndicator;
    private chatModeManager: ChatModeManager;

    constructor() {
        this.executionInterceptor = new ExecutionInterceptor();
        this.overlayIndicator = new OverlayIndicator();
        this.chatModeManager = new ChatModeManager();
        this.setupEventListeners();
    }

    private setupEventListeners(): void {
        // Listen for notebook changes
        vscode.window.onDidChangeActiveNotebookEditor((editor) => {
            this.updateOverlayForNotebook(editor);
        });

        // Listen for notebook document changes
        vscode.workspace.onDidChangeNotebookDocument((event) => {
            this.updateOverlayState(event.notebook);
        });
    }

    async toggleOverlay(notebook?: vscode.NotebookDocument): Promise<void> {
        const targetNotebook = notebook || vscode.window.activeNotebookEditor?.notebook;
        
        if (!targetNotebook) {
            vscode.window.showWarningMessage('No active notebook found');
            return;
        }

        const notebookUri = targetNotebook.uri.toString();
        const isActive = this.activeOverlays.has(notebookUri);

        if (isActive) {
            await this.disableOverlay(targetNotebook);
        } else {
            await this.enableOverlay(targetNotebook);
        }
    }

    async enableOverlay(notebook: vscode.NotebookDocument): Promise<void> {
        const notebookUri = notebook.uri.toString();
        
        // Update notebook metadata
        const edit = new vscode.WorkspaceEdit();
        const metadataEdit = vscode.NotebookEdit.updateNotebookMetadata({
            ...notebook.metadata,
            llm_overlay: {
                enabled: true,
                version: '1.0.0',
                enabledAt: new Date().toISOString()
            }
        });
        edit.set(notebook.uri, [metadataEdit]);
        await vscode.workspace.applyEdit(edit);

        // Track active overlay
        this.activeOverlays.add(notebookUri);

        // Set up execution interception
        this.executionInterceptor.enableForNotebook(notebook);

        // Update visual indicators
        this.overlayIndicator.showOverlayActive(notebook);

        vscode.window.showInformationMessage(
            '🤖 LLM Overlay enabled! Use %%llm for AI queries in any cell.',
            'Show Guide'
        ).then(choice => {
            if (choice === 'Show Guide') {
                this.showOverlayGuide();
            }
        });
    }

    async disableOverlay(notebook: vscode.NotebookDocument): Promise<void> {
        const notebookUri = notebook.uri.toString();

        // Update notebook metadata
        const edit = new vscode.WorkspaceEdit();
        const metadata = { ...notebook.metadata };
        if (metadata.llm_overlay) {
            metadata.llm_overlay.enabled = false;
            metadata.llm_overlay.disabledAt = new Date().toISOString();
        }
        const metadataEdit = vscode.NotebookEdit.updateNotebookMetadata(metadata);
        edit.set(notebook.uri, [metadataEdit]);
        await vscode.workspace.applyEdit(edit);

        // Remove from active overlays
        this.activeOverlays.delete(notebookUri);

        // Disable execution interception
        this.executionInterceptor.disableForNotebook(notebook);

        // Update visual indicators
        this.overlayIndicator.hideOverlayActive(notebook);

        vscode.window.showInformationMessage('LLM Overlay disabled');
    }

    isOverlayActive(notebook: vscode.NotebookDocument): boolean {
        const notebookUri = notebook.uri.toString();
        return this.activeOverlays.has(notebookUri);
    }

    private updateOverlayForNotebook(editor?: vscode.NotebookEditor): void {
        if (!editor) return;

        const isOverlayEnabled = editor.notebook.metadata?.llm_overlay?.enabled === true;
        const notebookUri = editor.notebook.uri.toString();

        if (isOverlayEnabled && !this.activeOverlays.has(notebookUri)) {
            // Restore overlay state
            this.activeOverlays.add(notebookUri);
            this.executionInterceptor.enableForNotebook(editor.notebook);
            this.overlayIndicator.showOverlayActive(editor.notebook);
        } else if (!isOverlayEnabled && this.activeOverlays.has(notebookUri)) {
            // Clean up overlay state
            this.activeOverlays.delete(notebookUri);
            this.executionInterceptor.disableForNotebook(editor.notebook);
            this.overlayIndicator.hideOverlayActive(editor.notebook);
        }
    }

    private updateOverlayState(notebook: vscode.NotebookDocument): void {
        const isOverlayEnabled = notebook.metadata?.llm_overlay?.enabled === true;
        const notebookUri = notebook.uri.toString();

        if (isOverlayEnabled !== this.activeOverlays.has(notebookUri)) {
            this.updateOverlayForNotebook(vscode.window.activeNotebookEditor);
        }
    }

    private showOverlayGuide(): void {
        const panel = vscode.window.createWebviewPanel(
            'llmOverlayGuide',
            'LLM Overlay Guide',
            vscode.ViewColumn.Beside,
            {}
        );

        panel.webview.html = `
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body { font-family: var(--vscode-font-family); padding: 20px; }
                    h1 { color: var(--vscode-foreground); }
                    .example { background: var(--vscode-textBlockQuote-background); padding: 10px; margin: 10px 0; border-radius: 4px; }
                    .shortcut { background: var(--vscode-badge-background); color: var(--vscode-badge-foreground); padding: 2px 6px; border-radius: 3px; }
                </style>
            </head>
            <body>
                <h1>🤖 LLM Overlay Guide</h1>
                
                <h2>What is LLM Overlay?</h2>
                <p>LLM Overlay adds AI capabilities to ANY notebook without changing kernels or workflows.</p>
                
                <h2>How to Use</h2>
                
                <h3>In any cell, use LLM magic commands:</h3>
                <div class="example">
                    <strong>Python/Jupyter style:</strong><br>
                    <code>%%llm --model=gpt-4o<br>
                    Explain this dataset and suggest visualizations</code>
                </div>
                
                <div class="example">
                    <strong>Comment style (any language):</strong><br>
                    <code># @llm model=claude-3-sonnet<br>
                    # Optimize this R code for performance</code>
                </div>
                
                <h3>Keyboard Shortcuts:</h3>
                <ul>
                    <li><span class="shortcut">Ctrl+Shift+L</span> - Add LLM query to current cell</li>
                    <li><span class="shortcut">Ctrl+Shift+E</span> - Explain current cell</li>
                    <li><span class="shortcut">Ctrl+Shift+R</span> - Refactor current cell</li>
                    <li><span class="shortcut">Ctrl+Shift+M</span> - Switch LLM model</li>
                </ul>
                
                <h2>Key Features</h2>
                <ul>
                    <li><strong>Works with any kernel</strong> - Python, R, Julia, JavaScript, etc.</li>
                    <li><strong>Works without kernel</strong> - Pure LLM mode for analysis</li>
                    <li><strong>Context aware</strong> - AI sees your variables, imports, functions</li>
                    <li><strong>Toggle on/off</strong> - Enable only when needed</li>
                </ul>
                
                <h2>Examples</h2>
                
                <div class="example">
                    <strong>Data Analysis (Python):</strong><br>
                    <code>import pandas as pd<br>
                    df = pd.read_csv('sales.csv')<br><br>
                    %%llm<br>
                    Analyze this sales dataset and create visualizations</code>
                </div>
                
                <div class="example">
                    <strong>Statistical Analysis (R):</strong><br>
                    <code>library(ggplot2)<br>
                    data <- read.csv('experiment.csv')<br><br>
                    # @llm<br>
                    # Suggest appropriate statistical tests for this experimental data</code>
                </div>
                
                <div class="example">
                    <strong>Pure LLM (No Kernel):</strong><br>
                    <code>%%llm --model=gpt-4o<br>
                    Write a machine learning pipeline for text classification</code>
                </div>
            </body>
            </html>
        `;
    }

    // Chat mode methods - delegate to ChatModeManager
    async toggleChatMode(notebook?: vscode.NotebookDocument): Promise<void> {
        return this.chatModeManager.toggleChatMode(notebook);
    }

    async enableChatMode(notebook: vscode.NotebookDocument): Promise<void> {
        return this.chatModeManager.enableChatMode(notebook);
    }

    async disableChatMode(notebook: vscode.NotebookDocument): Promise<void> {
        return this.chatModeManager.disableChatMode(notebook);
    }

    isChatModeActive(notebook: vscode.NotebookDocument): boolean {
        return this.chatModeManager.isChatModeActive(notebook);
    }

    isNaturalLanguageQuery(source: string, notebook: vscode.NotebookDocument): boolean {
        return this.chatModeManager.isNaturalLanguageQuery(source, notebook);
    }

    dispose(): void {
        this.executionInterceptor.dispose();
        this.overlayIndicator.dispose();
        this.chatModeManager.dispose();
    }
}

class ExecutionInterceptor {
    private interceptedNotebooks = new Set<string>();
    private originalControllers = new Map<string, vscode.NotebookController>();

    enableForNotebook(notebook: vscode.NotebookDocument): void {
        const notebookUri = notebook.uri.toString();
        if (this.interceptedNotebooks.has(notebookUri)) return;

        this.interceptedNotebooks.add(notebookUri);
        
        // Set up execution interception
        // This would hook into VS Code's execution pipeline
        // For now, we'll use the controller approach
    }

    disableForNotebook(notebook: vscode.NotebookDocument): void {
        const notebookUri = notebook.uri.toString();
        this.interceptedNotebooks.delete(notebookUri);
    }

    async interceptExecution(
        cell: vscode.NotebookCell,
        originalExecution: () => Promise<void>
    ): Promise<void> {
        const source = cell.document.getText().trim();
        
        if (this.isLLMQuery(source)) {
            await this.executeLLMQuery(cell, source);
        } else {
            // Pass through to original kernel
            await originalExecution();
        }
    }

    private isLLMQuery(source: string, notebook?: vscode.NotebookDocument): boolean {
        // Check for explicit LLM magic commands first
        const llmPatterns = [
            /^%%llm\b/m,
            /^%llm\b/m,
            /^@llm\b/m,
            /^#\s*@llm\b/m,
            /^\/\/\s*@llm\b/m,
            /^\s*LLM\(/m
        ];
        
        if (llmPatterns.some(pattern => pattern.test(source))) {
            return true;
        }

        // If notebook is provided and chat mode is active, check for natural language
        if (notebook) {
            // Get the overlay manager instance to check chat mode
            // This is a bit circular, but we need access to the parent's chat mode manager
            try {
                // Access the parent overlay manager through VS Code context
                const commands = vscode.commands.getCommands();
                // For now, we'll use a simpler approach - check notebook metadata directly
                const isChatModeActive = notebook.metadata?.llm_chat_mode?.enabled === true;
                
                if (isChatModeActive) {
                    // Use a simplified natural language detector
                    return this.isSimpleNaturalLanguage(source);
                }
            } catch (error) {
                // Fallback to magic command detection only
            }
        }
        
        return false;
    }

    private isSimpleNaturalLanguage(source: string): boolean {
        const trimmed = source.trim();
        
        // Empty or very short text
        if (trimmed.length < 3) {
            return false;
        }

        // Definitely code if matches programming patterns
        const codePatterns = [
            /^(import|from|def|class|if|for|while|try|function|var|let|const)\s/i,
            /^\s*\w+\s*[=<-]\s*/,
            /^\s*\w+\s*\(/,
            /^%%?\w+/,
            /^!\s*\w+/
        ];
        
        if (codePatterns.some(pattern => pattern.test(trimmed))) {
            return false;
        }

        // Likely natural language indicators
        const naturalPatterns = [
            /^(what|how|why|when|where|who|which|can|could|would|should|is|are|do|does|did)\s+/i,
            /^(please|help|explain|show|tell|describe|analyze|compare|summarize)\s+/i,
            /\?$/,
            /^(could you|can you|would you|please)/i
        ];
        
        return naturalPatterns.some(pattern => pattern.test(trimmed));
    }

    private async executeLLMQuery(cell: vscode.NotebookCell, source: string): Promise<void> {
        // This would be the same LLM execution logic
        // but now it works as an overlay on any kernel
        
        // Note: createNotebookCellExecution might not be available in older VS Code API versions
        // This is a mock implementation for compilation
        const execution = {
            start: (time: number) => {},
            end: (success: boolean, time: number) => {},
            replaceOutput: (outputs: vscode.NotebookCellOutput[]) => {}
        } as vscode.NotebookCellExecution;
        execution.start(Date.now());

        try {
            execution.replaceOutput([
                new vscode.NotebookCellOutput([
                    vscode.NotebookCellOutputItem.text('🤖 Processing LLM query...', 'text/plain')
                ])
            ]);

            // Mock LLM execution
            await new Promise(resolve => setTimeout(resolve, 2000));

            const response = {
                model: 'gpt-4o-mini',
                content: 'Here is your AI response based on the notebook context...',
                cost: 0.0023,
                tokens: 150,
                timestamp: new Date().toISOString(),
                context_size: 3
            };

            execution.replaceOutput([
                new vscode.NotebookCellOutput([
                    vscode.NotebookCellOutputItem.json(response, 'application/llm-response')
                ])
            ]);

            execution.end(true, Date.now());
        } catch (error) {
            execution.end(false, Date.now());
        }
    }

    dispose(): void {
        this.interceptedNotebooks.clear();
        this.originalControllers.clear();
    }
}

class OverlayIndicator {
    private statusBarItem?: vscode.StatusBarItem;
    private activeNotebook?: string;

    showOverlayActive(notebook: vscode.NotebookDocument): void {
        const notebookUri = notebook.uri.toString();
        this.activeNotebook = notebookUri;

        if (!this.statusBarItem) {
            this.statusBarItem = vscode.window.createStatusBarItem(
                vscode.StatusBarAlignment.Right,
                200
            );
        }

        this.statusBarItem.text = '$(robot) LLM Overlay';
        this.statusBarItem.tooltip = 'LLM Overlay is active - use %%llm for AI queries';
        this.statusBarItem.backgroundColor = new vscode.ThemeColor('statusBarItem.warningBackground');
        this.statusBarItem.command = 'llm-kernel.toggleOverlay';
        this.statusBarItem.show();
    }

    hideOverlayActive(notebook: vscode.NotebookDocument): void {
        const notebookUri = notebook.uri.toString();
        if (this.activeNotebook === notebookUri) {
            this.statusBarItem?.hide();
            this.activeNotebook = undefined;
        }
    }

    dispose(): void {
        this.statusBarItem?.dispose();
    }
}