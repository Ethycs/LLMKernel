import * as vscode from 'vscode';

/**
 * Manages the LLM overlay state for notebooks.
 * The overlay toggle sets notebook metadata and shows a visual indicator.
 * Actual LLM execution is handled by the real kernel via Jupyter protocol.
 */
export class LLMOverlayManager {
    private activeOverlays = new Set<string>();
    private overlayIndicator: OverlayIndicator;

    constructor() {
        this.overlayIndicator = new OverlayIndicator();
        this.setupEventListeners();
    }

    private setupEventListeners(): void {
        vscode.window.onDidChangeActiveNotebookEditor((editor) => {
            this.updateOverlayForNotebook(editor);
        });

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

        this.activeOverlays.add(notebookUri);
        this.overlayIndicator.showOverlayActive(notebook);

        vscode.window.showInformationMessage(
            'LLM Overlay enabled! Use %%llm for AI queries in any cell.',
            'Show Guide'
        ).then(choice => {
            if (choice === 'Show Guide') {
                this.showOverlayGuide();
            }
        });
    }

    async disableOverlay(notebook: vscode.NotebookDocument): Promise<void> {
        const notebookUri = notebook.uri.toString();

        const edit = new vscode.WorkspaceEdit();
        const metadata = { ...notebook.metadata };
        if (metadata.llm_overlay) {
            metadata.llm_overlay.enabled = false;
            metadata.llm_overlay.disabledAt = new Date().toISOString();
        }
        const metadataEdit = vscode.NotebookEdit.updateNotebookMetadata(metadata);
        edit.set(notebook.uri, [metadataEdit]);
        await vscode.workspace.applyEdit(edit);

        this.activeOverlays.delete(notebookUri);
        this.overlayIndicator.hideOverlayActive(notebook);

        vscode.window.showInformationMessage('LLM Overlay disabled');
    }

    isOverlayActive(notebook: vscode.NotebookDocument): boolean {
        return this.activeOverlays.has(notebook.uri.toString());
    }

    private updateOverlayForNotebook(editor?: vscode.NotebookEditor): void {
        if (!editor) return;

        const isOverlayEnabled = editor.notebook.metadata?.llm_overlay?.enabled === true;
        const notebookUri = editor.notebook.uri.toString();

        if (isOverlayEnabled && !this.activeOverlays.has(notebookUri)) {
            this.activeOverlays.add(notebookUri);
            this.overlayIndicator.showOverlayActive(editor.notebook);
        } else if (!isOverlayEnabled && this.activeOverlays.has(notebookUri)) {
            this.activeOverlays.delete(notebookUri);
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
                <h1>LLM Overlay Guide</h1>

                <h2>What is LLM Overlay?</h2>
                <p>LLM Overlay adds AI capabilities to ANY notebook without changing kernels or workflows.</p>

                <h2>How to Use</h2>

                <h3>In any cell, use LLM magic commands:</h3>
                <div class="example">
                    <code>%%llm --model=gpt-4o<br>
                    Explain this dataset and suggest visualizations</code>
                </div>

                <h3>Keyboard Shortcuts:</h3>
                <ul>
                    <li><span class="shortcut">Ctrl+Shift+L</span> - Add LLM query</li>
                    <li><span class="shortcut">Ctrl+Shift+E</span> - Explain current cell</li>
                    <li><span class="shortcut">Ctrl+Shift+R</span> - Refactor current cell</li>
                    <li><span class="shortcut">Ctrl+Shift+M</span> - Switch LLM model</li>
                    <li><span class="shortcut">Ctrl+Shift+T</span> - Toggle chat mode</li>
                </ul>

                <h2>Key Features</h2>
                <ul>
                    <li><strong>Works with any kernel</strong> - Python, R, Julia, JavaScript</li>
                    <li><strong>Context aware</strong> - AI sees your variables, imports, functions</li>
                    <li><strong>Toggle on/off</strong> - Enable only when needed</li>
                    <li><strong>Per-cell model</strong> - Different cells can use different LLMs</li>
                </ul>
            </body>
            </html>
        `;
    }

    dispose(): void {
        this.overlayIndicator.dispose();
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
