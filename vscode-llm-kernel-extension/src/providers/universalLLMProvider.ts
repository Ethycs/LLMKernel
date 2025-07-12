import * as vscode from 'vscode';

interface LLMOptions {
    model: string;
    temperature?: number;
    maxTokens?: number;
    stream?: boolean;
    comparisonId?: string;
}

interface LanguageSyntax {
    cellMagic?: string;
    lineMagic?: string;
    comment?: string;
    function?: string;
    decorator?: string;
    macro?: string;
}

const LLM_SYNTAXES: Record<string, LanguageSyntax> = {
    python: {
        cellMagic: '%%llm',
        lineMagic: '%llm',
        comment: '# @llm',
        function: 'LLM(',
        decorator: '@llm'
    },
    r: {
        cellMagic: '%%llm',
        comment: '# @llm',
        function: 'llm('
    },
    julia: {
        comment: '# @llm',
        macro: '@llm'
    },
    javascript: {
        comment: '// @llm',
        function: 'LLM('
    },
    typescript: {
        comment: '// @llm',
        function: 'LLM('
    },
    scala: {
        comment: '// @llm',
        lineMagic: '%llm'
    },
    java: {
        comment: '// @llm',
        function: 'LLM.query('
    },
    rust: {
        comment: '// @llm',
        macro: 'llm!'
    }
};

export class UniversalLLMProvider {
    private currentModel: string;
    private sessionCost: number = 0;
    private sessionTokens: number = 0;

    constructor() {
        const config = vscode.workspace.getConfiguration('llm-kernel');
        this.currentModel = config.get<string>('defaultModel', 'gpt-4o-mini');
    }

    async queryInCurrentCell(): Promise<void> {
        const editor = vscode.window.activeNotebookEditor;
        if (!editor) {
            vscode.window.showWarningMessage('No active notebook editor found');
            return;
        }

        // Show model selection
        const models = [
            { label: 'GPT-4o', description: 'Most capable, higher cost', value: 'gpt-4o' },
            { label: 'GPT-4o Mini', description: 'Fast and affordable', value: 'gpt-4o-mini' },
            { label: 'Claude 3 Sonnet', description: 'Balanced performance', value: 'claude-3-sonnet' },
            { label: 'Claude 3 Haiku', description: 'Fast and efficient', value: 'claude-3-haiku' },
            { label: 'Local Llama', description: 'Free, runs locally', value: 'local-llama' }
        ];

        const selected = await vscode.window.showQuickPick(models, {
            placeHolder: 'Select an LLM model',
            matchOnDescription: true
        });

        if (!selected) return;

        // Get user query
        const query = await vscode.window.showInputBox({
            prompt: 'Enter your query for the LLM',
            placeHolder: 'e.g., Explain this code, Generate a function that...',
            ignoreFocusOut: true
        });

        if (!query) return;

        // Create LLM cell
        await this.createLLMCell(editor, query, { model: selected.value });
    }

    async queryWithModel(model: string): Promise<void> {
        const editor = vscode.window.activeNotebookEditor;
        if (!editor) {
            vscode.window.showWarningMessage('No active notebook editor found');
            return;
        }

        const query = await vscode.window.showInputBox({
            prompt: `Enter your query for ${model}`,
            placeHolder: 'e.g., Explain this code, Generate a function that...',
            ignoreFocusOut: true
        });

        if (!query) return;

        await this.createLLMCell(editor, query, { model });
    }

    async explainCurrentCell(): Promise<void> {
        const editor = vscode.window.activeNotebookEditor;
        if (!editor) return;

        const activeCell = editor.notebook.cellAt(editor.selection.start);
        if (!activeCell) return;

        const cellCode = activeCell.document.getText();
        const language = this.detectLanguage(editor.notebook);

        const query = `Explain this ${language} code:\n\n${cellCode}`;
        await this.createLLMCell(editor, query, { model: 'gpt-4o-mini' });
    }

    async refactorCurrentCell(): Promise<void> {
        const editor = vscode.window.activeNotebookEditor;
        if (!editor) return;

        const activeCell = editor.notebook.cellAt(editor.selection.start);
        if (!activeCell) return;

        const cellCode = activeCell.document.getText();
        const language = this.detectLanguage(editor.notebook);

        const refactorType = await vscode.window.showQuickPick([
            'Improve performance',
            'Improve readability',
            'Add error handling',
            'Add type hints/annotations',
            'Convert to functional style',
            'Add documentation',
            'Custom refactoring...'
        ], {
            placeHolder: 'Select refactoring type'
        });

        if (!refactorType) return;

        let query: string;
        if (refactorType === 'Custom refactoring...') {
            const customQuery = await vscode.window.showInputBox({
                prompt: 'Describe the refactoring you want',
                placeHolder: 'e.g., Convert to async/await, Use list comprehension'
            });
            if (!customQuery) return;
            query = `Refactor this ${language} code: ${customQuery}\n\n${cellCode}`;
        } else {
            query = `Refactor this ${language} code to ${refactorType.toLowerCase()}:\n\n${cellCode}`;
        }

        await this.createLLMCell(editor, query, { model: 'gpt-4o' });
    }

    async switchModel(): Promise<void> {
        const models = [
            { label: 'GPT-4o', description: 'Most capable, higher cost', value: 'gpt-4o' },
            { label: 'GPT-4o Mini', description: 'Fast and affordable', value: 'gpt-4o-mini' },
            { label: 'Claude 3 Sonnet', description: 'Balanced performance', value: 'claude-3-sonnet' },
            { label: 'Claude 3 Haiku', description: 'Fast and efficient', value: 'claude-3-haiku' },
            { label: 'Local Llama', description: 'Free, runs locally', value: 'local-llama' }
        ];

        const selected = await vscode.window.showQuickPick(models, {
            placeHolder: `Current model: ${this.currentModel}`
        });

        if (selected) {
            this.currentModel = selected.value;
            vscode.window.showInformationMessage(`Switched to ${selected.label}`);
            
            // Update status bar
            vscode.commands.executeCommand('llm-kernel.updateStatusBar');
        }
    }

    async pruneContext(): Promise<void> {
        const editor = vscode.window.activeNotebookEditor;
        if (!editor) return;

        const options = [
            { label: 'Remove low-relevance cells', description: 'Keep only highly relevant context' },
            { label: 'Keep last N cells', description: 'Retain only recent context' },
            { label: 'Remove all except pinned', description: 'Clear all unpinned context' },
            { label: 'Smart prune', description: 'AI-powered context optimization' }
        ];

        const selected = await vscode.window.showQuickPick(options, {
            placeHolder: 'Select context pruning strategy'
        });

        if (!selected) return;

        // Create context management cell
        const language = this.detectLanguage(editor.notebook);
        let llmCode: string;

        switch (selected.label) {
            case 'Keep last N cells':
                const n = await vscode.window.showInputBox({
                    prompt: 'Number of cells to keep',
                    value: '5',
                    validateInput: (value) => {
                        const num = parseInt(value);
                        return isNaN(num) || num < 1 ? 'Please enter a valid number' : null;
                    }
                });
                if (!n) return;
                llmCode = this.generateLLMCode(`prune_context(keep_last=${n})`, language, { model: 'system' });
                break;
            
            case 'Smart prune':
                llmCode = this.generateLLMCode('smart_prune_context()', language, { model: 'system' });
                break;
            
            default:
                llmCode = this.generateLLMCode(`prune_context(strategy="${selected.label}")`, language, { model: 'system' });
        }

        await this.createLLMCell(editor, llmCode, { model: 'system' });
    }

    async compareModels(): Promise<void> {
        const editor = vscode.window.activeNotebookEditor;
        if (!editor) return;

        const query = await vscode.window.showInputBox({
            prompt: 'Enter query to compare across models',
            placeHolder: 'e.g., Generate a sorting algorithm',
            ignoreFocusOut: true
        });

        if (!query) return;

        const models = await vscode.window.showQuickPick([
            { label: 'GPT-4o', picked: true },
            { label: 'Claude 3 Sonnet', picked: true },
            { label: 'GPT-4o Mini', picked: false },
            { label: 'Claude 3 Haiku', picked: false },
            { label: 'Local Llama', picked: false }
        ], {
            canPickMany: true,
            placeHolder: 'Select models to compare (min 2)'
        });

        if (!models || models.length < 2) {
            vscode.window.showWarningMessage('Please select at least 2 models to compare');
            return;
        }

        // Create comparison cells
        const language = this.detectLanguage(editor.notebook);
        const comparisonId = Date.now().toString();

        for (const model of models) {
            const llmCode = this.generateLLMCode(
                query,
                language,
                { 
                    model: model.label.toLowerCase().replace(/ /g, '-'),
                    comparisonId 
                }
            );
            await this.createLLMCell(editor, llmCode, { model: model.label });
        }

        vscode.window.showInformationMessage(`Created ${models.length} comparison cells`);
    }

    async showDashboard(): Promise<void> {
        // Create and show webview dashboard
        const panel = vscode.window.createWebviewPanel(
            'llmDashboard',
            'LLM Dashboard',
            vscode.ViewColumn.Beside,
            {
                enableScripts: true,
                retainContextWhenHidden: true
            }
        );

        panel.webview.html = this.getDashboardHTML();
    }

    private async createLLMCell(
        editor: vscode.NotebookEditor,
        content: string,
        options: Partial<LLMOptions> = {}
    ): Promise<void> {
        const { model = this.currentModel } = options;

        const position = editor.selection.end + 1;
        
        // Create a dedicated LLM cell
        const newCell = new vscode.NotebookCellData(
            vscode.NotebookCellKind.Code,
            content,
            'llm-query'  // Use the custom LLM language
        );

        // Set LLM-specific metadata
        newCell.metadata = {
            cellType: 'llm',
            llmCell: true,
            model: model,
            temperature: options.temperature || 0.7,
            maxTokens: options.maxTokens || 1000,
            timestamp: new Date().toISOString()
        };

        const edit = new vscode.WorkspaceEdit();
        const notebookEdit = vscode.NotebookEdit.insertCells(position, [newCell]);
        edit.set(editor.notebook.uri, [notebookEdit]);

        await vscode.workspace.applyEdit(edit);

        // Auto-execute if configured
        const config = vscode.workspace.getConfiguration('llm-kernel');
        if (config.get('autoExecute', false)) {
            vscode.commands.executeCommand('notebook.cell.execute', {
                ranges: [{ start: position, end: position + 1 }]
            });
        }

        // Update status bar to show new context
        vscode.commands.executeCommand('llm-kernel.updateStatusBar');
    }

    async createNewLLMNotebook(): Promise<void> {
        const uri = await vscode.window.showSaveDialog({
            defaultUri: vscode.Uri.file('Untitled.llmnb'),
            filters: {
                'LLM Notebooks': ['llmnb'],
                'All Files': ['*']
            }
        });

        if (uri) {
            const { LLMNotebookSerializer } = await import('../serializers/llmNotebookSerializer');
            const notebookData = LLMNotebookSerializer.createNewLLMNotebook();
            
            const edit = new vscode.WorkspaceEdit();
            edit.createFile(uri);
            await vscode.workspace.applyEdit(edit);
            
            const document = await vscode.workspace.openNotebookDocument('llm-notebook', notebookData);
            await vscode.window.showNotebookDocument(document);
        }
    }

    private detectLanguage(notebook: vscode.NotebookDocument): string {
        const metadata = notebook.metadata;
        const kernelSpec = metadata?.kernelspec?.name || metadata?.language_info?.name || 'python';

        const languageMap: Record<string, string> = {
            'python3': 'python',
            'python': 'python',
            'ir': 'r',
            'julia-1.6': 'julia',
            'julia': 'julia',
            'javascript': 'javascript',
            'typescript': 'typescript',
            'scala': 'scala',
            'java': 'java',
            'rust': 'rust'
        };

        return languageMap[kernelSpec] || 'python';
    }

    private generateLLMCode(query: string, language: string, options: LLMOptions): string {
        const syntax = LLM_SYNTAXES[language] || LLM_SYNTAXES.python;
        const { model = this.currentModel } = options;

        // Choose the best syntax for the language
        if (language === 'python' && syntax.cellMagic) {
            return `%%llm --model=${model}\n${query}`;
        } else if (language === 'r' && syntax.comment) {
            return `# @llm model=${model}\n# ${query.split('\n').join('\n# ')}`;
        } else if (language === 'julia' && syntax.macro) {
            return `@llm "${query}" model="${model}"`;
        } else if ((language === 'javascript' || language === 'typescript') && syntax.comment) {
            return `// @llm model=${model}\n// ${query.split('\n').join('\n// ')}`;
        } else if (syntax.comment) {
            const commentChar = syntax.comment.substring(0, 2);
            return `${commentChar} @llm model=${model}\n${commentChar} ${query.split('\n').join('\n' + commentChar + ' ')}`;
        } else {
            // Fallback to Python-style
            return `%%llm --model=${model}\n${query}`;
        }
    }

    private getDashboardHTML(): string {
        return `
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>LLM Dashboard</title>
                <style>
                    body {
                        font-family: var(--vscode-font-family);
                        color: var(--vscode-foreground);
                        background-color: var(--vscode-editor-background);
                        padding: 20px;
                        margin: 0;
                    }
                    
                    h1 {
                        color: var(--vscode-foreground);
                        margin-bottom: 20px;
                    }
                    
                    .metrics {
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                        gap: 20px;
                        margin-bottom: 30px;
                    }
                    
                    .metric-card {
                        background: var(--vscode-editorWidget-background);
                        border: 1px solid var(--vscode-editorWidget-border);
                        border-radius: 8px;
                        padding: 20px;
                        text-align: center;
                    }
                    
                    .metric-value {
                        font-size: 36px;
                        font-weight: bold;
                        color: var(--vscode-textLink-foreground);
                        margin: 10px 0;
                    }
                    
                    .metric-label {
                        font-size: 14px;
                        color: var(--vscode-descriptionForeground);
                    }
                    
                    .model-usage {
                        background: var(--vscode-editorWidget-background);
                        border: 1px solid var(--vscode-editorWidget-border);
                        border-radius: 8px;
                        padding: 20px;
                        margin-bottom: 20px;
                    }
                    
                    .usage-bar {
                        display: flex;
                        align-items: center;
                        margin: 10px 0;
                    }
                    
                    .usage-label {
                        width: 120px;
                        font-size: 14px;
                    }
                    
                    .usage-progress {
                        flex: 1;
                        height: 20px;
                        background: var(--vscode-progressBar-background);
                        border-radius: 10px;
                        overflow: hidden;
                        margin: 0 10px;
                    }
                    
                    .usage-fill {
                        height: 100%;
                        background: var(--vscode-progressBar-foreground);
                        transition: width 0.3s ease;
                    }
                    
                    .usage-count {
                        width: 60px;
                        text-align: right;
                        font-size: 14px;
                    }
                </style>
            </head>
            <body>
                <h1>LLM Usage Dashboard</h1>
                
                <div class="metrics">
                    <div class="metric-card">
                        <div class="metric-label">Total Cost</div>
                        <div class="metric-value">$${this.sessionCost.toFixed(2)}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Total Tokens</div>
                        <div class="metric-value">${this.sessionTokens.toLocaleString()}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Active Model</div>
                        <div class="metric-value" style="font-size: 20px;">${this.currentModel}</div>
                    </div>
                </div>
                
                <div class="model-usage">
                    <h2>Model Usage Distribution</h2>
                    <div class="usage-bar">
                        <div class="usage-label">GPT-4o</div>
                        <div class="usage-progress">
                            <div class="usage-fill" style="width: 40%;"></div>
                        </div>
                        <div class="usage-count">40%</div>
                    </div>
                    <div class="usage-bar">
                        <div class="usage-label">Claude 3</div>
                        <div class="usage-progress">
                            <div class="usage-fill" style="width: 30%;"></div>
                        </div>
                        <div class="usage-count">30%</div>
                    </div>
                    <div class="usage-bar">
                        <div class="usage-label">GPT-4o Mini</div>
                        <div class="usage-progress">
                            <div class="usage-fill" style="width: 20%;"></div>
                        </div>
                        <div class="usage-count">20%</div>
                    </div>
                    <div class="usage-bar">
                        <div class="usage-label">Local Models</div>
                        <div class="usage-progress">
                            <div class="usage-fill" style="width: 10%;"></div>
                        </div>
                        <div class="usage-count">10%</div>
                    </div>
                </div>
                
                <script>
                    const vscode = acquireVsCodeApi();
                    
                    // Update dashboard with real data
                    window.addEventListener('message', event => {
                        const message = event.data;
                        if (message.type === 'update-dashboard') {
                            // Update metrics with real data
                        }
                    });
                </script>
            </body>
            </html>
        `;
    }
}