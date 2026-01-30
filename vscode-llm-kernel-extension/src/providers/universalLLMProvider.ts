import * as vscode from 'vscode';

export class UniversalLLMProvider {
    private currentModel: string;

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

        const models = this.getModelQuickPickItems();
        const selected = await vscode.window.showQuickPick(models, {
            placeHolder: 'Select an LLM model',
            matchOnDescription: true
        });
        if (!selected) return;

        const query = await vscode.window.showInputBox({
            prompt: 'Enter your query for the LLM',
            placeHolder: 'e.g., Explain this code, Generate a function that...',
            ignoreFocusOut: true
        });
        if (!query) return;

        await this.insertMagicCell(editor, `%%llm --model=${selected.value}\n${query}`);
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

        await this.insertMagicCell(editor, `%%llm --model=${model}\n${query}`);
    }

    async explainCurrentCell(): Promise<void> {
        const editor = vscode.window.activeNotebookEditor;
        if (!editor) return;

        const activeCell = editor.notebook.cellAt(editor.selection.start);
        if (!activeCell) return;

        const cellCode = activeCell.document.getText();
        const language = this.detectLanguage(editor.notebook);

        await this.insertMagicCell(editor, `%%llm\nExplain this ${language} code:\n\n${cellCode}`);
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

        let goal: string;
        if (refactorType === 'Custom refactoring...') {
            const customQuery = await vscode.window.showInputBox({
                prompt: 'Describe the refactoring you want',
                placeHolder: 'e.g., Convert to async/await, Use list comprehension'
            });
            if (!customQuery) return;
            goal = customQuery;
        } else {
            goal = refactorType.toLowerCase();
        }

        await this.insertMagicCell(editor, `%%llm\nRefactor this ${language} code to ${goal}:\n\n${cellCode}`);
    }

    async switchModel(): Promise<void> {
        const editor = vscode.window.activeNotebookEditor;
        const models = this.getModelQuickPickItems();
        const selected = await vscode.window.showQuickPick(models, {
            placeHolder: `Current model: ${this.currentModel}`
        });
        if (!selected) return;

        this.currentModel = selected.value;

        if (editor) {
            await this.insertMagicCell(editor, `%llm_model ${selected.value}`, true);
        } else {
            vscode.window.showInformationMessage(`Model will be set to ${selected.label} in the next notebook`);
        }
    }

    async pruneContext(): Promise<void> {
        const editor = vscode.window.activeNotebookEditor;
        if (!editor) return;

        const strategyMap: Record<string, string> = {
            'Smart prune (AI-powered)': '--strategy=smart',
            'Semantic similarity': '--strategy=semantic --threshold=0.7',
            'Keep recent only': '--strategy=recency --keep=10',
            'Keep last N cells...': ''
        };

        const selected = await vscode.window.showQuickPick(Object.keys(strategyMap), {
            placeHolder: 'Select context pruning strategy'
        });
        if (!selected) return;

        let args = strategyMap[selected];
        if (selected === 'Keep last N cells...') {
            const n = await vscode.window.showInputBox({
                prompt: 'Number of cells to keep',
                value: '5',
                validateInput: (value) => {
                    const num = parseInt(value);
                    return isNaN(num) || num < 1 ? 'Please enter a valid number' : null;
                }
            });
            if (!n) return;
            args = `--strategy=recency --keep=${n}`;
        }

        await this.insertMagicCell(editor, `%llm_prune ${args}`, true);
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
            { label: 'gpt-4o', picked: true },
            { label: 'claude-3-sonnet', picked: true },
            { label: 'gpt-4o-mini', picked: false },
            { label: 'claude-3-haiku', picked: false },
            { label: 'gemini-2.5-pro', picked: false }
        ], {
            canPickMany: true,
            placeHolder: 'Select models to compare (min 2)'
        });

        if (!models || models.length < 2) {
            vscode.window.showWarningMessage('Please select at least 2 models to compare');
            return;
        }

        const modelNames = models.map(m => m.label).join(' ');
        await this.insertMagicCell(editor, `%%llm_compare ${modelNames}\n${query}`);
    }

    async showDashboard(): Promise<void> {
        const editor = vscode.window.activeNotebookEditor;
        if (editor) {
            await this.insertMagicCell(editor, '%llm_status', true);
        } else {
            vscode.window.showWarningMessage('Open a notebook to view kernel status');
        }
    }

    async createNewLLMNotebook(): Promise<void> {
        // Create a new .ipynb notebook with a status cell
        const cells = [
            new vscode.NotebookCellData(
                vscode.NotebookCellKind.Markup,
                '# LLM Notebook\nUse `%%llm` to query models. Run `%llm_models` to see available models.',
                'markdown'
            ),
            new vscode.NotebookCellData(
                vscode.NotebookCellKind.Code,
                '%llm_status',
                'llm-kernel'
            )
        ];

        const notebookData = new vscode.NotebookData(cells);
        notebookData.metadata = {
            kernelspec: {
                name: 'llm_kernel',
                display_name: 'LLM Kernel',
                language: 'python'
            }
        };

        const document = await vscode.workspace.openNotebookDocument('jupyter-notebook', notebookData);
        await vscode.window.showNotebookDocument(document);
    }

    /**
     * Execute a simple magic command by inserting a cell and running it immediately.
     */
    async executeSimpleMagic(magic: string): Promise<void> {
        const editor = vscode.window.activeNotebookEditor;
        if (!editor) {
            vscode.window.showWarningMessage('No active notebook editor found');
            return;
        }
        await this.insertMagicCell(editor, magic, true);
    }

    /**
     * Insert a magic cell without auto-executing (public wrapper).
     */
    async insertMagicCellPublic(content: string): Promise<void> {
        const editor = vscode.window.activeNotebookEditor;
        if (!editor) {
            vscode.window.showWarningMessage('No active notebook editor found');
            return;
        }
        await this.insertMagicCell(editor, content, false);
    }

    /**
     * Insert a Python cell with magic command content into the notebook.
     * If autoExecute is true, the cell is executed immediately after insertion.
     */
    private async insertMagicCell(
        editor: vscode.NotebookEditor,
        content: string,
        autoExecute: boolean = false
    ): Promise<void> {
        const position = editor.selection.end + 1;

        const newCell = new vscode.NotebookCellData(
            vscode.NotebookCellKind.Code,
            content,
            'llm-kernel'
        );

        const edit = new vscode.WorkspaceEdit();
        const notebookEdit = vscode.NotebookEdit.insertCells(position, [newCell]);
        edit.set(editor.notebook.uri, [notebookEdit]);
        await vscode.workspace.applyEdit(edit);

        if (autoExecute) {
            await vscode.commands.executeCommand('notebook.cell.execute', {
                ranges: [{ start: position, end: position + 1 }]
            });
        }
    }

    private detectLanguage(notebook: vscode.NotebookDocument): string {
        const metadata = notebook.metadata;
        const kernelSpec = metadata?.kernelspec?.name || metadata?.language_info?.name || 'python';

        const languageMap: Record<string, string> = {
            'python3': 'python',
            'python': 'python',
            'llm_kernel': 'python',
            'llm-kernel': 'python',
            'ir': 'r',
            'julia-1.6': 'julia',
            'julia': 'julia',
            'javascript': 'javascript',
            'typescript': 'typescript'
        };

        return languageMap[kernelSpec] || 'python';
    }

    private getModelQuickPickItems() {
        return [
            { label: 'GPT-4o', description: 'Most capable, higher cost', value: 'gpt-4o' },
            { label: 'GPT-4o Mini', description: 'Fast and affordable', value: 'gpt-4o-mini' },
            { label: 'Claude 3 Sonnet', description: 'Balanced performance', value: 'claude-3-sonnet' },
            { label: 'Claude 3 Haiku', description: 'Fast and efficient', value: 'claude-3-haiku' },
            { label: 'Gemini 2.5 Pro', description: 'Google AI', value: 'gemini-2.5-pro' },
            { label: 'Local Llama', description: 'Free, runs locally', value: 'ollama/llama3' }
        ];
    }
}
