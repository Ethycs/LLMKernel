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
exports.UniversalLLMProvider = void 0;
const vscode = __importStar(require("vscode"));
class UniversalLLMProvider {
    constructor() {
        const config = vscode.workspace.getConfiguration('llm-kernel');
        this.currentModel = config.get('defaultModel', 'gpt-4o-mini');
    }
    queryInCurrentCell() {
        return __awaiter(this, void 0, void 0, function* () {
            const editor = vscode.window.activeNotebookEditor;
            if (!editor) {
                vscode.window.showWarningMessage('No active notebook editor found');
                return;
            }
            const models = this.getModelQuickPickItems();
            const selected = yield vscode.window.showQuickPick(models, {
                placeHolder: 'Select an LLM model',
                matchOnDescription: true
            });
            if (!selected)
                return;
            const query = yield vscode.window.showInputBox({
                prompt: 'Enter your query for the LLM',
                placeHolder: 'e.g., Explain this code, Generate a function that...',
                ignoreFocusOut: true
            });
            if (!query)
                return;
            yield this.insertMagicCell(editor, `%%llm --model=${selected.value}\n${query}`);
        });
    }
    queryWithModel(model) {
        return __awaiter(this, void 0, void 0, function* () {
            const editor = vscode.window.activeNotebookEditor;
            if (!editor) {
                vscode.window.showWarningMessage('No active notebook editor found');
                return;
            }
            const query = yield vscode.window.showInputBox({
                prompt: `Enter your query for ${model}`,
                placeHolder: 'e.g., Explain this code, Generate a function that...',
                ignoreFocusOut: true
            });
            if (!query)
                return;
            yield this.insertMagicCell(editor, `%%llm --model=${model}\n${query}`);
        });
    }
    explainCurrentCell() {
        return __awaiter(this, void 0, void 0, function* () {
            const editor = vscode.window.activeNotebookEditor;
            if (!editor)
                return;
            const activeCell = editor.notebook.cellAt(editor.selection.start);
            if (!activeCell)
                return;
            const cellCode = activeCell.document.getText();
            const language = this.detectLanguage(editor.notebook);
            yield this.insertMagicCell(editor, `%%llm\nExplain this ${language} code:\n\n${cellCode}`);
        });
    }
    refactorCurrentCell() {
        return __awaiter(this, void 0, void 0, function* () {
            const editor = vscode.window.activeNotebookEditor;
            if (!editor)
                return;
            const activeCell = editor.notebook.cellAt(editor.selection.start);
            if (!activeCell)
                return;
            const cellCode = activeCell.document.getText();
            const language = this.detectLanguage(editor.notebook);
            const refactorType = yield vscode.window.showQuickPick([
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
            if (!refactorType)
                return;
            let goal;
            if (refactorType === 'Custom refactoring...') {
                const customQuery = yield vscode.window.showInputBox({
                    prompt: 'Describe the refactoring you want',
                    placeHolder: 'e.g., Convert to async/await, Use list comprehension'
                });
                if (!customQuery)
                    return;
                goal = customQuery;
            }
            else {
                goal = refactorType.toLowerCase();
            }
            yield this.insertMagicCell(editor, `%%llm\nRefactor this ${language} code to ${goal}:\n\n${cellCode}`);
        });
    }
    switchModel() {
        return __awaiter(this, void 0, void 0, function* () {
            const editor = vscode.window.activeNotebookEditor;
            const models = this.getModelQuickPickItems();
            const selected = yield vscode.window.showQuickPick(models, {
                placeHolder: `Current model: ${this.currentModel}`
            });
            if (!selected)
                return;
            this.currentModel = selected.value;
            if (editor) {
                yield this.insertMagicCell(editor, `%llm_model ${selected.value}`, true);
            }
            else {
                vscode.window.showInformationMessage(`Model will be set to ${selected.label} in the next notebook`);
            }
        });
    }
    pruneContext() {
        return __awaiter(this, void 0, void 0, function* () {
            const editor = vscode.window.activeNotebookEditor;
            if (!editor)
                return;
            const strategyMap = {
                'Smart prune (AI-powered)': '--strategy=smart',
                'Semantic similarity': '--strategy=semantic --threshold=0.7',
                'Keep recent only': '--strategy=recency --keep=10',
                'Keep last N cells...': ''
            };
            const selected = yield vscode.window.showQuickPick(Object.keys(strategyMap), {
                placeHolder: 'Select context pruning strategy'
            });
            if (!selected)
                return;
            let args = strategyMap[selected];
            if (selected === 'Keep last N cells...') {
                const n = yield vscode.window.showInputBox({
                    prompt: 'Number of cells to keep',
                    value: '5',
                    validateInput: (value) => {
                        const num = parseInt(value);
                        return isNaN(num) || num < 1 ? 'Please enter a valid number' : null;
                    }
                });
                if (!n)
                    return;
                args = `--strategy=recency --keep=${n}`;
            }
            yield this.insertMagicCell(editor, `%llm_prune ${args}`, true);
        });
    }
    compareModels() {
        return __awaiter(this, void 0, void 0, function* () {
            const editor = vscode.window.activeNotebookEditor;
            if (!editor)
                return;
            const query = yield vscode.window.showInputBox({
                prompt: 'Enter query to compare across models',
                placeHolder: 'e.g., Generate a sorting algorithm',
                ignoreFocusOut: true
            });
            if (!query)
                return;
            const models = yield vscode.window.showQuickPick([
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
            yield this.insertMagicCell(editor, `%%llm_compare ${modelNames}\n${query}`);
        });
    }
    showDashboard() {
        return __awaiter(this, void 0, void 0, function* () {
            const editor = vscode.window.activeNotebookEditor;
            if (editor) {
                yield this.insertMagicCell(editor, '%llm_status', true);
            }
            else {
                vscode.window.showWarningMessage('Open a notebook to view kernel status');
            }
        });
    }
    createNewLLMNotebook() {
        return __awaiter(this, void 0, void 0, function* () {
            // Create a new .ipynb notebook with a status cell
            const cells = [
                new vscode.NotebookCellData(vscode.NotebookCellKind.Markup, '# LLM Notebook\nUse `%%llm` to query models. Run `%llm_models` to see available models.', 'markdown'),
                new vscode.NotebookCellData(vscode.NotebookCellKind.Code, '%llm_status', 'llm-kernel')
            ];
            const notebookData = new vscode.NotebookData(cells);
            notebookData.metadata = {
                kernelspec: {
                    name: 'llm_kernel',
                    display_name: 'LLM Kernel',
                    language: 'python'
                }
            };
            const document = yield vscode.workspace.openNotebookDocument('jupyter-notebook', notebookData);
            yield vscode.window.showNotebookDocument(document);
        });
    }
    /**
     * Execute a simple magic command by inserting a cell and running it immediately.
     */
    executeSimpleMagic(magic) {
        return __awaiter(this, void 0, void 0, function* () {
            const editor = vscode.window.activeNotebookEditor;
            if (!editor) {
                vscode.window.showWarningMessage('No active notebook editor found');
                return;
            }
            yield this.insertMagicCell(editor, magic, true);
        });
    }
    /**
     * Insert a magic cell without auto-executing (public wrapper).
     */
    insertMagicCellPublic(content) {
        return __awaiter(this, void 0, void 0, function* () {
            const editor = vscode.window.activeNotebookEditor;
            if (!editor) {
                vscode.window.showWarningMessage('No active notebook editor found');
                return;
            }
            yield this.insertMagicCell(editor, content, false);
        });
    }
    /**
     * Insert a Python cell with magic command content into the notebook.
     * If autoExecute is true, the cell is executed immediately after insertion.
     */
    insertMagicCell(editor, content, autoExecute = false) {
        return __awaiter(this, void 0, void 0, function* () {
            const position = editor.selection.end + 1;
            const newCell = new vscode.NotebookCellData(vscode.NotebookCellKind.Code, content, 'llm-kernel');
            const edit = new vscode.WorkspaceEdit();
            const notebookEdit = vscode.NotebookEdit.insertCells(position, [newCell]);
            edit.set(editor.notebook.uri, [notebookEdit]);
            yield vscode.workspace.applyEdit(edit);
            if (autoExecute) {
                yield vscode.commands.executeCommand('notebook.cell.execute', {
                    ranges: [{ start: position, end: position + 1 }]
                });
            }
        });
    }
    detectLanguage(notebook) {
        var _a, _b;
        const metadata = notebook.metadata;
        const kernelSpec = ((_a = metadata === null || metadata === void 0 ? void 0 : metadata.kernelspec) === null || _a === void 0 ? void 0 : _a.name) || ((_b = metadata === null || metadata === void 0 ? void 0 : metadata.language_info) === null || _b === void 0 ? void 0 : _b.name) || 'python';
        const languageMap = {
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
    getModelQuickPickItems() {
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
exports.UniversalLLMProvider = UniversalLLMProvider;
