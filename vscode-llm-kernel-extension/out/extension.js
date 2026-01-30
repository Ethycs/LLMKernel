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
exports.deactivate = exports.activate = void 0;
const vscode = __importStar(require("vscode"));
const completionProvider_1 = require("./providers/completionProvider");
const cellModelStatusProvider_1 = require("./providers/cellModelStatusProvider");
const kernelBootstrapper_1 = require("./services/kernelBootstrapper");
const statusBarManager_1 = require("./services/statusBarManager");
const universalLLMProvider_1 = require("./providers/universalLLMProvider");
const llmOverlayManager_1 = require("./services/llmOverlayManager");
function activate(context) {
    return __awaiter(this, void 0, void 0, function* () {
        const config = vscode.workspace.getConfiguration('llm-kernel');
        const bootstrapper = new kernelBootstrapper_1.KernelBootstrapper(context);
        const statusBarManager = new statusBarManager_1.StatusBarManager();
        const universalProvider = new universalLLMProvider_1.UniversalLLMProvider();
        // Check if kernel is installed and bootstrap if needed
        // Wrapped in try-catch so activation completes and commands register even if bootstrap fails
        try {
            const isInstalled = yield bootstrapper.isKernelInstalled();
            const isLocalDev = yield bootstrapper.isLocalDevelopment();
            if (!isInstalled) {
                if (isLocalDev) {
                    yield bootstrapper.bootstrapFromLocal();
                }
                else if (config.get('autoInstall', true)) {
                    yield bootstrapper.bootstrapFromRepository();
                }
                else {
                    const install = yield vscode.window.showInformationMessage('LLM Kernel not found. Install now?', 'Install', 'Not now');
                    if (install === 'Install') {
                        if (isLocalDev) {
                            yield bootstrapper.bootstrapFromLocal();
                        }
                        else {
                            yield bootstrapper.bootstrapFromRepository();
                        }
                    }
                }
            }
            else if (!isLocalDev) {
                if (config.get('autoUpdate', false)) {
                    const hasUpdate = yield bootstrapper.checkForUpdates();
                    if (hasUpdate) {
                        yield bootstrapper.performUpdate();
                    }
                }
            }
        }
        catch (err) {
            console.warn('LLM Kernel bootstrap failed (commands will still register):', err);
            vscode.window.showWarningMessage(`LLM Kernel bootstrap encountered an error: ${err instanceof Error ? err.message : String(err)}`);
        }
        // Initialize completion provider
        let overlayManager;
        try {
            const completionProvider = new completionProvider_1.CompletionProvider();
            // Register completion provider for multiple languages
            const languages = ['llm-kernel', 'python', 'javascript', 'typescript', 'r', 'julia'];
            languages.forEach(lang => {
                context.subscriptions.push(vscode.languages.registerCompletionItemProvider(lang, completionProvider, '%'));
            });
        }
        catch (err) {
            console.warn('Failed to initialize completion provider:', err);
        }
        // Register per-cell model status bar provider
        try {
            const cellModelProvider = new cellModelStatusProvider_1.CellModelStatusProvider();
            context.subscriptions.push(vscode.notebooks.registerNotebookCellStatusBarItemProvider('jupyter-notebook', cellModelProvider));
        }
        catch (err) {
            console.warn('Failed to register cell model status provider:', err);
        }
        try {
            overlayManager = new llmOverlayManager_1.LLMOverlayManager();
            context.subscriptions.push(overlayManager);
        }
        catch (err) {
            console.warn('Failed to create overlay manager:', err);
        }
        // Register universal LLM commands — these are the primary user-facing commands
        // and MUST always be registered
        registerUniversalLLMCommands(context, universalProvider, overlayManager);
        // Initialize status bar
        statusBarManager.initialize(context);
        // Start update checker
        startUpdateChecker(bootstrapper);
        // Set up event listeners
        vscode.workspace.onDidChangeConfiguration((e) => {
            if (e.affectsConfiguration('llm-kernel')) {
                // Reload configuration
                statusBarManager.updateFromConfig();
            }
        });
        // Chat-like behavior: auto-insert a new %%llm cell after execution completes
        let chatModeEnabled = false;
        context.subscriptions.push(vscode.workspace.onDidChangeNotebookDocument((e) => {
            for (const cellChange of e.cellChanges) {
                // Detect when a cell's execution finishes (executionSummary becomes set)
                if (cellChange.executionSummary === undefined)
                    continue;
                const cellText = cellChange.cell.document.getText().trim();
                const isChatToggle = cellText.startsWith('%llm_chat');
                // Track chat mode state from %llm_chat commands
                if (isChatToggle && cellChange.executionSummary.success) {
                    if (cellText.includes('off')) {
                        chatModeEnabled = false;
                    }
                    else {
                        chatModeEnabled = true;
                    }
                }
                // In chat mode, auto-insert a new %%llm cell after LLM queries complete
                const isLLMCell = cellText.startsWith('%%llm');
                if (isLLMCell && chatModeEnabled && !isChatToggle) {
                    const editor = vscode.window.activeNotebookEditor;
                    if (!editor || editor.notebook !== e.notebook)
                        continue;
                    const position = cellChange.cell.index + 1;
                    // Don't insert if next cell already exists and is empty
                    if (position < e.notebook.cellCount) {
                        const nextCell = e.notebook.cellAt(position);
                        if (nextCell.document.getText().trim() === '%%llm')
                            continue;
                    }
                    // Carry forward model from the executed cell
                    const modelMatch = cellText.match(/%%llm\s+--model=(\S+)/);
                    const newContent = modelMatch
                        ? `%%llm --model=${modelMatch[1]}\n`
                        : '%%llm\n';
                    const newCell = new vscode.NotebookCellData(vscode.NotebookCellKind.Code, newContent, 'llm-kernel');
                    const edit = new vscode.WorkspaceEdit();
                    const notebookEdit = vscode.NotebookEdit.insertCells(position, [newCell]);
                    edit.set(editor.notebook.uri, [notebookEdit]);
                    vscode.workspace.applyEdit(edit).then(() => {
                        // Auto-focus the new cell for immediate typing
                        const newRange = new vscode.NotebookRange(position, position + 1);
                        editor.selections = [newRange];
                        vscode.commands.executeCommand('notebook.cell.edit');
                    });
                }
            }
        }));
        // Allow programmatic control of chat mode state
        context.subscriptions.push(vscode.commands.registerCommand('llm-kernel.setChatModeState', (enabled) => {
            chatModeEnabled = enabled;
        }));
        // Show welcome message
        if (context.globalState.get('firstActivation', true)) {
            vscode.window.showInformationMessage('LLM Kernel Extension activated! Press Ctrl+Shift+L to add LLM queries to any notebook.', 'Show Guide', 'Dismiss').then(choice => {
                if (choice === 'Show Guide') {
                    vscode.commands.executeCommand('vscode.open', vscode.Uri.parse('https://github.com/your-username/LLMKernel/wiki'));
                }
            });
            context.globalState.update('firstActivation', false);
        }
    });
}
exports.activate = activate;
function deactivate() {
    // Clean up resources if necessary
}
exports.deactivate = deactivate;
function registerUniversalLLMCommands(context, provider, overlayManager) {
    // Query in current cell
    context.subscriptions.push(vscode.commands.registerCommand('llm-kernel.queryInCurrentCell', () => __awaiter(this, void 0, void 0, function* () {
        yield provider.queryInCurrentCell();
    })));
    // Model-specific queries
    context.subscriptions.push(vscode.commands.registerCommand('llm-kernel.queryGPT4', () => __awaiter(this, void 0, void 0, function* () {
        yield provider.queryWithModel('gpt-4o');
    })));
    context.subscriptions.push(vscode.commands.registerCommand('llm-kernel.queryClaude', () => __awaiter(this, void 0, void 0, function* () {
        yield provider.queryWithModel('claude-3-sonnet');
    })));
    // Switch model for a specific cell (triggered from cell status bar badge)
    context.subscriptions.push(vscode.commands.registerCommand('llm-kernel.switchCellModel', () => __awaiter(this, void 0, void 0, function* () {
        const editor = vscode.window.activeNotebookEditor;
        if (!editor)
            return;
        const models = [
            { label: 'GPT-4o', value: 'gpt-4o' },
            { label: 'GPT-4o Mini', value: 'gpt-4o-mini' },
            { label: 'GPT-4.1', value: 'gpt-4.1' },
            { label: 'o3-mini', value: 'o3-mini' },
            { label: 'Claude 3.5 Sonnet', value: 'claude-3-5-sonnet' },
            { label: 'Claude 3 Sonnet', value: 'claude-3-sonnet' },
            { label: 'Claude 3 Haiku', value: 'claude-3-haiku' },
            { label: 'Gemini 2.5 Pro', value: 'gemini-2.5-pro' },
            { label: 'Local Llama', value: 'ollama/llama3' },
        ];
        const selected = yield vscode.window.showQuickPick(models, {
            placeHolder: 'Select model for this cell'
        });
        if (!selected)
            return;
        const cellIndex = editor.selection.start;
        const cell = editor.notebook.cellAt(cellIndex);
        const cellText = cell.document.getText();
        let newText;
        if (cellText.match(/%%llm\s+--model=\S+/)) {
            // Replace existing --model=
            newText = cellText.replace(/%%llm\s+--model=\S+/, `%%llm --model=${selected.value}`);
        }
        else if (cellText.startsWith('%%llm_gpt4') || cellText.startsWith('%%llm_claude')) {
            // Replace shorthand with explicit model
            newText = cellText.replace(/^%%llm_\w+/, `%%llm --model=${selected.value}`);
        }
        else if (cellText.startsWith('%%llm')) {
            // Insert --model= after %%llm
            newText = cellText.replace(/^%%llm/, `%%llm --model=${selected.value}`);
        }
        else {
            return; // Not an LLM cell
        }
        const edit = new vscode.WorkspaceEdit();
        const fullRange = new vscode.Range(cell.document.positionAt(0), cell.document.positionAt(cellText.length));
        edit.replace(cell.document.uri, fullRange, newText);
        yield vscode.workspace.applyEdit(edit);
    })));
    // Explain current cell
    context.subscriptions.push(vscode.commands.registerCommand('llm-kernel.explainCurrentCell', () => __awaiter(this, void 0, void 0, function* () {
        yield provider.explainCurrentCell();
    })));
    // Refactor current cell
    context.subscriptions.push(vscode.commands.registerCommand('llm-kernel.refactorCurrentCell', () => __awaiter(this, void 0, void 0, function* () {
        yield provider.refactorCurrentCell();
    })));
    // Switch model
    context.subscriptions.push(vscode.commands.registerCommand('llm-kernel.switchModel', () => __awaiter(this, void 0, void 0, function* () {
        yield provider.switchModel();
    })));
    // Prune context
    context.subscriptions.push(vscode.commands.registerCommand('llm-kernel.pruneContext', () => __awaiter(this, void 0, void 0, function* () {
        yield provider.pruneContext();
    })));
    // Compare models
    context.subscriptions.push(vscode.commands.registerCommand('llm-kernel.compareModels', () => __awaiter(this, void 0, void 0, function* () {
        yield provider.compareModels();
    })));
    // Show dashboard
    context.subscriptions.push(vscode.commands.registerCommand('llm-kernel.showDashboard', () => __awaiter(this, void 0, void 0, function* () {
        yield provider.showDashboard();
    })));
    // Setup for workspace
    context.subscriptions.push(vscode.commands.registerCommand('llm-kernel.setupForWorkspace', () => __awaiter(this, void 0, void 0, function* () {
        const bootstrapper = new kernelBootstrapper_1.KernelBootstrapper(context);
        const isLocalDev = yield bootstrapper.isLocalDevelopment();
        if (isLocalDev) {
            yield bootstrapper.bootstrapFromLocal();
        }
        else {
            yield bootstrapper.bootstrapFromRepository();
        }
    })));
    // Add LLM cell command — inserts a Python cell with %%llm magic
    context.subscriptions.push(vscode.commands.registerCommand('llm-kernel.addLLMCell', () => __awaiter(this, void 0, void 0, function* () {
        const editor = vscode.window.activeNotebookEditor;
        if (!editor) {
            vscode.window.showWarningMessage('No active notebook editor found');
            return;
        }
        // Check current cell for model and carry forward
        const currentCell = editor.notebook.cellAt(editor.selection.start);
        const currentText = (currentCell === null || currentCell === void 0 ? void 0 : currentCell.document.getText().trim()) || '';
        const modelMatch = currentText.match(/%%llm\s+--model=(\S+)/);
        const content = modelMatch
            ? `%%llm --model=${modelMatch[1]}\n`
            : '%%llm\n';
        const position = editor.selection.end + 1;
        const newCell = new vscode.NotebookCellData(vscode.NotebookCellKind.Code, content, 'llm-kernel');
        const edit = new vscode.WorkspaceEdit();
        const notebookEdit = vscode.NotebookEdit.insertCells(position, [newCell]);
        edit.set(editor.notebook.uri, [notebookEdit]);
        yield vscode.workspace.applyEdit(edit);
    })));
    // Create new LLM notebook command
    context.subscriptions.push(vscode.commands.registerCommand('llm-kernel.createNewLLMNotebook', () => __awaiter(this, void 0, void 0, function* () {
        yield provider.createNewLLMNotebook();
    })));
    // Overlay toggle commands
    context.subscriptions.push(vscode.commands.registerCommand('llm-kernel.toggleOverlay', () => __awaiter(this, void 0, void 0, function* () {
        yield (overlayManager === null || overlayManager === void 0 ? void 0 : overlayManager.toggleOverlay());
    })));
    context.subscriptions.push(vscode.commands.registerCommand('llm-kernel.enableOverlay', () => __awaiter(this, void 0, void 0, function* () {
        var _a;
        const notebook = (_a = vscode.window.activeNotebookEditor) === null || _a === void 0 ? void 0 : _a.notebook;
        if (notebook) {
            yield (overlayManager === null || overlayManager === void 0 ? void 0 : overlayManager.enableOverlay(notebook));
        }
    })));
    context.subscriptions.push(vscode.commands.registerCommand('llm-kernel.disableOverlay', () => __awaiter(this, void 0, void 0, function* () {
        var _b;
        const notebook = (_b = vscode.window.activeNotebookEditor) === null || _b === void 0 ? void 0 : _b.notebook;
        if (notebook) {
            yield (overlayManager === null || overlayManager === void 0 ? void 0 : overlayManager.disableOverlay(notebook));
        }
    })));
    // Chat mode commands — insert and execute kernel magic commands
    const insertAndExecuteMagic = (magic) => __awaiter(this, void 0, void 0, function* () {
        const editor = vscode.window.activeNotebookEditor;
        if (!editor) {
            vscode.window.showWarningMessage('No active notebook editor found');
            return;
        }
        const position = editor.selection.end + 1;
        const newCell = new vscode.NotebookCellData(vscode.NotebookCellKind.Code, magic, 'llm-kernel');
        const edit = new vscode.WorkspaceEdit();
        const notebookEdit = vscode.NotebookEdit.insertCells(position, [newCell]);
        edit.set(editor.notebook.uri, [notebookEdit]);
        yield vscode.workspace.applyEdit(edit);
        yield vscode.commands.executeCommand('notebook.cell.execute', {
            ranges: [{ start: position, end: position + 1 }]
        });
    });
    context.subscriptions.push(vscode.commands.registerCommand('llm-kernel.toggleChatMode', () => __awaiter(this, void 0, void 0, function* () {
        yield insertAndExecuteMagic('%llm_chat');
    })));
    context.subscriptions.push(vscode.commands.registerCommand('llm-kernel.enableChatMode', () => __awaiter(this, void 0, void 0, function* () {
        yield insertAndExecuteMagic('%llm_chat on');
    })));
    context.subscriptions.push(vscode.commands.registerCommand('llm-kernel.disableChatMode', () => __awaiter(this, void 0, void 0, function* () {
        yield insertAndExecuteMagic('%llm_chat off');
    })));
    // === Context commands ===
    context.subscriptions.push(vscode.commands.registerCommand('llm-kernel.showContext', () => __awaiter(this, void 0, void 0, function* () {
        yield provider.executeSimpleMagic('%llm_context');
    })));
    context.subscriptions.push(vscode.commands.registerCommand('llm-kernel.resetContext', () => __awaiter(this, void 0, void 0, function* () {
        const confirm = yield vscode.window.showWarningMessage('Reset all LLM context? This cannot be undone.', { modal: true }, 'Reset');
        if (confirm === 'Reset') {
            yield provider.executeSimpleMagic('%llm_context_reset');
        }
    })));
    context.subscriptions.push(vscode.commands.registerCommand('llm-kernel.saveContext', () => __awaiter(this, void 0, void 0, function* () {
        const name = yield vscode.window.showInputBox({
            prompt: 'Name for saved context',
            placeHolder: 'e.g., my-analysis-session'
        });
        if (name) {
            yield provider.executeSimpleMagic(`%llm_context_save ${name}`);
        }
    })));
    context.subscriptions.push(vscode.commands.registerCommand('llm-kernel.loadContext', () => __awaiter(this, void 0, void 0, function* () {
        const name = yield vscode.window.showInputBox({
            prompt: 'Name of context to load',
            placeHolder: 'e.g., my-analysis-session'
        });
        if (name) {
            yield provider.executeSimpleMagic(`%llm_context_load ${name}`);
        }
    })));
    context.subscriptions.push(vscode.commands.registerCommand('llm-kernel.showHistory', () => __awaiter(this, void 0, void 0, function* () {
        yield provider.executeSimpleMagic('%llm_history');
    })));
    // === MCP commands ===
    context.subscriptions.push(vscode.commands.registerCommand('llm-kernel.mcpConnect', () => __awaiter(this, void 0, void 0, function* () {
        yield provider.executeSimpleMagic('%llm_mcp_connect');
    })));
    context.subscriptions.push(vscode.commands.registerCommand('llm-kernel.mcpTools', () => __awaiter(this, void 0, void 0, function* () {
        yield provider.executeSimpleMagic('%llm_mcp_tools');
    })));
    context.subscriptions.push(vscode.commands.registerCommand('llm-kernel.mcpQuery', () => __awaiter(this, void 0, void 0, function* () {
        const query = yield vscode.window.showInputBox({
            prompt: 'Enter MCP query',
            placeHolder: 'e.g., Search for files matching...',
            ignoreFocusOut: true
        });
        if (query) {
            yield provider.insertMagicCellPublic(`%%llm_mcp\n${query}`);
        }
    })));
    context.subscriptions.push(vscode.commands.registerCommand('llm-kernel.mcpConfig', () => __awaiter(this, void 0, void 0, function* () {
        yield provider.executeSimpleMagic('%llm_mcp_config');
    })));
    // === Multimodal commands ===
    context.subscriptions.push(vscode.commands.registerCommand('llm-kernel.pasteImage', () => __awaiter(this, void 0, void 0, function* () {
        yield provider.executeSimpleMagic('%llm_paste');
    })));
    context.subscriptions.push(vscode.commands.registerCommand('llm-kernel.addImage', () => __awaiter(this, void 0, void 0, function* () {
        const uri = yield vscode.window.showOpenDialog({
            canSelectMany: false,
            filters: { 'Images': ['png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp'] },
            title: 'Select image to add to context'
        });
        if (uri && uri[0]) {
            yield provider.executeSimpleMagic(`%llm_image ${uri[0].fsPath}`);
        }
    })));
    context.subscriptions.push(vscode.commands.registerCommand('llm-kernel.addPdf', () => __awaiter(this, void 0, void 0, function* () {
        const uri = yield vscode.window.showOpenDialog({
            canSelectMany: false,
            filters: { 'PDF': ['pdf'] },
            title: 'Select PDF to add to context'
        });
        if (uri && uri[0]) {
            yield provider.executeSimpleMagic(`%llm_pdf ${uri[0].fsPath}`);
        }
    })));
    context.subscriptions.push(vscode.commands.registerCommand('llm-kernel.visionQuery', () => __awaiter(this, void 0, void 0, function* () {
        const query = yield vscode.window.showInputBox({
            prompt: 'Enter vision query (images must be added first)',
            placeHolder: 'e.g., Describe what you see in this image',
            ignoreFocusOut: true
        });
        if (query) {
            yield provider.insertMagicCellPublic(`%%llm_vision\n${query}`);
        }
    })));
    // === Config/info commands ===
    context.subscriptions.push(vscode.commands.registerCommand('llm-kernel.showCost', () => __awaiter(this, void 0, void 0, function* () {
        yield provider.executeSimpleMagic('%llm_cost');
    })));
    context.subscriptions.push(vscode.commands.registerCommand('llm-kernel.showTokenCount', () => __awaiter(this, void 0, void 0, function* () {
        yield provider.executeSimpleMagic('%llm_token_count');
    })));
    context.subscriptions.push(vscode.commands.registerCommand('llm-kernel.showConfig', () => __awaiter(this, void 0, void 0, function* () {
        yield provider.executeSimpleMagic('%llm_config');
    })));
    context.subscriptions.push(vscode.commands.registerCommand('llm-kernel.showContextWindow', () => __awaiter(this, void 0, void 0, function* () {
        yield provider.executeSimpleMagic('%llm_context_window');
    })));
}
function startUpdateChecker(bootstrapper) {
    // Skip update checks in local development mode
    bootstrapper.isLocalDevelopment().then(isLocal => {
        if (isLocal) {
            return;
        }
        // Check for updates every 24 hours
        const checkInterval = 24 * 60 * 60 * 1000;
        setInterval(() => __awaiter(this, void 0, void 0, function* () {
            const config = vscode.workspace.getConfiguration('llm-kernel');
            if (config.get('checkForUpdates', true)) {
                const hasUpdate = yield bootstrapper.checkForUpdates();
                if (hasUpdate) {
                    const choice = yield vscode.window.showInformationMessage('LLM Kernel update available!', 'Update Now', 'Later', 'Auto-update');
                    if (choice === 'Update Now') {
                        yield bootstrapper.performUpdate();
                    }
                    else if (choice === 'Auto-update') {
                        yield config.update('autoUpdate', true, true);
                        yield bootstrapper.performUpdate();
                    }
                }
            }
        }), checkInterval);
        // Also check on startup (after a delay)
        setTimeout(() => __awaiter(this, void 0, void 0, function* () {
            const config = vscode.workspace.getConfiguration('llm-kernel');
            if (config.get('checkForUpdates', true) && !config.get('autoUpdate', false)) {
                const hasUpdate = yield bootstrapper.checkForUpdates();
                if (hasUpdate) {
                    vscode.window.showInformationMessage('LLM Kernel update available!', 'Update Now', 'Later').then((choice) => __awaiter(this, void 0, void 0, function* () {
                        if (choice === 'Update Now') {
                            yield bootstrapper.performUpdate();
                        }
                    }));
                }
            }
        }), 5000);
    });
}
