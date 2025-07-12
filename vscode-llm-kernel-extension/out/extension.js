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
const kernelCommands_1 = require("./commands/kernelCommands");
const contextCommands_1 = require("./commands/contextCommands");
const notebookCommands_1 = require("./commands/notebookCommands");
const kernelProvider_1 = require("./providers/kernelProvider");
const contextProvider_1 = require("./providers/contextProvider");
const completionProvider_1 = require("./providers/completionProvider");
const kernelBootstrapper_1 = require("./services/kernelBootstrapper");
const statusBarManager_1 = require("./services/statusBarManager");
const universalLLMProvider_1 = require("./providers/universalLLMProvider");
const apiService_1 = require("./services/apiService");
const kernelService_1 = require("./services/kernelService");
const llmNotebookController_1 = require("./controllers/llmNotebookController");
const llmNotebookSerializer_1 = require("./serializers/llmNotebookSerializer");
const universalLLMController_1 = require("./controllers/universalLLMController");
const llmOverlayManager_1 = require("./services/llmOverlayManager");
function activate(context) {
    return __awaiter(this, void 0, void 0, function* () {
        const config = vscode.workspace.getConfiguration('llm-kernel');
        const bootstrapper = new kernelBootstrapper_1.KernelBootstrapper(context);
        const statusBarManager = new statusBarManager_1.StatusBarManager();
        const universalProvider = new universalLLMProvider_1.UniversalLLMProvider();
        // Check if kernel is installed and bootstrap if needed
        const isInstalled = yield bootstrapper.isKernelInstalled();
        if (!isInstalled) {
            if (config.get('autoInstall', true)) {
                // Auto-install on first use
                yield bootstrapper.bootstrapFromRepository();
            }
            else {
                // Show install prompt
                const install = yield vscode.window.showInformationMessage('🤖 LLM Kernel not found. Install now?', 'Install', 'Not now');
                if (install === 'Install') {
                    yield bootstrapper.bootstrapFromRepository();
                }
            }
        }
        else {
            // Check for updates
            if (config.get('autoUpdate', false)) {
                const hasUpdate = yield bootstrapper.checkForUpdates();
                if (hasUpdate) {
                    yield bootstrapper.performUpdate();
                }
            }
        }
        // Initialize providers and controllers
        const apiService = new apiService_1.ApiService();
        const kernelService = new kernelService_1.KernelService();
        const kernelProvider = new kernelProvider_1.KernelProvider();
        const contextProvider = new contextProvider_1.ContextProvider(apiService);
        const completionProvider = new completionProvider_1.CompletionProvider(kernelService);
        const llmController = new llmNotebookController_1.LLMNotebookController();
        const llmSerializer = new llmNotebookSerializer_1.LLMNotebookSerializer();
        const universalController = new universalLLMController_1.UniversalLLMController();
        const overlayManager = new llmOverlayManager_1.LLMOverlayManager();
        // Register notebook components
        context.subscriptions.push(llmController);
        context.subscriptions.push(universalController);
        context.subscriptions.push(overlayManager);
        context.subscriptions.push(vscode.workspace.registerNotebookSerializer('llm-notebook', llmSerializer));
        // Register commands
        context.subscriptions.push((0, kernelCommands_1.registerKernelCommands)(kernelProvider));
        context.subscriptions.push((0, contextCommands_1.registerContextCommands)(context));
        context.subscriptions.push((0, notebookCommands_1.registerNotebookCommands)(kernelProvider));
        // Register universal LLM commands
        registerUniversalLLMCommands(context, universalProvider, overlayManager);
        // Register completion provider for multiple languages
        const languages = ['python', 'javascript', 'typescript', 'r', 'julia', 'llm-query'];
        languages.forEach(lang => {
            context.subscriptions.push(vscode.languages.registerCompletionItemProvider(lang, completionProvider));
        });
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
        // Show welcome message
        if (context.globalState.get('firstActivation', true)) {
            vscode.window.showInformationMessage('🚀 LLM Kernel Extension activated! Press Ctrl+Shift+L to add LLM queries to any notebook.', 'Show Guide', 'Dismiss').then(choice => {
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
        yield bootstrapper.bootstrapFromRepository();
    })));
    // Add LLM cell command
    context.subscriptions.push(vscode.commands.registerCommand('llm-kernel.addLLMCell', () => __awaiter(this, void 0, void 0, function* () {
        const editor = vscode.window.activeNotebookEditor;
        if (!editor) {
            vscode.window.showWarningMessage('No active notebook editor found');
            return;
        }
        const position = editor.selection.end + 1;
        const newCell = new vscode.NotebookCellData(vscode.NotebookCellKind.Code, 'Ask me anything!', 'llm-query');
        newCell.metadata = {
            cellType: 'llm',
            llmCell: true,
            model: vscode.workspace.getConfiguration('llm-kernel').get('defaultModel', 'gpt-4o-mini'),
            timestamp: new Date().toISOString()
        };
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
        yield overlayManager.toggleOverlay();
    })));
    context.subscriptions.push(vscode.commands.registerCommand('llm-kernel.enableOverlay', () => __awaiter(this, void 0, void 0, function* () {
        var _a;
        const notebook = (_a = vscode.window.activeNotebookEditor) === null || _a === void 0 ? void 0 : _a.notebook;
        if (notebook) {
            yield overlayManager.enableOverlay(notebook);
        }
    })));
    context.subscriptions.push(vscode.commands.registerCommand('llm-kernel.disableOverlay', () => __awaiter(this, void 0, void 0, function* () {
        var _b;
        const notebook = (_b = vscode.window.activeNotebookEditor) === null || _b === void 0 ? void 0 : _b.notebook;
        if (notebook) {
            yield overlayManager.disableOverlay(notebook);
        }
    })));
    // Chat mode commands
    context.subscriptions.push(vscode.commands.registerCommand('llm-kernel.toggleChatMode', () => __awaiter(this, void 0, void 0, function* () {
        yield overlayManager.toggleChatMode();
    })));
    context.subscriptions.push(vscode.commands.registerCommand('llm-kernel.enableChatMode', () => __awaiter(this, void 0, void 0, function* () {
        var _c;
        const notebook = (_c = vscode.window.activeNotebookEditor) === null || _c === void 0 ? void 0 : _c.notebook;
        if (notebook) {
            yield overlayManager.enableChatMode(notebook);
        }
    })));
    context.subscriptions.push(vscode.commands.registerCommand('llm-kernel.disableChatMode', () => __awaiter(this, void 0, void 0, function* () {
        var _d;
        const notebook = (_d = vscode.window.activeNotebookEditor) === null || _d === void 0 ? void 0 : _d.notebook;
        if (notebook) {
            yield overlayManager.disableChatMode(notebook);
        }
    })));
}
function startUpdateChecker(bootstrapper) {
    // Check for updates every 24 hours
    const checkInterval = 24 * 60 * 60 * 1000; // 24 hours
    setInterval(() => __awaiter(this, void 0, void 0, function* () {
        const config = vscode.workspace.getConfiguration('llm-kernel');
        if (config.get('checkForUpdates', true)) {
            const hasUpdate = yield bootstrapper.checkForUpdates();
            if (hasUpdate) {
                const choice = yield vscode.window.showInformationMessage('🔄 LLM Kernel update available!', 'Update Now', 'Later', 'Auto-update');
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
                vscode.window.showInformationMessage('🔄 LLM Kernel update available!', 'Update Now', 'Later').then((choice) => __awaiter(this, void 0, void 0, function* () {
                    if (choice === 'Update Now') {
                        yield bootstrapper.performUpdate();
                    }
                }));
            }
        }
    }), 5000); // Check 5 seconds after activation
}
