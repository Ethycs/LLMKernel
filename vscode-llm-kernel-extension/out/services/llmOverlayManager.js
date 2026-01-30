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
exports.LLMOverlayManager = void 0;
const vscode = __importStar(require("vscode"));
/**
 * Manages the LLM overlay state for notebooks.
 * The overlay toggle sets notebook metadata and shows a visual indicator.
 * Actual LLM execution is handled by the real kernel via Jupyter protocol.
 */
class LLMOverlayManager {
    constructor() {
        this.activeOverlays = new Set();
        this.overlayIndicator = new OverlayIndicator();
        this.setupEventListeners();
    }
    setupEventListeners() {
        vscode.window.onDidChangeActiveNotebookEditor((editor) => {
            this.updateOverlayForNotebook(editor);
        });
        vscode.workspace.onDidChangeNotebookDocument((event) => {
            this.updateOverlayState(event.notebook);
        });
    }
    toggleOverlay(notebook) {
        var _a;
        return __awaiter(this, void 0, void 0, function* () {
            const targetNotebook = notebook || ((_a = vscode.window.activeNotebookEditor) === null || _a === void 0 ? void 0 : _a.notebook);
            if (!targetNotebook) {
                vscode.window.showWarningMessage('No active notebook found');
                return;
            }
            const notebookUri = targetNotebook.uri.toString();
            const isActive = this.activeOverlays.has(notebookUri);
            if (isActive) {
                yield this.disableOverlay(targetNotebook);
            }
            else {
                yield this.enableOverlay(targetNotebook);
            }
        });
    }
    enableOverlay(notebook) {
        return __awaiter(this, void 0, void 0, function* () {
            const notebookUri = notebook.uri.toString();
            // Update notebook metadata
            const edit = new vscode.WorkspaceEdit();
            const metadataEdit = vscode.NotebookEdit.updateNotebookMetadata(Object.assign(Object.assign({}, notebook.metadata), { llm_overlay: {
                    enabled: true,
                    version: '1.0.0',
                    enabledAt: new Date().toISOString()
                } }));
            edit.set(notebook.uri, [metadataEdit]);
            yield vscode.workspace.applyEdit(edit);
            this.activeOverlays.add(notebookUri);
            this.overlayIndicator.showOverlayActive(notebook);
            vscode.window.showInformationMessage('LLM Overlay enabled! Use %%llm for AI queries in any cell.', 'Show Guide').then(choice => {
                if (choice === 'Show Guide') {
                    this.showOverlayGuide();
                }
            });
        });
    }
    disableOverlay(notebook) {
        return __awaiter(this, void 0, void 0, function* () {
            const notebookUri = notebook.uri.toString();
            const edit = new vscode.WorkspaceEdit();
            const metadata = Object.assign({}, notebook.metadata);
            if (metadata.llm_overlay) {
                metadata.llm_overlay.enabled = false;
                metadata.llm_overlay.disabledAt = new Date().toISOString();
            }
            const metadataEdit = vscode.NotebookEdit.updateNotebookMetadata(metadata);
            edit.set(notebook.uri, [metadataEdit]);
            yield vscode.workspace.applyEdit(edit);
            this.activeOverlays.delete(notebookUri);
            this.overlayIndicator.hideOverlayActive(notebook);
            vscode.window.showInformationMessage('LLM Overlay disabled');
        });
    }
    isOverlayActive(notebook) {
        return this.activeOverlays.has(notebook.uri.toString());
    }
    updateOverlayForNotebook(editor) {
        var _a, _b;
        if (!editor)
            return;
        const isOverlayEnabled = ((_b = (_a = editor.notebook.metadata) === null || _a === void 0 ? void 0 : _a.llm_overlay) === null || _b === void 0 ? void 0 : _b.enabled) === true;
        const notebookUri = editor.notebook.uri.toString();
        if (isOverlayEnabled && !this.activeOverlays.has(notebookUri)) {
            this.activeOverlays.add(notebookUri);
            this.overlayIndicator.showOverlayActive(editor.notebook);
        }
        else if (!isOverlayEnabled && this.activeOverlays.has(notebookUri)) {
            this.activeOverlays.delete(notebookUri);
            this.overlayIndicator.hideOverlayActive(editor.notebook);
        }
    }
    updateOverlayState(notebook) {
        var _a, _b;
        const isOverlayEnabled = ((_b = (_a = notebook.metadata) === null || _a === void 0 ? void 0 : _a.llm_overlay) === null || _b === void 0 ? void 0 : _b.enabled) === true;
        const notebookUri = notebook.uri.toString();
        if (isOverlayEnabled !== this.activeOverlays.has(notebookUri)) {
            this.updateOverlayForNotebook(vscode.window.activeNotebookEditor);
        }
    }
    showOverlayGuide() {
        const panel = vscode.window.createWebviewPanel('llmOverlayGuide', 'LLM Overlay Guide', vscode.ViewColumn.Beside, {});
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
    dispose() {
        this.overlayIndicator.dispose();
    }
}
exports.LLMOverlayManager = LLMOverlayManager;
class OverlayIndicator {
    showOverlayActive(notebook) {
        const notebookUri = notebook.uri.toString();
        this.activeNotebook = notebookUri;
        if (!this.statusBarItem) {
            this.statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 200);
        }
        this.statusBarItem.text = '$(robot) LLM Overlay';
        this.statusBarItem.tooltip = 'LLM Overlay is active - use %%llm for AI queries';
        this.statusBarItem.backgroundColor = new vscode.ThemeColor('statusBarItem.warningBackground');
        this.statusBarItem.command = 'llm-kernel.toggleOverlay';
        this.statusBarItem.show();
    }
    hideOverlayActive(notebook) {
        var _a;
        const notebookUri = notebook.uri.toString();
        if (this.activeNotebook === notebookUri) {
            (_a = this.statusBarItem) === null || _a === void 0 ? void 0 : _a.hide();
            this.activeNotebook = undefined;
        }
    }
    dispose() {
        var _a;
        (_a = this.statusBarItem) === null || _a === void 0 ? void 0 : _a.dispose();
    }
}
