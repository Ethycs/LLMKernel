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
Object.defineProperty(exports, "__esModule", { value: true });
exports.ModelSelectionView = void 0;
const vscode = __importStar(require("vscode"));
class ModelSelectionView {
    constructor() {
        this.createView();
    }
    createView() {
        this.panel = vscode.window.createWebviewPanel('modelSelection', 'Select LLM Model', vscode.ViewColumn.One, {
            enableScripts: true,
            localResourceRoots: [vscode.Uri.joinPath(vscode.workspace.workspaceFolders[0].uri, 'resources')]
        });
        this.panel.webview.html = this.getWebviewContent();
        this.panel.onDidDispose(() => this.panel = undefined);
    }
    getWebviewContent() {
        return `
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Select LLM Model</title>
            </head>
            <body>
                <h1>Select a Model</h1>
                <select id="modelSelect">
                    <option value="gpt-3">GPT-3</option>
                    <option value="gpt-4">GPT-4</option>
                    <option value="gpt-4o">GPT-4o</option>
                </select>
                <button id="selectButton">Select Model</button>
                <script>
                    const vscode = acquireVsCodeApi();
                    document.getElementById('selectButton').onclick = () => {
                        const selectedModel = document.getElementById('modelSelect').value;
                        vscode.postMessage({ command: 'selectModel', model: selectedModel });
                    };
                </script>
            </body>
            </html>
        `;
    }
    updateView() {
        if (this.panel) {
            this.panel.reveal(vscode.ViewColumn.One);
        }
        else {
            this.createView();
        }
    }
}
exports.ModelSelectionView = ModelSelectionView;
