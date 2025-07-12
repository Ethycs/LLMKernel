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
exports.KernelStatusView = void 0;
const vscode = __importStar(require("vscode"));
class KernelStatusView {
    constructor() {
        this.status = 'Disconnected';
    }
    createStatusView(context) {
        this.panel = vscode.window.createWebviewPanel('kernelStatus', 'Kernel Status', vscode.ViewColumn.One, {
            enableScripts: true,
        });
        this.updateStatusView();
        this.panel.onDidDispose(() => {
            this.panel = undefined;
        });
        // Listen for status updates
        context.subscriptions.push(vscode.commands.registerCommand('llmKernel.updateStatus', (newStatus) => {
            this.updateStatus(newStatus);
        }));
    }
    updateStatus(newStatus) {
        this.status = newStatus;
        this.updateStatusView();
    }
    updateStatusView() {
        if (this.panel) {
            this.panel.webview.html = this.getWebviewContent();
        }
    }
    getWebviewContent() {
        return `
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Kernel Status</title>
                <style>
                    body { font-family: Arial, sans-serif; padding: 20px; }
                    .status { font-size: 24px; }
                </style>
            </head>
            <body>
                <h1>Kernel Status</h1>
                <div class="status">${this.status}</div>
            </body>
            </html>
        `;
    }
}
exports.KernelStatusView = KernelStatusView;
