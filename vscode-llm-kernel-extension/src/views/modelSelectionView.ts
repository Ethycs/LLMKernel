import * as vscode from 'vscode';

export class ModelSelectionView {
    private panel: vscode.WebviewPanel | undefined;

    constructor() {
        this.createView();
    }

    private createView() {
        this.panel = vscode.window.createWebviewPanel(
            'modelSelection',
            'Select LLM Model',
            vscode.ViewColumn.One,
            {
                enableScripts: true,
                localResourceRoots: [vscode.Uri.joinPath(vscode.workspace.workspaceFolders![0].uri, 'resources')]
            }
        );

        this.panel.webview.html = this.getWebviewContent();
        this.panel.onDidDispose(() => this.panel = undefined);
    }

    private getWebviewContent(): string {
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

    public updateView() {
        if (this.panel) {
            this.panel.reveal(vscode.ViewColumn.One);
        } else {
            this.createView();
        }
    }
}