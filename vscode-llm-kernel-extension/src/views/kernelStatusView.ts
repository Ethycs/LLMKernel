import * as vscode from 'vscode';

export class KernelStatusView {
    private panel: vscode.WebviewPanel | undefined;
    private status: string;

    constructor() {
        this.status = 'Disconnected';
    }

    public createStatusView(context: vscode.ExtensionContext) {
        this.panel = vscode.window.createWebviewPanel(
            'kernelStatus',
            'Kernel Status',
            vscode.ViewColumn.One,
            {
                enableScripts: true,
            }
        );

        this.updateStatusView();

        this.panel.onDidDispose(() => {
            this.panel = undefined;
        });

        // Listen for status updates
        context.subscriptions.push(
            vscode.commands.registerCommand('llmKernel.updateStatus', (newStatus: string) => {
                this.updateStatus(newStatus);
            })
        );
    }

    public updateStatus(newStatus: string) {
        this.status = newStatus;
        this.updateStatusView();
    }

    private updateStatusView() {
        if (this.panel) {
            this.panel.webview.html = this.getWebviewContent();
        }
    }

    private getWebviewContent(): string {
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