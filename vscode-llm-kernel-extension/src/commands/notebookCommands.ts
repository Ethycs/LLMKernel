import * as vscode from 'vscode';
import { NotebookService } from '../services/notebookService';

export function registerNotebookCommands(kernelProvider: any): vscode.Disposable {
    const notebookService = new NotebookService();
    const subscriptions: vscode.Disposable[] = [];

    subscriptions.push(
        vscode.commands.registerCommand('llm.notebook.executeCell', async (cell: vscode.NotebookCell) => {
            try {
                const document = vscode.window.activeNotebookEditor?.notebook;
                if (!document) throw new Error('No active notebook');
                await notebookService.executeCell(cell, document);
                vscode.window.showInformationMessage('Cell executed successfully');
            } catch (error) {
                vscode.window.showErrorMessage(`Error executing cell: ${error instanceof Error ? error.message : String(error)}`);
            }
        }),

        vscode.commands.registerCommand('llm.notebook.saveState', async () => {
            try {
                await notebookService.saveNotebookState();
                vscode.window.showInformationMessage('Notebook state saved successfully.');
            } catch (error) {
                vscode.window.showErrorMessage(`Error saving notebook state: ${error instanceof Error ? error.message : String(error)}`);
            }
        }),

        vscode.commands.registerCommand('llm.notebook.loadState', async () => {
            try {
                await notebookService.loadNotebookState();
                vscode.window.showInformationMessage('Notebook state loaded successfully.');
            } catch (error) {
                vscode.window.showErrorMessage(`Error loading notebook state: ${error instanceof Error ? error.message : String(error)}`);
            }
        })
    );
    
    return vscode.Disposable.from(...subscriptions);
}