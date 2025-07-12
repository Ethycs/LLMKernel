import * as vscode from 'vscode';
import { commands, window } from 'vscode';
import { ContextProvider } from '../providers/contextProvider';
import { ApiService } from '../services/apiService';

export function registerContextCommands(context: any): vscode.Disposable {
    const apiService = new ApiService();
    const contextProvider = new ContextProvider(apiService);
    const subscriptions: vscode.Disposable[] = [];

    subscriptions.push(
        commands.registerCommand('llm.context.save', async () => {
            const fileName = await window.showInputBox({ prompt: 'Enter context file name' });
            if (fileName) {
                await contextProvider.saveContext(fileName);
                window.showInformationMessage(`Context saved as ${fileName}`);
            }
        }),

        commands.registerCommand('llm.context.load', async () => {
            const fileName = await window.showInputBox({ prompt: 'Enter context file name to load' });
            if (fileName) {
                await contextProvider.loadContext(fileName);
                window.showInformationMessage(`Context loaded from ${fileName}`);
            }
        }),

        commands.registerCommand('llm.context.reset', async () => {
            await contextProvider.resetContext();
            window.showInformationMessage('Context has been reset');
        }),

        commands.registerCommand('llm.context.status', async () => {
            await contextProvider.checkContextStatus();
        })
    );
    
    return vscode.Disposable.from(...subscriptions);
}