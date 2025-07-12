import * as vscode from 'vscode';
import { KernelService } from '../services/kernelService';

export class CompletionProvider implements vscode.CompletionItemProvider {
    private kernelService: KernelService;

    constructor(kernelService: KernelService) {
        this.kernelService = kernelService;
    }

    provideCompletionItems(
        document: vscode.TextDocument,
        position: vscode.Position,
        token: vscode.CancellationToken,
        context: vscode.CompletionContext
    ): vscode.ProviderResult<vscode.CompletionItem[]> {
        const linePrefix = document.lineAt(position).text.substr(0, position.character);
        
        // Only provide completions if the line starts with a specific prefix
        if (!linePrefix.startsWith('%llm')) {
            return undefined;
        }

        // Call the kernel service to get completions
        return this.kernelService.getCompletions(linePrefix).then(completions => {
            return completions.map(completion => {
                const item = new vscode.CompletionItem(completion, vscode.CompletionItemKind.Text);
                item.detail = 'LLM kernel completion';
                item.documentation = `Auto-completion for ${completion}`;
                return item;
            });
        });
    }
}