import * as vscode from 'vscode';

/**
 * Provides per-cell status bar items showing which LLM model each cell uses.
 * Clicking the badge opens a quick pick to switch the model for that cell.
 */
export class CellModelStatusProvider implements vscode.NotebookCellStatusBarItemProvider {
    private _onDidChangeCellStatusBarItems = new vscode.EventEmitter<void>();
    readonly onDidChangeCellStatusBarItems = this._onDidChangeCellStatusBarItems.event;

    provideCellStatusBarItems(
        cell: vscode.NotebookCell,
        _token: vscode.CancellationToken
    ): vscode.NotebookCellStatusBarItem[] {
        const text = cell.document.getText().trim();

        // Only show for LLM-related cells
        if (!text.startsWith('%%llm') && !text.startsWith('%llm_')) {
            return [];
        }

        const modelName = this.extractModelName(text);
        if (!modelName) {
            return [];
        }

        const item = new vscode.NotebookCellStatusBarItem(
            `$(hubot) ${modelName}`,
            vscode.NotebookCellStatusBarAlignment.Right
        );
        item.command = {
            title: 'Switch model for this cell',
            command: 'llm-kernel.switchCellModel',
        };
        item.tooltip = `Model: ${modelName} (click to change)`;

        return [item];
    }

    private extractModelName(text: string): string | undefined {
        // %%llm --model=gpt-4o
        const modelMatch = text.match(/%%llm\s+--model=(\S+)/);
        if (modelMatch) {
            return modelMatch[1];
        }

        // %%llm (no model specified) → "default"
        if (text.startsWith('%%llm')) {
            return 'default';
        }

        // %%llm_gpt4
        if (text.startsWith('%%llm_gpt4')) {
            return 'GPT-4';
        }

        // %%llm_claude
        if (text.startsWith('%%llm_claude')) {
            return 'Claude';
        }

        // %%llm_compare
        if (text.startsWith('%%llm_compare')) {
            return 'Compare';
        }

        // %%llm_vision
        if (text.startsWith('%%llm_vision')) {
            return 'Vision';
        }

        // %%llm_mcp
        if (text.startsWith('%%llm_mcp')) {
            return 'MCP';
        }

        // %llm_model <name>
        const setModelMatch = text.match(/^%llm_model\s+(\S+)/);
        if (setModelMatch) {
            return `→ ${setModelMatch[1]}`;
        }

        // Other %llm_ line magics: show the command name
        const lineMagicMatch = text.match(/^%llm_(\w+)/);
        if (lineMagicMatch) {
            return lineMagicMatch[1];
        }

        return undefined;
    }

    refresh(): void {
        this._onDidChangeCellStatusBarItems.fire();
    }
}
