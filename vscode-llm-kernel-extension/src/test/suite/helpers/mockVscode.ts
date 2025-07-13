import * as vscode from 'vscode';

export function createMockNotebook(cells: any[] = []): vscode.NotebookDocument {
    return {
        uri: vscode.Uri.file('/test/notebook.ipynb'),
        notebookType: 'jupyter-notebook',
        version: 1,
        isDirty: false,
        isUntitled: false,
        isClosed: false,
        metadata: {},
        cellCount: cells.length,
        cellAt: (index: number) => cells[index],
        getCells: (range?: vscode.NotebookRange) => cells,
        save: async () => true
    } as any;
}

export function createMockCell(
    value: string, 
    kind: vscode.NotebookCellKind = vscode.NotebookCellKind.Code,
    languageId: string = 'python'
): vscode.NotebookCell {
    const document = {
        getText: () => value,
        languageId: languageId,
        uri: vscode.Uri.file('/test/cell.py'),
        fileName: '/test/cell.py',
        isUntitled: false,
        version: 1,
        isDirty: false,
        isClosed: false,
        lineCount: value.split('\n').length,
        lineAt: (line: number) => ({
            text: value.split('\n')[line] || '',
            range: new vscode.Range(line, 0, line, value.split('\n')[line]?.length || 0),
            firstNonWhitespaceCharacterIndex: 0,
            isEmptyOrWhitespace: !value.split('\n')[line]?.trim()
        })
    } as any;

    return {
        index: 0,
        notebook: {} as any,
        kind: kind,
        document: document,
        metadata: {},
        outputs: [],
        executionSummary: undefined
    } as any;
}

export function createMockExtensionContext(): vscode.ExtensionContext {
    const subscriptions: vscode.Disposable[] = [];
    
    return {
        subscriptions: subscriptions,
        workspaceState: {
            get: (key: string) => undefined,
            update: async (key: string, value: any) => {},
            keys: () => []
        },
        globalState: {
            get: (key: string) => undefined,
            update: async (key: string, value: any) => {},
            keys: () => [],
            setKeysForSync: (keys: string[]) => {}
        },
        secrets: {
            get: async (key: string) => undefined,
            store: async (key: string, value: string) => {},
            delete: async (key: string) => {},
            onDidChange: new vscode.EventEmitter<vscode.SecretStorageChangeEvent>().event
        },
        extensionUri: vscode.Uri.file('/test/extension'),
        extensionPath: '/test/extension',
        storagePath: '/test/storage',
        globalStoragePath: '/test/global-storage',
        logPath: '/test/logs',
        extensionMode: vscode.ExtensionMode.Test,
        storageUri: vscode.Uri.file('/test/storage'),
        globalStorageUri: vscode.Uri.file('/test/global-storage'),
        logUri: vscode.Uri.file('/test/logs'),
        asAbsolutePath: (relativePath: string) => `/test/extension/${relativePath}`,
        environmentVariableCollection: {
            persistent: true,
            replace: (variable: string, value: string) => {},
            append: (variable: string, value: string) => {},
            prepend: (variable: string, value: string) => {},
            get: (variable: string) => undefined,
            forEach: (callback: Function) => {},
            delete: (variable: string) => {},
            clear: () => {}
        }
    } as any;
}

export class MockStatusBarItem implements vscode.StatusBarItem {
    id = 'test';
    alignment = vscode.StatusBarAlignment.Left;
    priority = 0;
    text = '';
    tooltip: string | vscode.MarkdownString | undefined = '';
    color: string | vscode.ThemeColor | undefined;
    backgroundColor: vscode.ThemeColor | undefined;
    command: string | vscode.Command | undefined;
    accessibilityInformation: vscode.AccessibilityInformation | undefined;
    name: string | undefined;
    
    show(): void {}
    hide(): void {}
    dispose(): void {}
}