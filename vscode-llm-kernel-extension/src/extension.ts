import * as vscode from 'vscode';
import { CompletionProvider } from './providers/completionProvider';
import { CellModelStatusProvider } from './providers/cellModelStatusProvider';
import { KernelBootstrapper } from './services/kernelBootstrapper';
import { StatusBarManager } from './services/statusBarManager';
import { UniversalLLMProvider } from './providers/universalLLMProvider';
import { LLMOverlayManager } from './services/llmOverlayManager';

export async function activate(context: vscode.ExtensionContext) {
    const config = vscode.workspace.getConfiguration('llm-kernel');
    const bootstrapper = new KernelBootstrapper(context);
    const statusBarManager = new StatusBarManager();
    const universalProvider = new UniversalLLMProvider();

    // Check if kernel is installed and bootstrap if needed
    // Wrapped in try-catch so activation completes and commands register even if bootstrap fails
    try {
        const isInstalled = await bootstrapper.isKernelInstalled();
        const isLocalDev = await bootstrapper.isLocalDevelopment();

        if (!isInstalled) {
            if (isLocalDev) {
                await bootstrapper.bootstrapFromLocal();
            } else if (config.get('autoInstall', true)) {
                await bootstrapper.bootstrapFromRepository();
            } else {
                const install = await vscode.window.showInformationMessage(
                    'LLM Kernel not found. Install now?',
                    'Install', 'Not now'
                );

                if (install === 'Install') {
                    if (isLocalDev) {
                        await bootstrapper.bootstrapFromLocal();
                    } else {
                        await bootstrapper.bootstrapFromRepository();
                    }
                }
            }
        } else if (!isLocalDev) {
            if (config.get('autoUpdate', false)) {
                const hasUpdate = await bootstrapper.checkForUpdates();
                if (hasUpdate) {
                    await bootstrapper.performUpdate();
                }
            }
        }
    } catch (err) {
        console.warn('LLM Kernel bootstrap failed (commands will still register):', err);
        vscode.window.showWarningMessage(
            `LLM Kernel bootstrap encountered an error: ${err instanceof Error ? err.message : String(err)}`
        );
    }

    // Initialize completion provider
    let overlayManager: LLMOverlayManager | undefined;

    try {
        const completionProvider = new CompletionProvider();

        // Register completion provider for multiple languages
        const languages = ['llm-kernel', 'python', 'javascript', 'typescript', 'r', 'julia'];
        languages.forEach(lang => {
            context.subscriptions.push(
                vscode.languages.registerCompletionItemProvider(lang, completionProvider, '%')
            );
        });
    } catch (err) {
        console.warn('Failed to initialize completion provider:', err);
    }

    // Register per-cell model status bar provider
    try {
        const cellModelProvider = new CellModelStatusProvider();
        context.subscriptions.push(
            vscode.notebooks.registerNotebookCellStatusBarItemProvider(
                'jupyter-notebook', cellModelProvider
            )
        );
    } catch (err) {
        console.warn('Failed to register cell model status provider:', err);
    }

    try {
        overlayManager = new LLMOverlayManager();
        context.subscriptions.push(overlayManager);
    } catch (err) {
        console.warn('Failed to create overlay manager:', err);
    }

    // Register universal LLM commands — these are the primary user-facing commands
    // and MUST always be registered
    registerUniversalLLMCommands(context, universalProvider, overlayManager);

    // Initialize status bar
    statusBarManager.initialize(context);

    // Start update checker
    startUpdateChecker(bootstrapper);

    // Set up event listeners
    vscode.workspace.onDidChangeConfiguration((e) => {
        if (e.affectsConfiguration('llm-kernel')) {
            // Reload configuration
            statusBarManager.updateFromConfig();
        }
    });

    // Chat-like behavior: auto-insert a new %%llm cell after execution completes
    let chatModeEnabled = false;
    context.subscriptions.push(
        vscode.workspace.onDidChangeNotebookDocument((e) => {
            for (const cellChange of e.cellChanges) {
                // Detect when a cell's execution finishes (executionSummary becomes set)
                if (cellChange.executionSummary === undefined) continue;

                const cellText = cellChange.cell.document.getText().trim();
                const isChatToggle = cellText.startsWith('%llm_chat');

                // Track chat mode state from %llm_chat commands
                if (isChatToggle && cellChange.executionSummary.success) {
                    if (cellText.includes('off')) {
                        chatModeEnabled = false;
                    } else {
                        chatModeEnabled = true;
                    }
                }

                // In chat mode, auto-insert a new %%llm cell after LLM queries complete
                const isLLMCell = cellText.startsWith('%%llm');
                if (isLLMCell && chatModeEnabled && !isChatToggle) {
                    const editor = vscode.window.activeNotebookEditor;
                    if (!editor || editor.notebook !== e.notebook) continue;

                    const position = cellChange.cell.index + 1;
                    // Don't insert if next cell already exists and is empty
                    if (position < e.notebook.cellCount) {
                        const nextCell = e.notebook.cellAt(position);
                        if (nextCell.document.getText().trim() === '%%llm') continue;
                    }

                    // Carry forward model from the executed cell
                    const modelMatch = cellText.match(/%%llm\s+--model=(\S+)/);
                    const newContent = modelMatch
                        ? `%%llm --model=${modelMatch[1]}\n`
                        : '%%llm\n';

                    const newCell = new vscode.NotebookCellData(
                        vscode.NotebookCellKind.Code,
                        newContent,
                        'llm-kernel'
                    );
                    const edit = new vscode.WorkspaceEdit();
                    const notebookEdit = vscode.NotebookEdit.insertCells(position, [newCell]);
                    edit.set(editor.notebook.uri, [notebookEdit]);
                    vscode.workspace.applyEdit(edit).then(() => {
                        // Auto-focus the new cell for immediate typing
                        const newRange = new vscode.NotebookRange(position, position + 1);
                        editor.selections = [newRange];
                        vscode.commands.executeCommand('notebook.cell.edit');
                    });
                }
            }
        })
    );

    // Allow programmatic control of chat mode state
    context.subscriptions.push(
        vscode.commands.registerCommand('llm-kernel.setChatModeState', (enabled: boolean) => {
            chatModeEnabled = enabled;
        })
    );

    // Show welcome message
    if (context.globalState.get('firstActivation', true)) {
        vscode.window.showInformationMessage(
            'LLM Kernel Extension activated! Press Ctrl+Shift+L to add LLM queries to any notebook.',
            'Show Guide',
            'Dismiss'
        ).then(choice => {
            if (choice === 'Show Guide') {
                vscode.commands.executeCommand('vscode.open', vscode.Uri.parse('https://github.com/your-username/LLMKernel/wiki'));
            }
        });
        context.globalState.update('firstActivation', false);
    }
}

export function deactivate() {
    // Clean up resources if necessary
}

function registerUniversalLLMCommands(context: vscode.ExtensionContext, provider: UniversalLLMProvider, overlayManager: LLMOverlayManager | undefined) {
    // Query in current cell
    context.subscriptions.push(
        vscode.commands.registerCommand('llm-kernel.queryInCurrentCell', async () => {
            await provider.queryInCurrentCell();
        })
    );

    // Model-specific queries
    context.subscriptions.push(
        vscode.commands.registerCommand('llm-kernel.queryGPT4', async () => {
            await provider.queryWithModel('gpt-4o');
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('llm-kernel.queryClaude', async () => {
            await provider.queryWithModel('claude-3-sonnet');
        })
    );

    // Switch model for a specific cell (triggered from cell status bar badge)
    context.subscriptions.push(
        vscode.commands.registerCommand('llm-kernel.switchCellModel', async () => {
            const editor = vscode.window.activeNotebookEditor;
            if (!editor) return;

            const models = [
                { label: 'GPT-4o', value: 'gpt-4o' },
                { label: 'GPT-4o Mini', value: 'gpt-4o-mini' },
                { label: 'GPT-4.1', value: 'gpt-4.1' },
                { label: 'o3-mini', value: 'o3-mini' },
                { label: 'Claude 3.5 Sonnet', value: 'claude-3-5-sonnet' },
                { label: 'Claude 3 Sonnet', value: 'claude-3-sonnet' },
                { label: 'Claude 3 Haiku', value: 'claude-3-haiku' },
                { label: 'Gemini 2.5 Pro', value: 'gemini-2.5-pro' },
                { label: 'Local Llama', value: 'ollama/llama3' },
            ];

            const selected = await vscode.window.showQuickPick(models, {
                placeHolder: 'Select model for this cell'
            });
            if (!selected) return;

            const cellIndex = editor.selection.start;
            const cell = editor.notebook.cellAt(cellIndex);
            const cellText = cell.document.getText();

            let newText: string;
            if (cellText.match(/%%llm\s+--model=\S+/)) {
                // Replace existing --model=
                newText = cellText.replace(/%%llm\s+--model=\S+/, `%%llm --model=${selected.value}`);
            } else if (cellText.startsWith('%%llm_gpt4') || cellText.startsWith('%%llm_claude')) {
                // Replace shorthand with explicit model
                newText = cellText.replace(/^%%llm_\w+/, `%%llm --model=${selected.value}`);
            } else if (cellText.startsWith('%%llm')) {
                // Insert --model= after %%llm
                newText = cellText.replace(/^%%llm/, `%%llm --model=${selected.value}`);
            } else {
                return; // Not an LLM cell
            }

            const edit = new vscode.WorkspaceEdit();
            const fullRange = new vscode.Range(
                cell.document.positionAt(0),
                cell.document.positionAt(cellText.length)
            );
            edit.replace(cell.document.uri, fullRange, newText);
            await vscode.workspace.applyEdit(edit);
        })
    );

    // Explain current cell
    context.subscriptions.push(
        vscode.commands.registerCommand('llm-kernel.explainCurrentCell', async () => {
            await provider.explainCurrentCell();
        })
    );

    // Refactor current cell
    context.subscriptions.push(
        vscode.commands.registerCommand('llm-kernel.refactorCurrentCell', async () => {
            await provider.refactorCurrentCell();
        })
    );

    // Switch model
    context.subscriptions.push(
        vscode.commands.registerCommand('llm-kernel.switchModel', async () => {
            await provider.switchModel();
        })
    );

    // Prune context
    context.subscriptions.push(
        vscode.commands.registerCommand('llm-kernel.pruneContext', async () => {
            await provider.pruneContext();
        })
    );

    // Compare models
    context.subscriptions.push(
        vscode.commands.registerCommand('llm-kernel.compareModels', async () => {
            await provider.compareModels();
        })
    );

    // Show dashboard
    context.subscriptions.push(
        vscode.commands.registerCommand('llm-kernel.showDashboard', async () => {
            await provider.showDashboard();
        })
    );

    // Setup for workspace
    context.subscriptions.push(
        vscode.commands.registerCommand('llm-kernel.setupForWorkspace', async () => {
            const bootstrapper = new KernelBootstrapper(context);
            const isLocalDev = await bootstrapper.isLocalDevelopment();
            if (isLocalDev) {
                await bootstrapper.bootstrapFromLocal();
            } else {
                await bootstrapper.bootstrapFromRepository();
            }
        })
    );

    // Add LLM cell command — inserts a Python cell with %%llm magic
    context.subscriptions.push(
        vscode.commands.registerCommand('llm-kernel.addLLMCell', async () => {
            const editor = vscode.window.activeNotebookEditor;
            if (!editor) {
                vscode.window.showWarningMessage('No active notebook editor found');
                return;
            }

            // Check current cell for model and carry forward
            const currentCell = editor.notebook.cellAt(editor.selection.start);
            const currentText = currentCell?.document.getText().trim() || '';
            const modelMatch = currentText.match(/%%llm\s+--model=(\S+)/);
            const content = modelMatch
                ? `%%llm --model=${modelMatch[1]}\n`
                : '%%llm\n';

            const position = editor.selection.end + 1;
            const newCell = new vscode.NotebookCellData(
                vscode.NotebookCellKind.Code,
                content,
                'llm-kernel'
            );

            const edit = new vscode.WorkspaceEdit();
            const notebookEdit = vscode.NotebookEdit.insertCells(position, [newCell]);
            edit.set(editor.notebook.uri, [notebookEdit]);

            await vscode.workspace.applyEdit(edit);
        })
    );

    // Create new LLM notebook command
    context.subscriptions.push(
        vscode.commands.registerCommand('llm-kernel.createNewLLMNotebook', async () => {
            await provider.createNewLLMNotebook();
        })
    );

    // Overlay toggle commands
    context.subscriptions.push(
        vscode.commands.registerCommand('llm-kernel.toggleOverlay', async () => {
            await overlayManager?.toggleOverlay();
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('llm-kernel.enableOverlay', async () => {
            const notebook = vscode.window.activeNotebookEditor?.notebook;
            if (notebook) {
                await overlayManager?.enableOverlay(notebook);
            }
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('llm-kernel.disableOverlay', async () => {
            const notebook = vscode.window.activeNotebookEditor?.notebook;
            if (notebook) {
                await overlayManager?.disableOverlay(notebook);
            }
        })
    );

    // Chat mode commands — insert and execute kernel magic commands
    const insertAndExecuteMagic = async (magic: string) => {
        const editor = vscode.window.activeNotebookEditor;
        if (!editor) {
            vscode.window.showWarningMessage('No active notebook editor found');
            return;
        }
        const position = editor.selection.end + 1;
        const newCell = new vscode.NotebookCellData(
            vscode.NotebookCellKind.Code,
            magic,
            'llm-kernel'
        );
        const edit = new vscode.WorkspaceEdit();
        const notebookEdit = vscode.NotebookEdit.insertCells(position, [newCell]);
        edit.set(editor.notebook.uri, [notebookEdit]);
        await vscode.workspace.applyEdit(edit);
        await vscode.commands.executeCommand('notebook.cell.execute', {
            ranges: [{ start: position, end: position + 1 }]
        });
    };

    context.subscriptions.push(
        vscode.commands.registerCommand('llm-kernel.toggleChatMode', async () => {
            await insertAndExecuteMagic('%llm_chat');
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('llm-kernel.enableChatMode', async () => {
            await insertAndExecuteMagic('%llm_chat on');
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('llm-kernel.disableChatMode', async () => {
            await insertAndExecuteMagic('%llm_chat off');
        })
    );

    // === Context commands ===
    context.subscriptions.push(
        vscode.commands.registerCommand('llm-kernel.showContext', async () => {
            await provider.executeSimpleMagic('%llm_context');
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('llm-kernel.resetContext', async () => {
            const confirm = await vscode.window.showWarningMessage(
                'Reset all LLM context? This cannot be undone.',
                { modal: true },
                'Reset'
            );
            if (confirm === 'Reset') {
                await provider.executeSimpleMagic('%llm_context_reset');
            }
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('llm-kernel.saveContext', async () => {
            const name = await vscode.window.showInputBox({
                prompt: 'Name for saved context',
                placeHolder: 'e.g., my-analysis-session'
            });
            if (name) {
                await provider.executeSimpleMagic(`%llm_context_save ${name}`);
            }
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('llm-kernel.loadContext', async () => {
            const name = await vscode.window.showInputBox({
                prompt: 'Name of context to load',
                placeHolder: 'e.g., my-analysis-session'
            });
            if (name) {
                await provider.executeSimpleMagic(`%llm_context_load ${name}`);
            }
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('llm-kernel.showHistory', async () => {
            await provider.executeSimpleMagic('%llm_history');
        })
    );

    // === MCP commands ===
    context.subscriptions.push(
        vscode.commands.registerCommand('llm-kernel.mcpConnect', async () => {
            await provider.executeSimpleMagic('%llm_mcp_connect');
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('llm-kernel.mcpTools', async () => {
            await provider.executeSimpleMagic('%llm_mcp_tools');
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('llm-kernel.mcpQuery', async () => {
            const query = await vscode.window.showInputBox({
                prompt: 'Enter MCP query',
                placeHolder: 'e.g., Search for files matching...',
                ignoreFocusOut: true
            });
            if (query) {
                await provider.insertMagicCellPublic(`%%llm_mcp\n${query}`);
            }
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('llm-kernel.mcpConfig', async () => {
            await provider.executeSimpleMagic('%llm_mcp_config');
        })
    );

    // === Multimodal commands ===
    context.subscriptions.push(
        vscode.commands.registerCommand('llm-kernel.pasteImage', async () => {
            await provider.executeSimpleMagic('%llm_paste');
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('llm-kernel.addImage', async () => {
            const uri = await vscode.window.showOpenDialog({
                canSelectMany: false,
                filters: { 'Images': ['png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp'] },
                title: 'Select image to add to context'
            });
            if (uri && uri[0]) {
                await provider.executeSimpleMagic(`%llm_image ${uri[0].fsPath}`);
            }
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('llm-kernel.addPdf', async () => {
            const uri = await vscode.window.showOpenDialog({
                canSelectMany: false,
                filters: { 'PDF': ['pdf'] },
                title: 'Select PDF to add to context'
            });
            if (uri && uri[0]) {
                await provider.executeSimpleMagic(`%llm_pdf ${uri[0].fsPath}`);
            }
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('llm-kernel.visionQuery', async () => {
            const query = await vscode.window.showInputBox({
                prompt: 'Enter vision query (images must be added first)',
                placeHolder: 'e.g., Describe what you see in this image',
                ignoreFocusOut: true
            });
            if (query) {
                await provider.insertMagicCellPublic(`%%llm_vision\n${query}`);
            }
        })
    );

    // === Config/info commands ===
    context.subscriptions.push(
        vscode.commands.registerCommand('llm-kernel.showCost', async () => {
            await provider.executeSimpleMagic('%llm_cost');
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('llm-kernel.showTokenCount', async () => {
            await provider.executeSimpleMagic('%llm_token_count');
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('llm-kernel.showConfig', async () => {
            await provider.executeSimpleMagic('%llm_config');
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('llm-kernel.showContextWindow', async () => {
            await provider.executeSimpleMagic('%llm_context_window');
        })
    );
}

function startUpdateChecker(bootstrapper: KernelBootstrapper) {
    // Skip update checks in local development mode
    bootstrapper.isLocalDevelopment().then(isLocal => {
        if (isLocal) {
            return;
        }

        // Check for updates every 24 hours
        const checkInterval = 24 * 60 * 60 * 1000;

        setInterval(async () => {
            const config = vscode.workspace.getConfiguration('llm-kernel');
            if (config.get('checkForUpdates', true)) {
                const hasUpdate = await bootstrapper.checkForUpdates();
                if (hasUpdate) {
                    const choice = await vscode.window.showInformationMessage(
                        'LLM Kernel update available!',
                        'Update Now', 'Later', 'Auto-update'
                    );

                    if (choice === 'Update Now') {
                        await bootstrapper.performUpdate();
                    } else if (choice === 'Auto-update') {
                        await config.update('autoUpdate', true, true);
                        await bootstrapper.performUpdate();
                    }
                }
            }
        }, checkInterval);

        // Also check on startup (after a delay)
        setTimeout(async () => {
            const config = vscode.workspace.getConfiguration('llm-kernel');
            if (config.get('checkForUpdates', true) && !config.get('autoUpdate', false)) {
                const hasUpdate = await bootstrapper.checkForUpdates();
                if (hasUpdate) {
                    vscode.window.showInformationMessage(
                        'LLM Kernel update available!',
                        'Update Now', 'Later'
                    ).then(async (choice) => {
                        if (choice === 'Update Now') {
                            await bootstrapper.performUpdate();
                        }
                    });
                }
            }
        }, 5000);
    });
}
