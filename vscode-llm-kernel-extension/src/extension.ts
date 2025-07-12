import * as vscode from 'vscode';
import { registerKernelCommands } from './commands/kernelCommands';
import { registerContextCommands } from './commands/contextCommands';
import { registerNotebookCommands } from './commands/notebookCommands';
import { KernelProvider } from './providers/kernelProvider';
import { ContextProvider } from './providers/contextProvider';
import { CompletionProvider } from './providers/completionProvider';
import { KernelBootstrapper } from './services/kernelBootstrapper';
import { StatusBarManager } from './services/statusBarManager';
import { UniversalLLMProvider } from './providers/universalLLMProvider';
import { ApiService } from './services/apiService';
import { KernelService } from './services/kernelService';
import { LLMNotebookController } from './controllers/llmNotebookController';
import { LLMNotebookSerializer } from './serializers/llmNotebookSerializer';
import { TransparentProxyController } from './controllers/transparentProxyController';
import { UniversalLLMController } from './controllers/universalLLMController';
import { LLMOverlayManager } from './services/llmOverlayManager';

export async function activate(context: vscode.ExtensionContext) {
    const config = vscode.workspace.getConfiguration('llm-kernel');
    const bootstrapper = new KernelBootstrapper(context);
    const statusBarManager = new StatusBarManager();
    const universalProvider = new UniversalLLMProvider();
    
    // Check if kernel is installed and bootstrap if needed
    const isInstalled = await bootstrapper.isKernelInstalled();
    
    if (!isInstalled) {
        if (config.get('autoInstall', true)) {
            // Auto-install on first use
            await bootstrapper.bootstrapFromRepository();
        } else {
            // Show install prompt
            const install = await vscode.window.showInformationMessage(
                '🤖 LLM Kernel not found. Install now?',
                'Install', 'Not now'
            );
            
            if (install === 'Install') {
                await bootstrapper.bootstrapFromRepository();
            }
        }
    } else {
        // Check for updates
        if (config.get('autoUpdate', false)) {
            const hasUpdate = await bootstrapper.checkForUpdates();
            if (hasUpdate) {
                await bootstrapper.performUpdate();
            }
        }
    }
    
    // Initialize providers and controllers
    const apiService = new ApiService();
    const kernelService = new KernelService();
    const kernelProvider = new KernelProvider();
    const contextProvider = new ContextProvider(apiService);
    const completionProvider = new CompletionProvider(kernelService);
    const llmController = new LLMNotebookController();
    const llmSerializer = new LLMNotebookSerializer();
    const universalController = new UniversalLLMController();
    const overlayManager = new LLMOverlayManager();

    // Register notebook components
    context.subscriptions.push(llmController);
    context.subscriptions.push(universalController);
    context.subscriptions.push(overlayManager);
    context.subscriptions.push(
        vscode.workspace.registerNotebookSerializer('llm-notebook', llmSerializer)
    );

    // Register commands
    context.subscriptions.push(registerKernelCommands(kernelProvider));
    context.subscriptions.push(registerContextCommands(context));
    context.subscriptions.push(registerNotebookCommands(kernelProvider));
    
    // Register universal LLM commands
    registerUniversalLLMCommands(context, universalProvider, overlayManager);
    
    // Register completion provider for multiple languages
    const languages = ['python', 'javascript', 'typescript', 'r', 'julia', 'llm-query'];
    languages.forEach(lang => {
        context.subscriptions.push(
            vscode.languages.registerCompletionItemProvider(lang, completionProvider)
        );
    });
    
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
    
    // Show welcome message
    if (context.globalState.get('firstActivation', true)) {
        vscode.window.showInformationMessage(
            '🚀 LLM Kernel Extension activated! Press Ctrl+Shift+L to add LLM queries to any notebook.',
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

function registerUniversalLLMCommands(context: vscode.ExtensionContext, provider: UniversalLLMProvider, overlayManager: LLMOverlayManager) {
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
            await bootstrapper.bootstrapFromRepository();
        })
    );
    
    // Add LLM cell command
    context.subscriptions.push(
        vscode.commands.registerCommand('llm-kernel.addLLMCell', async () => {
            const editor = vscode.window.activeNotebookEditor;
            if (!editor) {
                vscode.window.showWarningMessage('No active notebook editor found');
                return;
            }
            
            const position = editor.selection.end + 1;
            const newCell = new vscode.NotebookCellData(
                vscode.NotebookCellKind.Code,
                'Ask me anything!',
                'llm-query'
            );
            
            newCell.metadata = {
                cellType: 'llm',
                llmCell: true,
                model: vscode.workspace.getConfiguration('llm-kernel').get('defaultModel', 'gpt-4o-mini'),
                timestamp: new Date().toISOString()
            };
            
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
            await overlayManager.toggleOverlay();
        })
    );
    
    context.subscriptions.push(
        vscode.commands.registerCommand('llm-kernel.enableOverlay', async () => {
            const notebook = vscode.window.activeNotebookEditor?.notebook;
            if (notebook) {
                await overlayManager.enableOverlay(notebook);
            }
        })
    );
    
    context.subscriptions.push(
        vscode.commands.registerCommand('llm-kernel.disableOverlay', async () => {
            const notebook = vscode.window.activeNotebookEditor?.notebook;
            if (notebook) {
                await overlayManager.disableOverlay(notebook);
            }
        })
    );

    // Chat mode commands
    context.subscriptions.push(
        vscode.commands.registerCommand('llm-kernel.toggleChatMode', async () => {
            await overlayManager.toggleChatMode();
        })
    );
    
    context.subscriptions.push(
        vscode.commands.registerCommand('llm-kernel.enableChatMode', async () => {
            const notebook = vscode.window.activeNotebookEditor?.notebook;
            if (notebook) {
                await overlayManager.enableChatMode(notebook);
            }
        })
    );
    
    context.subscriptions.push(
        vscode.commands.registerCommand('llm-kernel.disableChatMode', async () => {
            const notebook = vscode.window.activeNotebookEditor?.notebook;
            if (notebook) {
                await overlayManager.disableChatMode(notebook);
            }
        })
    );
}

function startUpdateChecker(bootstrapper: KernelBootstrapper) {
    // Check for updates every 24 hours
    const checkInterval = 24 * 60 * 60 * 1000; // 24 hours
    
    setInterval(async () => {
        const config = vscode.workspace.getConfiguration('llm-kernel');
        if (config.get('checkForUpdates', true)) {
            const hasUpdate = await bootstrapper.checkForUpdates();
            if (hasUpdate) {
                const choice = await vscode.window.showInformationMessage(
                    '🔄 LLM Kernel update available!',
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
                    '🔄 LLM Kernel update available!',
                    'Update Now', 'Later'
                ).then(async (choice) => {
                    if (choice === 'Update Now') {
                        await bootstrapper.performUpdate();
                    }
                });
            }
        }
    }, 5000); // Check 5 seconds after activation
}