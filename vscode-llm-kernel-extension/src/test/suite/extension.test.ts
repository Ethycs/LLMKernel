import * as assert from 'assert';
import * as vscode from 'vscode';

suite('Extension Test Suite', () => {
    
    test('Extension should be present', () => {
        assert.ok(vscode.extensions.getExtension('llm-kernel.llm-kernel-universal'));
    });

    test('Extension should activate', async () => {
        const extension = vscode.extensions.getExtension('llm-kernel.llm-kernel-universal');
        assert.ok(extension);
        
        // Activate the extension
        await extension!.activate();
        assert.strictEqual(extension!.isActive, true);
    });

    test('All expected commands should be registered', async () => {
        // Get all commands
        const allCommands = await vscode.commands.getCommands(true);
        
        // Commands that should be registered by our extension
        const expectedCommands = [
            'llm-kernel.setupForWorkspace',
            'llm-kernel.queryGPT4',
            'llm-kernel.queryClaude',
            'llm-kernel.switchModel',
            'llm-kernel.pruneContext',
            'llm-kernel.explainCurrentCell',
            'llm-kernel.refactorCurrentCell',
            'llm-kernel.compareModels',
            'llm-kernel.showDashboard',
            'llm-kernel.queryInCurrentCell',
            'llm-kernel.addLLMCell',
            'llm-kernel.createNewLLMNotebook',
            'llm-kernel.toggleChatMode',
            'llm-kernel.enableChatMode',
            'llm-kernel.disableChatMode',
            'llm-kernel.toggleOverlay',
            'llm-kernel.enableOverlay',
            'llm-kernel.disableOverlay'
        ];
        
        for (const cmd of expectedCommands) {
            assert.ok(
                allCommands.includes(cmd),
                `Command ${cmd} is not registered`
            );
        }
    });

    test('Configuration should have expected properties', () => {
        const config = vscode.workspace.getConfiguration('llm-kernel');
        
        // Check that configuration properties exist
        const expectedConfigs = [
            'repository',
            'autoInstall',
            'autoUpdate',
            'checkForUpdates',
            'updateChannel',
            'defaultModel',
            'contextStrategy',
            'modelPreferences',
            'showCostWarnings',
            'costThreshold',
            'streamResponses'
        ];
        
        for (const configKey of expectedConfigs) {
            const value = config.get(configKey);
            assert.ok(
                value !== undefined,
                `Configuration ${configKey} is not defined`
            );
        }
    });

    test('Create new LLM notebook command works', async () => {
        // Execute the command
        await vscode.commands.executeCommand('llm-kernel.createNewLLMNotebook');
        
        // Check that a notebook was opened
        const editor = vscode.window.activeNotebookEditor;
        assert.ok(editor, 'No notebook editor opened');
        
        if (editor) {
            // Check notebook metadata
            assert.ok(editor.notebook.metadata?.custom?.llm_kernel);
            
            // Clean up
            await vscode.commands.executeCommand('workbench.action.closeActiveEditor');
        }
    });
});