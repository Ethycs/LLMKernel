import * as assert from 'assert';
import * as vscode from 'vscode';
import { ChatModeManager } from '../../services/chatModeManager';

suite('Chat Mode Test Suite', () => {
    let chatModeManager: ChatModeManager;

    setup(() => {
        chatModeManager = new ChatModeManager();
    });

    teardown(() => {
        chatModeManager.dispose();
    });

    test('Chat mode can be enabled for a notebook', async () => {
        // Create a test notebook
        const notebook = await vscode.workspace.openNotebookDocument('jupyter-notebook', {
            cells: [{
                kind: vscode.NotebookCellKind.Code,
                value: 'print("test")',
                languageId: 'python'
            }]
        });

        // Enable chat mode
        await chatModeManager.enableChatMode(notebook);
        
        // Verify chat mode is active
        assert.strictEqual(chatModeManager.isChatModeActive(notebook), true);
        
        // Clean up
        await vscode.commands.executeCommand('workbench.action.closeActiveEditor');
    });

    test('Chat mode can be disabled for a notebook', async () => {
        const notebook = await vscode.workspace.openNotebookDocument('jupyter-notebook', {
            cells: [{
                kind: vscode.NotebookCellKind.Code,
                value: 'print("test")',
                languageId: 'python'
            }]
        });

        // Enable then disable chat mode
        await chatModeManager.enableChatMode(notebook);
        await chatModeManager.disableChatMode(notebook);
        
        // Verify chat mode is not active
        assert.strictEqual(chatModeManager.isChatModeActive(notebook), false);
        
        await vscode.commands.executeCommand('workbench.action.closeActiveEditor');
    });

    test('Natural language detection works correctly', async () => {
        const notebook = await vscode.workspace.openNotebookDocument('jupyter-notebook', {
            cells: []
        });

        await chatModeManager.enableChatMode(notebook);

        // Test natural language queries
        const naturalQueries = [
            'What is machine learning?',
            'How do I optimize this function?',
            'Can you explain this code?',
            'Please help me understand this',
            'Why does this error occur?'
        ];

        for (const query of naturalQueries) {
            assert.strictEqual(
                chatModeManager.isNaturalLanguageQuery(query, notebook),
                true,
                `Failed to detect natural language: "${query}"`
            );
        }

        // Test code that should NOT be detected as natural language
        const codeExamples = [
            'import pandas as pd',
            'def function():',
            'x = 5',
            'for i in range(10):',
            '%%llm'
        ];

        for (const code of codeExamples) {
            assert.strictEqual(
                chatModeManager.isNaturalLanguageQuery(code, notebook),
                false,
                `Incorrectly detected code as natural language: "${code}"`
            );
        }

        await vscode.commands.executeCommand('workbench.action.closeActiveEditor');
    });

    test('Chat mode toggle works correctly', async () => {
        const notebook = await vscode.workspace.openNotebookDocument('jupyter-notebook', {
            cells: []
        });

        // Initially should be disabled
        assert.strictEqual(chatModeManager.isChatModeActive(notebook), false);

        // Toggle on
        await chatModeManager.toggleChatMode(notebook);
        assert.strictEqual(chatModeManager.isChatModeActive(notebook), true);

        // Toggle off
        await chatModeManager.toggleChatMode(notebook);
        assert.strictEqual(chatModeManager.isChatModeActive(notebook), false);

        await vscode.commands.executeCommand('workbench.action.closeActiveEditor');
    });

    test('Chat mode persists in notebook metadata', async () => {
        const notebook = await vscode.workspace.openNotebookDocument('jupyter-notebook', {
            cells: []
        });

        // Enable chat mode
        await chatModeManager.enableChatMode(notebook);

        // Check metadata
        assert.strictEqual(notebook.metadata?.llm_chat_mode?.enabled, true);
        assert.ok(notebook.metadata?.llm_chat_mode?.enabledAt);
        assert.strictEqual(notebook.metadata?.llm_chat_mode?.version, '1.0.0');

        await vscode.commands.executeCommand('workbench.action.closeActiveEditor');
    });
});