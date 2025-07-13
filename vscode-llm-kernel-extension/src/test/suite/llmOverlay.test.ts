import * as assert from 'assert';
import * as vscode from 'vscode';
import { LLMOverlayManager } from '../../services/llmOverlayManager';

suite('LLM Overlay Test Suite', () => {
    let overlayManager: LLMOverlayManager;

    setup(() => {
        overlayManager = new LLMOverlayManager();
    });

    teardown(() => {
        overlayManager.dispose();
    });

    test('Overlay can be enabled for a notebook', async () => {
        const notebook = await vscode.workspace.openNotebookDocument('jupyter-notebook', {
            cells: [{
                kind: vscode.NotebookCellKind.Code,
                value: 'print("test")',
                languageId: 'python'
            }]
        });

        // Enable overlay
        await overlayManager.enableOverlay(notebook);
        
        // Verify overlay is active
        assert.strictEqual(overlayManager.isOverlayActive(notebook), true);
        
        // Check metadata
        assert.strictEqual(notebook.metadata?.llm_overlay?.enabled, true);
        assert.ok(notebook.metadata?.llm_overlay?.enabledAt);
        
        await vscode.commands.executeCommand('workbench.action.closeActiveEditor');
    });

    test('Overlay can be disabled', async () => {
        const notebook = await vscode.workspace.openNotebookDocument('jupyter-notebook', {
            cells: []
        });

        // Enable then disable
        await overlayManager.enableOverlay(notebook);
        await overlayManager.disableOverlay(notebook);
        
        // Verify overlay is not active
        assert.strictEqual(overlayManager.isOverlayActive(notebook), false);
        assert.strictEqual(notebook.metadata?.llm_overlay?.enabled, false);
        
        await vscode.commands.executeCommand('workbench.action.closeActiveEditor');
    });

    test('Overlay toggle works correctly', async () => {
        const notebook = await vscode.workspace.openNotebookDocument('jupyter-notebook', {
            cells: []
        });

        // Initially disabled
        assert.strictEqual(overlayManager.isOverlayActive(notebook), false);

        // Toggle on
        await overlayManager.toggleOverlay(notebook);
        assert.strictEqual(overlayManager.isOverlayActive(notebook), true);

        // Toggle off
        await overlayManager.toggleOverlay(notebook);
        assert.strictEqual(overlayManager.isOverlayActive(notebook), false);

        await vscode.commands.executeCommand('workbench.action.closeActiveEditor');
    });

    test('Multiple notebooks can have independent overlay states', async () => {
        const notebook1 = await vscode.workspace.openNotebookDocument('jupyter-notebook', {
            cells: []
        });
        const notebook2 = await vscode.workspace.openNotebookDocument('jupyter-notebook', {
            cells: []
        });

        // Enable overlay on first notebook only
        await overlayManager.enableOverlay(notebook1);

        // Verify states
        assert.strictEqual(overlayManager.isOverlayActive(notebook1), true);
        assert.strictEqual(overlayManager.isOverlayActive(notebook2), false);

        // Enable on second notebook
        await overlayManager.enableOverlay(notebook2);
        
        // Both should be active
        assert.strictEqual(overlayManager.isOverlayActive(notebook1), true);
        assert.strictEqual(overlayManager.isOverlayActive(notebook2), true);

        // Disable first
        await overlayManager.disableOverlay(notebook1);
        
        // Verify independent states
        assert.strictEqual(overlayManager.isOverlayActive(notebook1), false);
        assert.strictEqual(overlayManager.isOverlayActive(notebook2), true);

        await vscode.commands.executeCommand('workbench.action.closeAllEditors');
    });
});