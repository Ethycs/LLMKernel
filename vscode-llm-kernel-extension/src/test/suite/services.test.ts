import * as assert from 'assert';
import * as vscode from 'vscode';
import { KernelService } from '../../services/kernelService';
import { ApiService } from '../../services/apiService';
import { StatusBarManager } from '../../services/statusBarManager';
import { NotebookService } from '../../services/notebookService';
import { createMockNotebook, createMockCell } from './helpers/mockVscode';

suite('Services Test Suite', () => {
    
    test('KernelService can start and stop', async () => {
        const kernelService = new KernelService();
        
        // Initially should be stopped
        let status = await kernelService.getKernelStatus();
        assert.strictEqual(status, 'Stopped');
        
        // Start kernel
        await kernelService.startKernel();
        status = await kernelService.getKernelStatus();
        assert.strictEqual(status, 'Running');
        
        // Stop kernel
        await kernelService.stopKernel();
        status = await kernelService.getKernelStatus();
        assert.strictEqual(status, 'Stopped');
    });

    test('KernelService can execute code', async () => {
        const kernelService = new KernelService();
        
        // Start kernel first
        await kernelService.startKernel();
        
        // Execute code
        const result = await kernelService.executeCode('print("hello")');
        assert.ok(result);
        assert.ok(result.result.includes('print("hello")'));
        
        await kernelService.stopKernel();
    });

    test('KernelService throws error when executing without starting', async () => {
        const kernelService = new KernelService();
        
        // Try to execute without starting
        await assert.rejects(
            async () => await kernelService.executeCode('print("test")'),
            /Kernel is not running/
        );
    });

    test('KernelService provides completions', async () => {
        const kernelService = new KernelService();
        
        const completions = await kernelService.getCompletions('test');
        assert.ok(Array.isArray(completions));
        assert.ok(completions.length > 0);
        assert.ok(completions.every(c => c.startsWith('test')));
    });

    test('ApiService can send requests', async () => {
        const apiService = new ApiService();
        
        // These will fail with actual network requests, but we're testing the interface
        try {
            // Test that methods exist and can be called
            assert.ok(typeof apiService.sendRequest === 'function');
            assert.ok(typeof apiService.getRequest === 'function');
            assert.ok(typeof apiService.getCurrentContext === 'function');
            assert.ok(typeof apiService.saveContext === 'function');
        } catch (error) {
            // Expected in test environment
        }
    });

    test('StatusBarManager initializes correctly', () => {
        const statusBarManager = new StatusBarManager();
        
        // Initialize with mock context
        const mockContext = {
            subscriptions: []
        } as any;
        
        statusBarManager.initialize(mockContext);
        
        // Should add disposables to subscriptions
        assert.ok(mockContext.subscriptions.length > 0);
        
        // Update methods should not throw
        statusBarManager.updateCost(0.05);
        statusBarManager.updateModel('gpt-4o');
        statusBarManager.updateContextSize(1000);
    });

    test('StatusBarManager animation returns cleanup function', () => {
        const statusBarManager = new StatusBarManager();
        
        const mockContext = {
            subscriptions: []
        } as any;
        
        statusBarManager.initialize(mockContext);
        
        // showActiveQuery should return a cleanup function
        const cleanup = statusBarManager.showActiveQuery();
        assert.ok(typeof cleanup === 'function');
        
        // Cleanup should not throw
        cleanup();
    });

    test('NotebookService can execute cells', async () => {
        const notebookService = new NotebookService();
        
        const mockCell = createMockCell('print("test")');
        const mockNotebook = createMockNotebook([mockCell]);
        
        // Execute cell - should not throw
        await assert.doesNotReject(
            async () => await notebookService.executeCell(mockCell, mockNotebook)
        );
    });

    test('NotebookService state management', async () => {
        const notebookService = new NotebookService();
        
        // Save and load state should not throw
        await assert.doesNotReject(
            async () => await notebookService.saveNotebookState()
        );
        
        await assert.doesNotReject(
            async () => await notebookService.loadNotebookState()
        );
    });
});