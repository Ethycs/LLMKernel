import * as assert from 'assert';
import * as vscode from 'vscode';
import { KernelProvider } from '../../providers/kernelProvider';
import { ContextProvider } from '../../providers/contextProvider';
import { CompletionProvider } from '../../providers/completionProvider';
import { ApiService } from '../../services/apiService';
import { KernelService } from '../../services/kernelService';

suite('Providers Test Suite', () => {
    
    test('KernelProvider can start and stop kernel', async () => {
        const kernelProvider = new KernelProvider();
        
        // Start kernel
        await kernelProvider.startKernel();
        
        // Check status
        const status = await kernelProvider.getKernelStatus();
        assert.strictEqual(status, 'Running');
        
        // Stop kernel
        await kernelProvider.stopKernel();
        
        // Check status again
        const statusAfterStop = await kernelProvider.getKernelStatus();
        assert.strictEqual(statusAfterStop, 'Stopped');
    });

    test('ContextProvider can save and load context', async () => {
        const apiService = new ApiService();
        const contextProvider = new ContextProvider(apiService);
        
        // Mock save and load - in real tests these would interact with actual API
        const testContextName = 'test-context';
        
        // Save context should not throw
        await assert.doesNotReject(
            async () => await contextProvider.saveContext(testContextName)
        );
        
        // Load context should not throw
        await assert.doesNotReject(
            async () => await contextProvider.loadContext(testContextName)
        );
    });

    test('CompletionProvider provides completions for LLM queries', async () => {
        const kernelService = new KernelService();
        const completionProvider = new CompletionProvider(kernelService);
        
        // Create a mock document
        const position = new vscode.Position(0, 5);
        const document = {
            lineAt: (line: number) => ({
                text: '%llm test',
                firstNonWhitespaceCharacterIndex: 0
            }),
            getText: (range?: vscode.Range) => '%llm test'
        } as any;
        
        const token = new vscode.CancellationTokenSource().token;
        const context = {
            triggerKind: vscode.CompletionTriggerKind.Invoke,
            triggerCharacter: undefined
        };
        
        // Get completions
        const completions = await completionProvider.provideCompletionItems(
            document,
            position,
            token,
            context
        );
        
        // Should return array of completion items
        assert.ok(Array.isArray(completions));
        if (Array.isArray(completions)) {
            assert.ok(completions.length > 0);
            
            // Check completion items structure
            const firstItem = completions[0];
            assert.ok(firstItem instanceof vscode.CompletionItem);
            assert.ok(firstItem.label);
            assert.ok(firstItem.detail);
        }
    });

    test('CompletionProvider returns undefined for non-LLM queries', async () => {
        const kernelService = new KernelService();
        const completionProvider = new CompletionProvider(kernelService);
        
        // Create a mock document without %llm prefix
        const position = new vscode.Position(0, 10);
        const document = {
            lineAt: (line: number) => ({
                text: 'print("hello")',
                firstNonWhitespaceCharacterIndex: 0
            }),
            getText: (range?: vscode.Range) => 'print("hello")'
        } as any;
        
        const token = new vscode.CancellationTokenSource().token;
        const context = {
            triggerKind: vscode.CompletionTriggerKind.Invoke,
            triggerCharacter: undefined
        };
        
        // Get completions
        const completions = await completionProvider.provideCompletionItems(
            document,
            position,
            token,
            context
        );
        
        // Should return undefined for non-LLM queries
        assert.strictEqual(completions, undefined);
    });
});