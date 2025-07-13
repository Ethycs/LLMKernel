import * as assert from 'assert';
import * as vscode from 'vscode';
import { LLMNotebookSerializer } from '../../serializers/llmNotebookSerializer';

suite('LLM Notebook Serializer Test Suite', () => {
    let serializer: LLMNotebookSerializer;

    setup(() => {
        serializer = new LLMNotebookSerializer();
    });

    test('Can deserialize a basic notebook', async () => {
        const notebookContent = {
            cells: [
                {
                    cell_type: 'code',
                    source: ['print("Hello World")'],
                    metadata: {}
                },
                {
                    cell_type: 'llm',
                    source: ['What is machine learning?'],
                    metadata: {
                        model: 'gpt-4o-mini'
                    }
                }
            ],
            metadata: {
                kernelspec: {
                    display_name: 'Python 3',
                    language: 'python',
                    name: 'python3'
                }
            },
            nbformat: 4,
            nbformat_minor: 5
        };

        const jsonBytes = new TextEncoder().encode(JSON.stringify(notebookContent));
        
        // Deserialize
        const notebookData = await serializer.deserializeNotebook(
            jsonBytes,
            new vscode.CancellationTokenSource().token
        );

        // Check cells
        assert.strictEqual(notebookData.cells.length, 2);
        
        // Check first cell (code)
        assert.strictEqual(notebookData.cells[0].kind, vscode.NotebookCellKind.Code);
        assert.strictEqual(notebookData.cells[0].value, 'print("Hello World")');
        assert.strictEqual(notebookData.cells[0].languageId, 'python');
        
        // Check second cell (LLM)
        assert.strictEqual(notebookData.cells[1].kind, vscode.NotebookCellKind.Code);
        assert.strictEqual(notebookData.cells[1].value, 'What is machine learning?');
        assert.strictEqual(notebookData.cells[1].metadata?.cellType, 'llm');
        assert.strictEqual(notebookData.cells[1].metadata?.llmCell, true);
    });

    test('Can serialize a notebook', async () => {
        // Create notebook data
        const cells = [
            new vscode.NotebookCellData(
                vscode.NotebookCellKind.Code,
                'import pandas as pd',
                'python'
            ),
            new vscode.NotebookCellData(
                vscode.NotebookCellKind.Code,
                '%%llm\nExplain pandas',
                'llm-query'
            )
        ];
        
        cells[1].metadata = {
            cellType: 'llm',
            llmCell: true,
            model: 'gpt-4o'
        };

        const notebookData = new vscode.NotebookData(cells);
        notebookData.metadata = {
            custom: {
                llm_kernel: {
                    version: '1.0.0',
                    default_model: 'gpt-4o-mini'
                }
            }
        };

        // Serialize
        const serialized = await serializer.serializeNotebook(
            notebookData,
            new vscode.CancellationTokenSource().token
        );

        // Parse the result
        const parsed = JSON.parse(new TextDecoder().decode(serialized));

        // Verify structure
        assert.strictEqual(parsed.cells.length, 2);
        assert.strictEqual(parsed.cells[0].cell_type, 'code');
        assert.strictEqual(parsed.cells[1].cell_type, 'llm');
        assert.deepStrictEqual(parsed.cells[0].source, ['import pandas as pd']);
        assert.deepStrictEqual(parsed.cells[1].source, ['%%llm\nExplain pandas']);
    });

    test('Handles notebook outputs correctly', async () => {
        const notebookContent = {
            cells: [{
                cell_type: 'code',
                source: ['1 + 1'],
                outputs: [{
                    output_type: 'execute_result',
                    data: {
                        'text/plain': '2'
                    },
                    execution_count: 1
                }],
                execution_count: 1,
                metadata: {}
            }],
            metadata: {},
            nbformat: 4,
            nbformat_minor: 5
        };

        const jsonBytes = new TextEncoder().encode(JSON.stringify(notebookContent));
        
        // Deserialize
        const notebookData = await serializer.deserializeNotebook(
            jsonBytes,
            new vscode.CancellationTokenSource().token
        );

        // Check output
        assert.strictEqual(notebookData.cells[0].outputs?.length, 1);
        
        const output = notebookData.cells[0].outputs![0];
        assert.ok(output.items.length > 0);
        
        // Check execution summary
        assert.ok(notebookData.cells[0].executionSummary);
        assert.strictEqual(notebookData.cells[0].executionSummary?.executionOrder, 1);
    });

    test('Handles empty notebook correctly', async () => {
        const emptyNotebook = {
            cells: [],
            metadata: {},
            nbformat: 4,
            nbformat_minor: 5
        };

        const jsonBytes = new TextEncoder().encode(JSON.stringify(emptyNotebook));
        
        // Should deserialize empty notebook without errors
        const notebookData = await serializer.deserializeNotebook(
            jsonBytes,
            new vscode.CancellationTokenSource().token
        );

        assert.strictEqual(notebookData.cells.length, 0);
        assert.ok(notebookData.metadata);
    });
});