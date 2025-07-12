import * as vscode from 'vscode';

interface RawNotebookCell {
    cell_type: 'llm' | 'code' | 'markdown';
    language_id?: string;
    source: string[];
    metadata?: any;
    outputs?: any[];
    execution_count?: number | null;
}

interface RawNotebook {
    cells: RawNotebookCell[];
    metadata: {
        kernelspec?: {
            display_name: string;
            language: string;
            name: string;
        };
        language_info?: any;
        llm_kernel?: {
            version: string;
            default_model: string;
            session_id: string;
        };
    };
    nbformat: number;
    nbformat_minor: number;
}

export class LLMNotebookSerializer implements vscode.NotebookSerializer {
    async deserializeNotebook(
        content: Uint8Array,
        _token: vscode.CancellationToken
    ): Promise<vscode.NotebookData> {
        const contents = new TextDecoder().decode(content);
        
        let raw: RawNotebook;
        try {
            raw = JSON.parse(contents);
        } catch {
            raw = {
                cells: [],
                metadata: {
                    llm_kernel: {
                        version: '1.0.0',
                        default_model: 'gpt-4o-mini',
                        session_id: this.generateSessionId()
                    }
                },
                nbformat: 4,
                nbformat_minor: 2
            };
        }

        const cells = raw.cells.map(item => this.rawToNotebookCellData(item));
        
        const notebookData = new vscode.NotebookData(cells);
        notebookData.metadata = {
            ...raw.metadata,
            custom: {
                llm_kernel: raw.metadata.llm_kernel || {
                    version: '1.0.0',
                    default_model: 'gpt-4o-mini',
                    session_id: this.generateSessionId()
                }
            }
        };
        return notebookData;
    }

    async serializeNotebook(
        data: vscode.NotebookData,
        _token: vscode.CancellationToken
    ): Promise<Uint8Array> {
        const contents: RawNotebook = {
            cells: data.cells.map(cell => this.notebookCellDataToRaw(cell)),
            metadata: {
                ...data.metadata,
                llm_kernel: data.metadata?.custom?.llm_kernel || {
                    version: '1.0.0',
                    default_model: 'gpt-4o-mini',
                    session_id: this.generateSessionId()
                }
            },
            nbformat: 4,
            nbformat_minor: 2
        };

        return new TextEncoder().encode(JSON.stringify(contents, null, 2));
    }

    private rawToNotebookCellData(cell: RawNotebookCell): vscode.NotebookCellData {
        const cellData = new vscode.NotebookCellData(
            this.getCellKind(cell.cell_type),
            Array.isArray(cell.source) ? cell.source.join('') : cell.source,
            this.getLanguageId(cell)
        );

        // Set cell metadata
        cellData.metadata = {
            ...cell.metadata,
            cellType: cell.cell_type,
            llmCell: cell.cell_type === 'llm'
        };

        // Handle execution count
        if (cell.execution_count !== undefined && cell.execution_count !== null) {
            cellData.executionSummary = {
                executionOrder: cell.execution_count,
                success: true
            };
        }

        // Convert outputs if present
        if (cell.outputs && cell.outputs.length > 0) {
            cellData.outputs = cell.outputs.map(output => this.convertOutput(output));
        }

        return cellData;
    }

    private notebookCellDataToRaw(cell: vscode.NotebookCellData): RawNotebookCell {
        const cellType = cell.metadata?.cellType || this.inferCellType(cell);
        
        return {
            cell_type: cellType,
            language_id: cellType === 'llm' ? 'llm-query' : cell.languageId,
            source: cell.value.split('\n').map((line, i, arr) => 
                i === arr.length - 1 ? line : line + '\n'
            ),
            metadata: cell.metadata || {},
            execution_count: cell.executionSummary?.executionOrder || null,
            outputs: cell.outputs?.map(output => this.convertOutputToRaw(output)) || []
        };
    }

    private getCellKind(cellType: string): vscode.NotebookCellKind {
        switch (cellType) {
            case 'markdown':
                return vscode.NotebookCellKind.Markup;
            case 'llm':
            case 'code':
            default:
                return vscode.NotebookCellKind.Code;
        }
    }

    private getLanguageId(cell: RawNotebookCell): string {
        if (cell.cell_type === 'llm') {
            return 'llm-query';
        }
        if (cell.cell_type === 'markdown') {
            return 'markdown';
        }
        return cell.language_id || 'python';
    }

    private inferCellType(cell: vscode.NotebookCellData): 'llm' | 'code' | 'markdown' {
        if (cell.metadata?.llmCell || cell.languageId === 'llm-query') {
            return 'llm';
        }
        if (cell.kind === vscode.NotebookCellKind.Markup) {
            return 'markdown';
        }
        
        // Check if cell contains LLM patterns
        const source = cell.value;
        const llmPatterns = [
            /^%%llm\b/m,
            /^%llm\b/m,
            /^@llm\b/m,
            /^#\s*@llm\b/m,
            /^\/\/\s*@llm\b/m
        ];
        
        if (llmPatterns.some(pattern => pattern.test(source))) {
            return 'llm';
        }
        
        return 'code';
    }

    private convertOutput(output: any): vscode.NotebookCellOutput {
        const items: vscode.NotebookCellOutputItem[] = [];

        if (output.output_type === 'execute_result' || output.output_type === 'display_data') {
            // Handle various MIME types
            for (const [mimeType, data] of Object.entries(output.data || {})) {
                if (mimeType === 'application/llm-response') {
                    items.push(vscode.NotebookCellOutputItem.json(data, mimeType));
                } else if (mimeType === 'text/plain') {
                    const text = Array.isArray(data) ? (data as string[]).join('') : data as string;
                    items.push(vscode.NotebookCellOutputItem.text(text));
                } else if (mimeType === 'text/html') {
                    const html = Array.isArray(data) ? (data as string[]).join('') : data as string;
                    items.push(vscode.NotebookCellOutputItem.text(html, 'text/html'));
                } else {
                    items.push(vscode.NotebookCellOutputItem.json(data, mimeType));
                }
            }
        } else if (output.output_type === 'stream') {
            const text = Array.isArray(output.text) ? output.text.join('') : output.text;
            items.push(vscode.NotebookCellOutputItem.stdout(text));
        } else if (output.output_type === 'error') {
            items.push(vscode.NotebookCellOutputItem.error({
                name: output.ename || 'Error',
                message: output.evalue || 'Unknown error',
                stack: Array.isArray(output.traceback) ? output.traceback.join('\n') : output.traceback
            }));
        }

        return new vscode.NotebookCellOutput(items, output.metadata);
    }

    private convertOutputToRaw(output: vscode.NotebookCellOutput): any {
        const outputData: any = {
            output_type: 'execute_result',
            data: {},
            metadata: output.metadata || {}
        };

        for (const item of output.items) {
            if (item.mime === 'application/vnd.code.notebook.error') {
                const error = JSON.parse(new TextDecoder().decode(item.data));
                return {
                    output_type: 'error',
                    ename: error.name,
                    evalue: error.message,
                    traceback: error.stack ? error.stack.split('\n') : []
                };
            } else if (item.mime === 'application/vnd.code.notebook.stdout') {
                return {
                    output_type: 'stream',
                    name: 'stdout',
                    text: new TextDecoder().decode(item.data)
                };
            } else if (item.mime === 'application/vnd.code.notebook.stderr') {
                return {
                    output_type: 'stream',
                    name: 'stderr',
                    text: new TextDecoder().decode(item.data)
                };
            } else {
                try {
                    const data = item.mime.startsWith('application/') 
                        ? JSON.parse(new TextDecoder().decode(item.data))
                        : new TextDecoder().decode(item.data);
                    outputData.data[item.mime] = data;
                } catch {
                    outputData.data[item.mime] = new TextDecoder().decode(item.data);
                }
            }
        }

        return outputData;
    }

    private generateSessionId(): string {
        return `llm_session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    // Helper method to create new LLM notebook
    static createNewLLMNotebook(): vscode.NotebookData {
        const welcomeCell = new vscode.NotebookCellData(
            vscode.NotebookCellKind.Code,
            'Welcome to LLM Enhanced Notebook!\n\nThis notebook supports:\n- Regular code cells (Python, R, Julia, etc.)\n- LLM query cells for AI assistance\n- Mixed execution with context sharing\n\nPress Ctrl+Shift+L to add an LLM query cell.',
            'llm-query'
        );
        
        welcomeCell.metadata = {
            cellType: 'llm',
            llmCell: true,
            isWelcome: true
        };

        const notebookData = new vscode.NotebookData([welcomeCell]);
        notebookData.metadata = {
            custom: {
                llm_kernel: {
                    version: '1.0.0',
                    default_model: 'gpt-4o-mini',
                    session_id: `llm_session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
                }
            }
        };
        return notebookData;
    }
}