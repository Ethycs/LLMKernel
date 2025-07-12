import * as vscode from 'vscode';

export class UniversalLLMController {
    private readonly _controller: vscode.NotebookController;
    private _executionOrder = 0;

    constructor() {
        this._controller = vscode.notebooks.createNotebookController(
            'llm-universal-controller',
            'jupyter-notebook', // Works with standard Jupyter notebooks
            'Universal LLM Controller'
        );

        this._controller.supportedLanguages = [
            'python', 'javascript', 'typescript', 'r', 'julia', 'scala', 'java',
            'rust', 'go', 'sql', 'markdown', 'llm-query'
        ];
        
        this._controller.supportsExecutionOrder = true;
        this._controller.description = 'Execute code with any kernel + LLM queries without kernel dependencies';
        this._controller.detail = 'Works standalone or overlays existing kernels. Use %%llm for AI queries.';
        this._controller.executeHandler = this._execute.bind(this);
        
        // Make it appear as a lightweight option
        this._controller.label = 'Universal (Code + LLM)';
    }

    private async _execute(
        cells: vscode.NotebookCell[],
        notebook: vscode.NotebookDocument,
        controller: vscode.NotebookController
    ): Promise<void> {
        for (const cell of cells) {
            await this._executeCell(cell, notebook);
        }
    }

    private async _executeCell(
        cell: vscode.NotebookCell,
        notebook: vscode.NotebookDocument
    ): Promise<void> {
        const execution = this._controller.createNotebookCellExecution(cell);
        execution.executionOrder = ++this._executionOrder;
        execution.start(Date.now());

        try {
            const source = cell.document.getText().trim();
            
            if (this.isLLMQuery(source)) {
                await this.executeLLMQuery(cell, execution, source, notebook);
            } else {
                await this.executeCodeCell(cell, execution, source, notebook);
            }
        } catch (error) {
            execution.replaceOutput([
                new vscode.NotebookCellOutput([
                    vscode.NotebookCellOutputItem.error({
                        name: error instanceof Error ? error.constructor.name : 'error',
                        message: error instanceof Error ? error.message : String(error),
                        stack: error instanceof Error ? error.stack : ''
                    })
                ])
            ]);
        }

        execution.end(true, Date.now());
    }

    private isLLMQuery(source: string): boolean {
        const llmPatterns = [
            /^%%llm\b/m,
            /^%llm\b/m,
            /^@llm\b/m,
            /^#\s*@llm\b/m,
            /^\/\/\s*@llm\b/m,
            /^\s*LLM\(/m
        ];
        
        return llmPatterns.some(pattern => pattern.test(source));
    }

    private async executeLLMQuery(
        cell: vscode.NotebookCell,
        execution: vscode.NotebookCellExecution,
        source: string,
        notebook: vscode.NotebookDocument
    ): Promise<void> {
        // Parse LLM command
        const llmRequest = this.parseLLMCommand(source, cell.document.languageId);
        
        // Show progress
        execution.replaceOutput([
            new vscode.NotebookCellOutput([
                vscode.NotebookCellOutputItem.text(
                    `🤖 Querying ${llmRequest.model}...`,
                    'text/plain'
                )
            ])
        ]);

        try {
            // Extract context from notebook
            const context = await this.extractNotebookContext(notebook, cell, llmRequest.includeContext);
            
            // Execute LLM query
            const response = await this.queryLLM(llmRequest, context);
            
            // Create rich LLM response
            execution.replaceOutput([
                new vscode.NotebookCellOutput([
                    vscode.NotebookCellOutputItem.json({
                        model: llmRequest.model,
                        content: response.content,
                        cost: response.cost,
                        tokens: response.tokens,
                        timestamp: new Date().toISOString(),
                        language: llmRequest.language,
                        context_size: context.cellCount,
                        completion_tokens: response.completionTokens,
                        mode: 'universal'
                    }, 'application/llm-response')
                ])
            ]);

            // Update costs
            vscode.commands.executeCommand('llm-kernel.updateCost', response.cost);

        } catch (error) {
            execution.replaceOutput([
                new vscode.NotebookCellOutput([
                    vscode.NotebookCellOutputItem.json({
                        model: llmRequest.model,
                        content: '',
                        cost: 0,
                        tokens: 0,
                        timestamp: new Date().toISOString(),
                        error: error instanceof Error ? error.message : String(error),
                        mode: 'universal'
                    }, 'application/llm-response')
                ])
            ]);
        }
    }

    private async executeCodeCell(
        cell: vscode.NotebookCell,
        execution: vscode.NotebookCellExecution,
        source: string,
        notebook: vscode.NotebookDocument
    ): Promise<void> {
        const language = cell.document.languageId;
        
        // Check if there's an active kernel we can delegate to
        const activeKernel = await this.findActiveKernel(notebook);
        
        if (activeKernel) {
            // Delegate to active kernel
            await this.delegateToKernel(cell, execution, source, activeKernel);
        } else {
            // Execute in lightweight mode
            await this.executeLightweightCode(cell, execution, source, language);
        }
    }

    private async findActiveKernel(notebook: vscode.NotebookDocument): Promise<string | null> {
        // Try to find if there's already an active kernel session
        // This would integrate with VS Code's kernel management
        
        // For now, return null to use lightweight mode
        return null;
    }

    private async delegateToKernel(
        cell: vscode.NotebookCell,
        execution: vscode.NotebookCellExecution,
        source: string,
        kernelId: string
    ): Promise<void> {
        // Forward execution to the active kernel
        execution.replaceOutput([
            new vscode.NotebookCellOutput([
                vscode.NotebookCellOutputItem.text(
                    `[Delegated to ${kernelId}]\n${source}`,
                    'text/plain'
                )
            ])
        ]);

        // In real implementation, this would:
        // 1. Connect to the active kernel
        // 2. Send code for execution
        // 3. Stream back results
        await new Promise(resolve => setTimeout(resolve, 500));
    }

    private async executeLightweightCode(
        cell: vscode.NotebookCell,
        execution: vscode.NotebookCellExecution,
        source: string,
        language: string
    ): Promise<void> {
        // Lightweight execution for common scenarios
        
        if (language === 'markdown') {
            // Render markdown
            execution.replaceOutput([
                new vscode.NotebookCellOutput([
                    vscode.NotebookCellOutputItem.text(source, 'text/markdown')
                ])
            ]);
            return;
        }

        if (this.isSimpleExpression(source, language)) {
            // Handle simple expressions
            const result = await this.evaluateSimpleExpression(source, language);
            execution.replaceOutput([
                new vscode.NotebookCellOutput([
                    vscode.NotebookCellOutputItem.text(result, 'text/plain')
                ])
            ]);
            return;
        }

        // For complex code, suggest using a proper kernel
        execution.replaceOutput([
            new vscode.NotebookCellOutput([
                vscode.NotebookCellOutputItem.text(
                    `⚠️  Complex ${language} code detected. For full execution, select a ${language} kernel.\n\n` +
                    `Code to execute:\n${source}\n\n` +
                    `Tip: Use %%llm to ask AI about this code instead!`,
                    'text/plain'
                )
            ])
        ]);
    }

    private isSimpleExpression(source: string, language: string): boolean {
        // Detect simple expressions that can be evaluated without a full kernel
        const trimmed = source.trim();
        
        if (language === 'python') {
            // Simple math, string operations, etc.
            return /^[\d\s+\-*/().]+$/.test(trimmed) || 
                   /^["'].*["']$/.test(trimmed) ||
                   /^print\s*\(.+\)$/.test(trimmed);
        }
        
        if (language === 'javascript') {
            return /^[\d\s+\-*/().]+$/.test(trimmed) ||
                   /^["'`].*["'`]$/.test(trimmed) ||
                   /^console\.log\s*\(.+\)$/.test(trimmed);
        }

        return false;
    }

    private async evaluateSimpleExpression(source: string, language: string): Promise<string> {
        const trimmed = source.trim();
        
        if (language === 'python') {
            if (/^[\d\s+\-*/().]+$/.test(trimmed)) {
                try {
                    // Safe math evaluation
                    const result = Function(`"use strict"; return (${trimmed})`)();
                    return `${result}`;
                } catch {
                    return `SyntaxError: Invalid expression`;
                }
            }
            
            if (/^print\s*\((.+)\)$/.test(trimmed)) {
                const match = trimmed.match(/^print\s*\((.+)\)$/);
                if (match) {
                    const content = match[1].replace(/^["']|["']$/g, '');
                    return content;
                }
            }
        }

        if (language === 'javascript') {
            if (/^[\d\s+\-*/().]+$/.test(trimmed)) {
                try {
                    const result = Function(`"use strict"; return (${trimmed})`)();
                    return `${result}`;
                } catch {
                    return `SyntaxError: Invalid expression`;
                }
            }
            
            if (/^console\.log\s*\((.+)\)$/.test(trimmed)) {
                const match = trimmed.match(/^console\.log\s*\((.+)\)$/);
                if (match) {
                    const content = match[1].replace(/^["'`]|["'`]$/g, '');
                    return content;
                }
            }
        }

        return `Cannot evaluate: ${trimmed}`;
    }

    private parseLLMCommand(source: string, languageId: string): LLMRequest {
        const lines = source.split('\n');
        let model = 'gpt-4o-mini';
        let query = '';
        let includeContext = true;
        let temperature = 0.7;

        const firstLine = lines[0];
        if (firstLine.startsWith('%%llm') || firstLine.startsWith('%llm')) {
            // Parse magic command
            const modelMatch = firstLine.match(/--model[=\s]+([^\s]+)/);
            if (modelMatch) model = modelMatch[1];
            
            const tempMatch = firstLine.match(/--temperature[=\s]+([\d.]+)/);
            if (tempMatch) temperature = parseFloat(tempMatch[1]);
            
            if (firstLine.includes('--no-context')) includeContext = false;
            
            query = lines.slice(1).join('\n').trim();
        } else {
            // Comment-style or direct query
            query = source.replace(/^[#\/\*]*\s*@llm[^:\n]*:?\s*/gm, '').trim();
        }

        return {
            model,
            query: query || 'Please provide a query',
            language: languageId,
            includeContext,
            temperature,
            maxTokens: 1000
        };
    }

    private async extractNotebookContext(
        notebook: vscode.NotebookDocument,
        currentCell: vscode.NotebookCell,
        includeContext: boolean
    ): Promise<NotebookContext> {
        const context: NotebookContext = {
            language: currentCell.document.languageId,
            cellCount: 0,
            codeBlocks: [],
            variables: [],
            imports: [],
            functions: []
        };

        if (!includeContext) {
            return context;
        }

        // Extract context from all code cells before current cell
        const currentIndex = notebook.getCells().indexOf(currentCell);
        
        for (let i = 0; i < currentIndex; i++) {
            const cell = notebook.cellAt(i);
            
            if (cell.kind === vscode.NotebookCellKind.Code && 
                !this.isLLMQuery(cell.document.getText())) {
                
                const cellText = cell.document.getText();
                context.cellCount++;
                context.codeBlocks.push({
                    index: i,
                    content: cellText,
                    hasOutput: (cell.outputs?.length || 0) > 0
                });

                // Basic parsing for context
                this.parseCodeForContext(cellText, context.language, context);
            }
        }

        return context;
    }

    private parseCodeForContext(code: string, language: string, context: NotebookContext): void {
        // Simple context extraction
        if (language === 'python') {
            const imports = code.match(/^(?:import|from)\s+[\w.]+(?:\s+import\s+[\w,\s*]+)?/gm);
            if (imports) context.imports.push(...imports);
            
            const functions = code.match(/^def\s+(\w+)\s*\(/gm);
            if (functions) context.functions.push(...functions);
            
            const variables = code.match(/^(\w+)\s*=/gm);
            if (variables) context.variables.push(...variables);
        }
        // Add other languages as needed
    }

    private async queryLLM(request: LLMRequest, context: NotebookContext): Promise<LLMResponse> {
        // Mock LLM execution
        await new Promise(resolve => setTimeout(resolve, 1500));

        const responses = [
            `I can help with that! Based on your ${request.language} context with ${context.cellCount} cells...`,
            `Looking at your notebook, I see you have ${context.imports.length} imports and ${context.variables.length} variables...`,
            `Here's my analysis of your ${request.language} code...`
        ];

        const response = responses[Math.floor(Math.random() * responses.length)];
        const tokens = 100 + Math.floor(Math.random() * 200);
        const cost = tokens * 0.00002;

        return {
            content: response,
            tokens,
            cost,
            contextSize: context.cellCount,
            completionTokens: tokens
        };
    }

    dispose(): void {
        this._controller.dispose();
    }
}

interface LLMRequest {
    model: string;
    query: string;
    language: string;
    includeContext: boolean;
    temperature: number;
    maxTokens: number;
}

interface LLMResponse {
    content: string;
    tokens: number;
    cost: number;
    contextSize: number;
    completionTokens: number;
}

interface NotebookContext {
    language: string;
    cellCount: number;
    codeBlocks: Array<{
        index: number;
        content: string;
        hasOutput: boolean;
    }>;
    variables: string[];
    imports: string[];
    functions: string[];
}