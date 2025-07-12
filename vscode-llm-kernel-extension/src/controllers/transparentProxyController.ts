import * as vscode from 'vscode';

export class TransparentProxyController {
    private readonly _controllers: Map<string, vscode.NotebookController> = new Map();
    private _executionOrder = 0;

    constructor() {
        this.setupKernelDetection();
    }

    private setupKernelDetection(): void {
        // Monitor available kernels and create proxy controllers
        vscode.workspace.onDidChangeConfiguration((e) => {
            if (e.affectsConfiguration('jupyter')) {
                this.refreshKernelControllers();
            }
        });

        // Initial setup
        this.refreshKernelControllers();
    }

    private async refreshKernelControllers(): Promise<void> {
        // Detect available Jupyter kernels
        const availableKernels = await this.detectAvailableKernels();
        
        for (const kernel of availableKernels) {
            if (!this._controllers.has(kernel.name)) {
                this.createProxyController(kernel);
            }
        }
    }

    private async detectAvailableKernels(): Promise<KernelSpec[]> {
        // In a real implementation, this would query jupyter kernelspec list
        // For now, return common kernels
        return [
            {
                name: 'python3-llm',
                displayName: 'Python 3 (LLM Enhanced)',
                language: 'python',
                originalKernel: 'python3'
            },
            {
                name: 'ir-llm',
                displayName: 'R (LLM Enhanced)',
                language: 'r',
                originalKernel: 'ir'
            },
            {
                name: 'julia-llm',
                displayName: 'Julia (LLM Enhanced)',
                language: 'julia',
                originalKernel: 'julia-1.6'
            },
            {
                name: 'javascript-llm',
                displayName: 'JavaScript (LLM Enhanced)',
                language: 'javascript',
                originalKernel: 'javascript'
            }
        ];
    }

    private createProxyController(kernelSpec: KernelSpec): void {
        const controller = vscode.notebooks.createNotebookController(
            kernelSpec.name,
            'jupyter-notebook',
            kernelSpec.displayName
        );

        controller.supportedLanguages = [kernelSpec.language, 'llm-query'];
        controller.supportsExecutionOrder = true;
        controller.description = `${kernelSpec.displayName} with AI assistance`;
        controller.executeHandler = this.createExecuteHandler(kernelSpec);

        // Add LLM-specific capabilities
        controller.detail = 'Enhanced with LLM capabilities - use %%llm for AI queries';
        
        this._controllers.set(kernelSpec.name, controller);
    }

    private createExecuteHandler(kernelSpec: KernelSpec) {
        return async (
            cells: vscode.NotebookCell[],
            notebook: vscode.NotebookDocument,
            controller: vscode.NotebookController
        ): Promise<void> => {
            for (const cell of cells) {
                await this.executeCell(cell, kernelSpec, controller);
            }
        };
    }

    private async executeCell(
        cell: vscode.NotebookCell,
        kernelSpec: KernelSpec,
        controller: vscode.NotebookController
    ): Promise<void> {
        const execution = controller.createNotebookCellExecution(cell);
        execution.executionOrder = ++this._executionOrder;
        execution.start(Date.now());

        try {
            const source = cell.document.getText().trim();
            
            if (this.isLLMQuery(source)) {
                await this.executeLLMQuery(cell, execution, source, kernelSpec);
            } else {
                await this.executeNativeCode(cell, execution, source, kernelSpec);
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
        kernelSpec: KernelSpec
    ): Promise<void> {
        // Parse LLM command
        const llmRequest = this.parseLLMCommand(source, kernelSpec.language);
        
        // Show progress
        execution.replaceOutput([
            new vscode.NotebookCellOutput([
                vscode.NotebookCellOutputItem.text(
                    `🤖 Querying ${llmRequest.model} for ${kernelSpec.language}...`,
                    'text/plain'
                )
            ])
        ]);

        try {
            // Get context from notebook
            const context = await this.extractNotebookContext(cell.notebook, kernelSpec.language);
            
            // Execute LLM query with context
            const response = await this.queryLLMWithContext(llmRequest, context);
            
            // Create rich LLM response output
            const llmOutput = new vscode.NotebookCellOutput([
                vscode.NotebookCellOutputItem.json({
                    model: llmRequest.model,
                    content: response.content,
                    cost: response.cost,
                    tokens: response.tokens,
                    timestamp: new Date().toISOString(),
                    language: kernelSpec.language,
                    kernel: kernelSpec.displayName,
                    context_size: context.cellCount,
                    completion_tokens: response.completionTokens
                }, 'application/llm-response')
            ]);

            execution.replaceOutput([llmOutput]);

            // Update extension state
            vscode.commands.executeCommand('llm-kernel.updateCost', response.cost);

        } catch (error) {
            const errorOutput = new vscode.NotebookCellOutput([
                vscode.NotebookCellOutputItem.json({
                    model: llmRequest.model,
                    content: '',
                    cost: 0,
                    tokens: 0,
                    timestamp: new Date().toISOString(),
                    language: kernelSpec.language,
                    error: error instanceof Error ? error.message : String(error)
                }, 'application/llm-response')
            ]);

            execution.replaceOutput([errorOutput]);
            throw error;
        }
    }

    private async executeNativeCode(
        cell: vscode.NotebookCell,
        execution: vscode.NotebookCellExecution,
        source: string,
        kernelSpec: KernelSpec
    ): Promise<void> {
        // Forward to the actual kernel
        // This is where we'd communicate with the real Jupyter kernel
        
        execution.replaceOutput([
            new vscode.NotebookCellOutput([
                vscode.NotebookCellOutputItem.text(
                    `[Proxied to ${kernelSpec.originalKernel}]\n${source}`,
                    'text/plain'
                )
            ])
        ]);

        // In a real implementation, this would:
        // 1. Connect to the actual Jupyter kernel using kernelSpec.originalKernel
        // 2. Send the code for execution
        // 3. Stream back results in real-time
        // 4. Handle errors and interruptions
        // 5. Maintain kernel state and variables

        // Mock successful execution
        await new Promise(resolve => setTimeout(resolve, 500));
    }

    private parseLLMCommand(source: string, language: string): LLMRequest {
        const lines = source.split('\n');
        let model = 'gpt-4o-mini';
        let query = '';
        let includeContext = true;

        const firstLine = lines[0];
        if (firstLine.startsWith('%%llm') || firstLine.startsWith('%llm')) {
            // Parse magic command parameters
            const modelMatch = firstLine.match(/--model[=\s]+([^\s]+)/);
            if (modelMatch) model = modelMatch[1];
            
            const noContextMatch = firstLine.includes('--no-context');
            if (noContextMatch) includeContext = false;
            
            query = lines.slice(1).join('\n').trim();
        } else {
            // Extract from comment-style or detect query
            query = source.replace(/^[#\/\*]*\s*@llm[^:\n]*:?\s*/gm, '').trim();
        }

        return {
            model,
            query: query || 'Please provide a query',
            language,
            includeContext,
            temperature: 0.7,
            maxTokens: 1000
        };
    }

    private async extractNotebookContext(
        notebook: vscode.NotebookDocument,
        language: string
    ): Promise<NotebookContext> {
        const context: NotebookContext = {
            language,
            cellCount: 0,
            codeBlocks: [],
            variables: [],
            imports: [],
            functions: []
        };

        for (let i = 0; i < notebook.cellCount; i++) {
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

                // Language-specific parsing
                await this.parseCodeForContext(cellText, language, context);
            }
        }

        return context;
    }

    private async parseCodeForContext(
        code: string,
        language: string,
        context: NotebookContext
    ): Promise<void> {
        switch (language) {
            case 'python':
                this.parsePythonContext(code, context);
                break;
            case 'r':
                this.parseRContext(code, context);
                break;
            case 'julia':
                this.parseJuliaContext(code, context);
                break;
            case 'javascript':
                this.parseJavaScriptContext(code, context);
                break;
        }
    }

    private parsePythonContext(code: string, context: NotebookContext): void {
        // Extract imports
        const importMatches = code.match(/^(?:import|from)\s+[\w.]+(?:\s+import\s+[\w,\s*]+)?/gm);
        if (importMatches) {
            context.imports.push(...importMatches);
        }

        // Extract function definitions
        const functionMatches = code.match(/^def\s+(\w+)\s*\(/gm);
        if (functionMatches) {
            context.functions.push(...functionMatches.map(m => m.replace(/^def\s+/, '').replace(/\s*\($/, '')));
        }

        // Extract variable assignments (basic)
        const varMatches = code.match(/^(\w+)\s*=/gm);
        if (varMatches) {
            context.variables.push(...varMatches.map(m => m.replace(/\s*=$/, '')));
        }
    }

    private parseRContext(code: string, context: NotebookContext): void {
        // Extract library calls
        const libraryMatches = code.match(/library\([\w"']+\)|require\([\w"']+\)/gm);
        if (libraryMatches) {
            context.imports.push(...libraryMatches);
        }

        // Extract function definitions
        const functionMatches = code.match(/(\w+)\s*<-\s*function\s*\(/gm);
        if (functionMatches) {
            context.functions.push(...functionMatches.map(m => m.split('<-')[0].trim()));
        }

        // Extract variable assignments
        const varMatches = code.match(/(\w+)\s*<-/gm);
        if (varMatches) {
            context.variables.push(...varMatches.map(m => m.replace(/\s*<-$/, '')));
        }
    }

    private parseJuliaContext(code: string, context: NotebookContext): void {
        // Extract using statements
        const usingMatches = code.match(/using\s+[\w,.]+/gm);
        if (usingMatches) {
            context.imports.push(...usingMatches);
        }

        // Extract function definitions
        const functionMatches = code.match(/function\s+(\w+)\s*\(/gm);
        if (functionMatches) {
            context.functions.push(...functionMatches.map(m => m.replace(/function\s+/, '').replace(/\s*\($/, '')));
        }
    }

    private parseJavaScriptContext(code: string, context: NotebookContext): void {
        // Extract imports/requires
        const importMatches = code.match(/(?:import|const|let|var)\s+.*?(?:from\s+['"][^'"]+['"]|require\s*\(\s*['"][^'"]+['"]\s*\))/gm);
        if (importMatches) {
            context.imports.push(...importMatches);
        }

        // Extract function definitions
        const functionMatches = code.match(/(?:function\s+(\w+)|const\s+(\w+)\s*=.*?=>|(\w+)\s*=.*?function)/gm);
        if (functionMatches) {
            context.functions.push(...functionMatches);
        }
    }

    private async queryLLMWithContext(
        request: LLMRequest,
        context: NotebookContext
    ): Promise<LLMResponse> {
        // Simulate API call
        await new Promise(resolve => setTimeout(resolve, 1000 + Math.random() * 2000));

        // In real implementation, this would:
        // 1. Format context for the specific LLM
        // 2. Send request to LLM service with context
        // 3. Handle streaming responses
        // 4. Calculate costs and token usage

        const contextPrompt = this.formatContextForLLM(context, request);
        const fullPrompt = `${contextPrompt}\n\nUser Query: ${request.query}`;

        // Mock response
        const responses = [
            `Based on your ${request.language} code context, here's what I found...`,
            `Looking at your notebook with ${context.cellCount} cells, I can help with...`,
            `Analyzing your ${request.language} variables and functions, here's my response...`,
        ];

        const randomResponse = responses[Math.floor(Math.random() * responses.length)];
        const tokenCount = Math.floor(100 + Math.random() * 300);
        const cost = tokenCount * 0.00002;

        return {
            content: randomResponse,
            tokens: tokenCount,
            cost: cost,
            contextSize: context.cellCount,
            completionTokens: tokenCount
        };
    }

    private formatContextForLLM(context: NotebookContext, request: LLMRequest): string {
        if (!request.includeContext) {
            return `Language: ${context.language}`;
        }

        let contextStr = `Language: ${context.language}\n`;
        
        if (context.imports.length > 0) {
            contextStr += `\nImports/Libraries:\n${context.imports.slice(0, 10).join('\n')}\n`;
        }
        
        if (context.variables.length > 0) {
            contextStr += `\nVariables: ${context.variables.slice(0, 20).join(', ')}\n`;
        }
        
        if (context.functions.length > 0) {
            contextStr += `\nFunctions: ${context.functions.slice(0, 10).join(', ')}\n`;
        }
        
        if (context.codeBlocks.length > 0) {
            contextStr += `\nRecent code (last 3 cells):\n`;
            const recentCells = context.codeBlocks.slice(-3);
            recentCells.forEach((block, i) => {
                contextStr += `\nCell ${block.index}:\n${block.content.substring(0, 200)}${block.content.length > 200 ? '...' : ''}\n`;
            });
        }

        return contextStr;
    }

    dispose(): void {
        for (const controller of this._controllers.values()) {
            controller.dispose();
        }
        this._controllers.clear();
    }
}

interface KernelSpec {
    name: string;
    displayName: string;
    language: string;
    originalKernel: string;
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