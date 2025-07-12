import * as vscode from 'vscode';

export class LLMNotebookController {
    readonly controllerId = 'llm-controller';
    readonly notebookType = 'llm-notebook';
    readonly label = 'LLM Controller';
    readonly supportedLanguages = ['llm-query', 'python', 'javascript', 'typescript', 'r', 'julia'];

    private readonly _controller: vscode.NotebookController;
    private _executionOrder = 0;

    constructor() {
        this._controller = vscode.notebooks.createNotebookController(
            this.controllerId,
            this.notebookType,
            this.label
        );

        this._controller.supportedLanguages = this.supportedLanguages;
        this._controller.supportsExecutionOrder = true;
        this._controller.description = 'Execute LLM queries and code cells with enhanced AI capabilities';
        this._controller.executeHandler = this._execute.bind(this);
    }

    private async _execute(
        cells: vscode.NotebookCell[],
        _notebook: vscode.NotebookDocument,
        _controller: vscode.NotebookController
    ): Promise<void> {
        for (let cell of cells) {
            await this._doExecution(cell);
        }
    }

    private async _doExecution(cell: vscode.NotebookCell): Promise<void> {
        const execution = this._controller.createNotebookCellExecution(cell);
        execution.executionOrder = ++this._executionOrder;
        execution.start(Date.now());

        try {
            const source = cell.document.getText();
            
            if (this.isLLMCell(cell) || this.containsLLMQuery(source)) {
                await this.executeLLMQuery(cell, execution, source);
            } else {
                await this.executeCodeCell(cell, execution, source);
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

    private isLLMCell(cell: vscode.NotebookCell): boolean {
        return cell.document.languageId === 'llm-query' || 
               cell.metadata?.cellType === 'llm' ||
               cell.metadata?.llmCell === true;
    }

    private containsLLMQuery(source: string): boolean {
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
        source: string
    ): Promise<void> {
        execution.replaceOutput([
            new vscode.NotebookCellOutput([
                vscode.NotebookCellOutputItem.text('🤖 Processing LLM query...', 'text/plain')
            ])
        ]);

        // Parse LLM command
        const llmRequest = this.parseLLMCommand(source);
        
        // Show progress
        let dotCount = 0;
        const progressInterval = setInterval(() => {
            const dots = '.'.repeat((dotCount % 3) + 1);
            const spaces = ' '.repeat(3 - dots.length);
            execution.replaceOutput([
                new vscode.NotebookCellOutput([
                    vscode.NotebookCellOutputItem.text(
                        `🤖 Querying ${llmRequest.model}${dots}${spaces}`, 
                        'text/plain'
                    )
                ])
            ]);
            dotCount++;
        }, 500);

        try {
            // Simulate LLM execution (replace with actual LLM call)
            const response = await this.queryLLM(llmRequest);
            
            clearInterval(progressInterval);
            
            // Create rich LLM response output
            const llmOutput = new vscode.NotebookCellOutput([
                vscode.NotebookCellOutputItem.json({
                    model: llmRequest.model,
                    content: response.content,
                    cost: response.cost,
                    tokens: response.tokens,
                    timestamp: new Date().toISOString(),
                    streaming: false,
                    context_size: response.contextSize,
                    completion_tokens: response.completionTokens
                }, 'application/llm-response')
            ]);

            execution.replaceOutput([llmOutput]);

            // Update status bar
            vscode.commands.executeCommand('llm-kernel.updateCost', response.cost);

        } catch (error) {
            clearInterval(progressInterval);
            
            const errorOutput = new vscode.NotebookCellOutput([
                vscode.NotebookCellOutputItem.json({
                    model: llmRequest.model,
                    content: '',
                    cost: 0,
                    tokens: 0,
                    timestamp: new Date().toISOString(),
                    streaming: false,
                    error: error instanceof Error ? error.message : String(error)
                }, 'application/llm-response')
            ]);

            execution.replaceOutput([errorOutput]);
            throw error;
        }
    }

    private async executeCodeCell(
        cell: vscode.NotebookCell,
        execution: vscode.NotebookCellExecution,
        source: string
    ): Promise<void> {
        // For regular code cells, pass through to the underlying kernel
        const languageId = cell.document.languageId;
        
        // Simulate code execution (replace with actual kernel communication)
        execution.replaceOutput([
            new vscode.NotebookCellOutput([
                vscode.NotebookCellOutputItem.text(
                    `Executing ${languageId} code...\n${source}`, 
                    'text/plain'
                )
            ])
        ]);

        // In a real implementation, this would:
        // 1. Forward the execution to the appropriate kernel (Python, R, etc.)
        // 2. Capture the output and return it
        // 3. Handle errors appropriately
    }

    private parseLLMCommand(source: string): LLMRequest {
        const lines = source.split('\n');
        let model = 'gpt-4o-mini';
        let temperature = 0.7;
        let maxTokens = 1000;
        let query = '';

        // Parse first line for magic command parameters
        const firstLine = lines[0];
        if (firstLine.startsWith('%%llm') || firstLine.startsWith('%llm')) {
            const params = firstLine.match(/--(\w+)=([^\s]+)/g);
            if (params) {
                params.forEach(param => {
                    const [key, value] = param.split('=');
                    switch (key) {
                        case '--model':
                            model = value;
                            break;
                        case '--temperature':
                            temperature = parseFloat(value);
                            break;
                        case '--max-tokens':
                            maxTokens = parseInt(value);
                            break;
                    }
                });
            }
            query = lines.slice(1).join('\n').trim();
        } else if (firstLine.includes('@llm')) {
            // Parse comment-style commands
            const modelMatch = firstLine.match(/model=([^\s,]+)/);
            if (modelMatch) {
                model = modelMatch[1];
            }
            query = lines.slice(1).join('\n').trim();
        } else {
            // Entire cell is the query
            query = source.trim();
        }

        return {
            model,
            temperature,
            maxTokens,
            query: query || 'Please provide a query'
        };
    }

    private async queryLLM(request: LLMRequest): Promise<LLMResponse> {
        // Simulate API call delay
        await new Promise(resolve => setTimeout(resolve, 1000 + Math.random() * 2000));

        // In a real implementation, this would call your LLM service
        // For now, return a mock response
        const mockResponses = [
            "Here's a detailed explanation of the concept you asked about...",
            "Based on your requirements, here's a solution that should work well...",
            "I can help you with that! Let me break this down step by step...",
            "Here's an optimized approach to solve your problem..."
        ];

        const randomResponse = mockResponses[Math.floor(Math.random() * mockResponses.length)];
        const tokenCount = Math.floor(50 + Math.random() * 200);
        const cost = tokenCount * 0.00002; // Approximate cost calculation

        return {
            content: randomResponse,
            tokens: tokenCount,
            cost: cost,
            contextSize: 3,
            completionTokens: tokenCount
        };
    }

    dispose(): void {
        this._controller.dispose();
    }
}

interface LLMRequest {
    model: string;
    temperature: number;
    maxTokens: number;
    query: string;
}

interface LLMResponse {
    content: string;
    tokens: number;
    cost: number;
    contextSize: number;
    completionTokens: number;
}