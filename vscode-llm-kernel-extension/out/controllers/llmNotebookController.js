"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || function (mod) {
    if (mod && mod.__esModule) return mod;
    var result = {};
    if (mod != null) for (var k in mod) if (k !== "default" && Object.prototype.hasOwnProperty.call(mod, k)) __createBinding(result, mod, k);
    __setModuleDefault(result, mod);
    return result;
};
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.LLMNotebookController = void 0;
const vscode = __importStar(require("vscode"));
class LLMNotebookController {
    constructor() {
        this.controllerId = 'llm-controller';
        this.notebookType = 'llm-notebook';
        this.label = 'LLM Controller';
        this.supportedLanguages = ['llm-query', 'python', 'javascript', 'typescript', 'r', 'julia'];
        this._executionOrder = 0;
        this._controller = vscode.notebooks.createNotebookController(this.controllerId, this.notebookType, this.label);
        this._controller.supportedLanguages = this.supportedLanguages;
        this._controller.supportsExecutionOrder = true;
        this._controller.description = 'Execute LLM queries and code cells with enhanced AI capabilities';
        this._controller.executeHandler = this._execute.bind(this);
    }
    _execute(cells, _notebook, _controller) {
        return __awaiter(this, void 0, void 0, function* () {
            for (let cell of cells) {
                yield this._doExecution(cell);
            }
        });
    }
    _doExecution(cell) {
        return __awaiter(this, void 0, void 0, function* () {
            const execution = this._controller.createNotebookCellExecution(cell);
            execution.executionOrder = ++this._executionOrder;
            execution.start(Date.now());
            try {
                const source = cell.document.getText();
                if (this.isLLMCell(cell) || this.containsLLMQuery(source)) {
                    yield this.executeLLMQuery(cell, execution, source);
                }
                else {
                    yield this.executeCodeCell(cell, execution, source);
                }
            }
            catch (error) {
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
        });
    }
    isLLMCell(cell) {
        var _a, _b;
        return cell.document.languageId === 'llm-query' ||
            ((_a = cell.metadata) === null || _a === void 0 ? void 0 : _a.cellType) === 'llm' ||
            ((_b = cell.metadata) === null || _b === void 0 ? void 0 : _b.llmCell) === true;
    }
    containsLLMQuery(source) {
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
    executeLLMQuery(cell, execution, source) {
        return __awaiter(this, void 0, void 0, function* () {
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
                        vscode.NotebookCellOutputItem.text(`🤖 Querying ${llmRequest.model}${dots}${spaces}`, 'text/plain')
                    ])
                ]);
                dotCount++;
            }, 500);
            try {
                // Simulate LLM execution (replace with actual LLM call)
                const response = yield this.queryLLM(llmRequest);
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
            }
            catch (error) {
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
        });
    }
    executeCodeCell(cell, execution, source) {
        return __awaiter(this, void 0, void 0, function* () {
            // For regular code cells, pass through to the underlying kernel
            const languageId = cell.document.languageId;
            // Simulate code execution (replace with actual kernel communication)
            execution.replaceOutput([
                new vscode.NotebookCellOutput([
                    vscode.NotebookCellOutputItem.text(`Executing ${languageId} code...\n${source}`, 'text/plain')
                ])
            ]);
            // In a real implementation, this would:
            // 1. Forward the execution to the appropriate kernel (Python, R, etc.)
            // 2. Capture the output and return it
            // 3. Handle errors appropriately
        });
    }
    parseLLMCommand(source) {
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
        }
        else if (firstLine.includes('@llm')) {
            // Parse comment-style commands
            const modelMatch = firstLine.match(/model=([^\s,]+)/);
            if (modelMatch) {
                model = modelMatch[1];
            }
            query = lines.slice(1).join('\n').trim();
        }
        else {
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
    queryLLM(request) {
        return __awaiter(this, void 0, void 0, function* () {
            // Simulate API call delay
            yield new Promise(resolve => setTimeout(resolve, 1000 + Math.random() * 2000));
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
        });
    }
    dispose() {
        this._controller.dispose();
    }
}
exports.LLMNotebookController = LLMNotebookController;
