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
exports.UniversalLLMController = void 0;
const vscode = __importStar(require("vscode"));
class UniversalLLMController {
    constructor() {
        this._executionOrder = 0;
        this._controller = vscode.notebooks.createNotebookController('llm-universal-controller', 'jupyter-notebook', // Works with standard Jupyter notebooks
        'Universal LLM Controller');
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
    _execute(cells, notebook, controller) {
        return __awaiter(this, void 0, void 0, function* () {
            for (const cell of cells) {
                yield this._executeCell(cell, notebook);
            }
        });
    }
    _executeCell(cell, notebook) {
        return __awaiter(this, void 0, void 0, function* () {
            const execution = this._controller.createNotebookCellExecution(cell);
            execution.executionOrder = ++this._executionOrder;
            execution.start(Date.now());
            try {
                const source = cell.document.getText().trim();
                if (this.isLLMQuery(source)) {
                    yield this.executeLLMQuery(cell, execution, source, notebook);
                }
                else {
                    yield this.executeCodeCell(cell, execution, source, notebook);
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
    isLLMQuery(source) {
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
    executeLLMQuery(cell, execution, source, notebook) {
        return __awaiter(this, void 0, void 0, function* () {
            // Parse LLM command
            const llmRequest = this.parseLLMCommand(source, cell.document.languageId);
            // Show progress
            execution.replaceOutput([
                new vscode.NotebookCellOutput([
                    vscode.NotebookCellOutputItem.text(`🤖 Querying ${llmRequest.model}...`, 'text/plain')
                ])
            ]);
            try {
                // Extract context from notebook
                const context = yield this.extractNotebookContext(notebook, cell, llmRequest.includeContext);
                // Execute LLM query
                const response = yield this.queryLLM(llmRequest, context);
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
            }
            catch (error) {
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
        });
    }
    executeCodeCell(cell, execution, source, notebook) {
        return __awaiter(this, void 0, void 0, function* () {
            const language = cell.document.languageId;
            // Check if there's an active kernel we can delegate to
            const activeKernel = yield this.findActiveKernel(notebook);
            if (activeKernel) {
                // Delegate to active kernel
                yield this.delegateToKernel(cell, execution, source, activeKernel);
            }
            else {
                // Execute in lightweight mode
                yield this.executeLightweightCode(cell, execution, source, language);
            }
        });
    }
    findActiveKernel(notebook) {
        return __awaiter(this, void 0, void 0, function* () {
            // Try to find if there's already an active kernel session
            // This would integrate with VS Code's kernel management
            // For now, return null to use lightweight mode
            return null;
        });
    }
    delegateToKernel(cell, execution, source, kernelId) {
        return __awaiter(this, void 0, void 0, function* () {
            // Forward execution to the active kernel
            execution.replaceOutput([
                new vscode.NotebookCellOutput([
                    vscode.NotebookCellOutputItem.text(`[Delegated to ${kernelId}]\n${source}`, 'text/plain')
                ])
            ]);
            // In real implementation, this would:
            // 1. Connect to the active kernel
            // 2. Send code for execution
            // 3. Stream back results
            yield new Promise(resolve => setTimeout(resolve, 500));
        });
    }
    executeLightweightCode(cell, execution, source, language) {
        return __awaiter(this, void 0, void 0, function* () {
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
                const result = yield this.evaluateSimpleExpression(source, language);
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
                    vscode.NotebookCellOutputItem.text(`⚠️  Complex ${language} code detected. For full execution, select a ${language} kernel.\n\n` +
                        `Code to execute:\n${source}\n\n` +
                        `Tip: Use %%llm to ask AI about this code instead!`, 'text/plain')
                ])
            ]);
        });
    }
    isSimpleExpression(source, language) {
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
    evaluateSimpleExpression(source, language) {
        return __awaiter(this, void 0, void 0, function* () {
            const trimmed = source.trim();
            if (language === 'python') {
                if (/^[\d\s+\-*/().]+$/.test(trimmed)) {
                    try {
                        // Safe math evaluation
                        const result = Function(`"use strict"; return (${trimmed})`)();
                        return `${result}`;
                    }
                    catch (_a) {
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
                    }
                    catch (_b) {
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
        });
    }
    parseLLMCommand(source, languageId) {
        const lines = source.split('\n');
        let model = 'gpt-4o-mini';
        let query = '';
        let includeContext = true;
        let temperature = 0.7;
        const firstLine = lines[0];
        if (firstLine.startsWith('%%llm') || firstLine.startsWith('%llm')) {
            // Parse magic command
            const modelMatch = firstLine.match(/--model[=\s]+([^\s]+)/);
            if (modelMatch)
                model = modelMatch[1];
            const tempMatch = firstLine.match(/--temperature[=\s]+([\d.]+)/);
            if (tempMatch)
                temperature = parseFloat(tempMatch[1]);
            if (firstLine.includes('--no-context'))
                includeContext = false;
            query = lines.slice(1).join('\n').trim();
        }
        else {
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
    extractNotebookContext(notebook, currentCell, includeContext) {
        var _a;
        return __awaiter(this, void 0, void 0, function* () {
            const context = {
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
                        hasOutput: (((_a = cell.outputs) === null || _a === void 0 ? void 0 : _a.length) || 0) > 0
                    });
                    // Basic parsing for context
                    this.parseCodeForContext(cellText, context.language, context);
                }
            }
            return context;
        });
    }
    parseCodeForContext(code, language, context) {
        // Simple context extraction
        if (language === 'python') {
            const imports = code.match(/^(?:import|from)\s+[\w.]+(?:\s+import\s+[\w,\s*]+)?/gm);
            if (imports)
                context.imports.push(...imports);
            const functions = code.match(/^def\s+(\w+)\s*\(/gm);
            if (functions)
                context.functions.push(...functions);
            const variables = code.match(/^(\w+)\s*=/gm);
            if (variables)
                context.variables.push(...variables);
        }
        // Add other languages as needed
    }
    queryLLM(request, context) {
        return __awaiter(this, void 0, void 0, function* () {
            // Mock LLM execution
            yield new Promise(resolve => setTimeout(resolve, 1500));
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
        });
    }
    dispose() {
        this._controller.dispose();
    }
}
exports.UniversalLLMController = UniversalLLMController;
