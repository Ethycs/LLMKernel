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
exports.TransparentProxyController = void 0;
const vscode = __importStar(require("vscode"));
class TransparentProxyController {
    constructor() {
        this._controllers = new Map();
        this._executionOrder = 0;
        this.setupKernelDetection();
    }
    setupKernelDetection() {
        // Monitor available kernels and create proxy controllers
        vscode.workspace.onDidChangeConfiguration((e) => {
            if (e.affectsConfiguration('jupyter')) {
                this.refreshKernelControllers();
            }
        });
        // Initial setup
        this.refreshKernelControllers();
    }
    refreshKernelControllers() {
        return __awaiter(this, void 0, void 0, function* () {
            // Detect available Jupyter kernels
            const availableKernels = yield this.detectAvailableKernels();
            for (const kernel of availableKernels) {
                if (!this._controllers.has(kernel.name)) {
                    this.createProxyController(kernel);
                }
            }
        });
    }
    detectAvailableKernels() {
        return __awaiter(this, void 0, void 0, function* () {
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
        });
    }
    createProxyController(kernelSpec) {
        const controller = vscode.notebooks.createNotebookController(kernelSpec.name, 'llm-notebook', // Only for .llmnb files — jupyter-notebook uses the real llm_kernel
        kernelSpec.displayName);
        controller.supportedLanguages = [kernelSpec.language, 'llm-query'];
        controller.supportsExecutionOrder = true;
        controller.description = `${kernelSpec.displayName} with AI assistance`;
        controller.executeHandler = this.createExecuteHandler(kernelSpec);
        // Add LLM-specific capabilities
        controller.detail = 'Enhanced with LLM capabilities - use %%llm for AI queries';
        this._controllers.set(kernelSpec.name, controller);
    }
    createExecuteHandler(kernelSpec) {
        return (cells, notebook, controller) => __awaiter(this, void 0, void 0, function* () {
            for (const cell of cells) {
                yield this.executeCell(cell, kernelSpec, controller);
            }
        });
    }
    executeCell(cell, kernelSpec, controller) {
        return __awaiter(this, void 0, void 0, function* () {
            const execution = controller.createNotebookCellExecution(cell);
            execution.executionOrder = ++this._executionOrder;
            execution.start(Date.now());
            try {
                const source = cell.document.getText().trim();
                if (this.isLLMQuery(source)) {
                    yield this.executeLLMQuery(cell, execution, source, kernelSpec);
                }
                else {
                    yield this.executeNativeCode(cell, execution, source, kernelSpec);
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
    executeLLMQuery(cell, execution, source, kernelSpec) {
        return __awaiter(this, void 0, void 0, function* () {
            // Parse LLM command
            const llmRequest = this.parseLLMCommand(source, kernelSpec.language);
            // Show progress
            execution.replaceOutput([
                new vscode.NotebookCellOutput([
                    vscode.NotebookCellOutputItem.text(`🤖 Querying ${llmRequest.model} for ${kernelSpec.language}...`, 'text/plain')
                ])
            ]);
            try {
                // Get context from notebook
                const context = yield this.extractNotebookContext(cell.notebook, kernelSpec.language);
                // Execute LLM query with context
                const response = yield this.queryLLMWithContext(llmRequest, context);
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
            }
            catch (error) {
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
        });
    }
    executeNativeCode(cell, execution, source, kernelSpec) {
        return __awaiter(this, void 0, void 0, function* () {
            // Forward to the actual kernel
            // This is where we'd communicate with the real Jupyter kernel
            execution.replaceOutput([
                new vscode.NotebookCellOutput([
                    vscode.NotebookCellOutputItem.text(`[Proxied to ${kernelSpec.originalKernel}]\n${source}`, 'text/plain')
                ])
            ]);
            // In a real implementation, this would:
            // 1. Connect to the actual Jupyter kernel using kernelSpec.originalKernel
            // 2. Send the code for execution
            // 3. Stream back results in real-time
            // 4. Handle errors and interruptions
            // 5. Maintain kernel state and variables
            // Mock successful execution
            yield new Promise(resolve => setTimeout(resolve, 500));
        });
    }
    parseLLMCommand(source, language) {
        const lines = source.split('\n');
        let model = 'gpt-4o-mini';
        let query = '';
        let includeContext = true;
        const firstLine = lines[0];
        if (firstLine.startsWith('%%llm') || firstLine.startsWith('%llm')) {
            // Parse magic command parameters
            const modelMatch = firstLine.match(/--model[=\s]+([^\s]+)/);
            if (modelMatch)
                model = modelMatch[1];
            const noContextMatch = firstLine.includes('--no-context');
            if (noContextMatch)
                includeContext = false;
            query = lines.slice(1).join('\n').trim();
        }
        else {
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
    extractNotebookContext(notebook, language) {
        var _a;
        return __awaiter(this, void 0, void 0, function* () {
            const context = {
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
                        hasOutput: (((_a = cell.outputs) === null || _a === void 0 ? void 0 : _a.length) || 0) > 0
                    });
                    // Language-specific parsing
                    yield this.parseCodeForContext(cellText, language, context);
                }
            }
            return context;
        });
    }
    parseCodeForContext(code, language, context) {
        return __awaiter(this, void 0, void 0, function* () {
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
        });
    }
    parsePythonContext(code, context) {
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
    parseRContext(code, context) {
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
    parseJuliaContext(code, context) {
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
    parseJavaScriptContext(code, context) {
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
    queryLLMWithContext(request, context) {
        return __awaiter(this, void 0, void 0, function* () {
            // Simulate API call
            yield new Promise(resolve => setTimeout(resolve, 1000 + Math.random() * 2000));
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
        });
    }
    formatContextForLLM(context, request) {
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
    dispose() {
        for (const controller of this._controllers.values()) {
            controller.dispose();
        }
        this._controllers.clear();
    }
}
exports.TransparentProxyController = TransparentProxyController;
