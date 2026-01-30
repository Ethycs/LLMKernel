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
exports.LLMNotebookSerializer = void 0;
const vscode = __importStar(require("vscode"));
class LLMNotebookSerializer {
    deserializeNotebook(content, _token) {
        return __awaiter(this, void 0, void 0, function* () {
            const contents = new TextDecoder().decode(content);
            let raw;
            try {
                raw = JSON.parse(contents);
            }
            catch (_a) {
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
            notebookData.metadata = Object.assign(Object.assign({}, raw.metadata), { custom: {
                    llm_kernel: raw.metadata.llm_kernel || {
                        version: '1.0.0',
                        default_model: 'gpt-4o-mini',
                        session_id: this.generateSessionId()
                    }
                } });
            return notebookData;
        });
    }
    serializeNotebook(data, _token) {
        var _a, _b;
        return __awaiter(this, void 0, void 0, function* () {
            const contents = {
                cells: data.cells.map(cell => this.notebookCellDataToRaw(cell)),
                metadata: Object.assign(Object.assign({}, data.metadata), { llm_kernel: ((_b = (_a = data.metadata) === null || _a === void 0 ? void 0 : _a.custom) === null || _b === void 0 ? void 0 : _b.llm_kernel) || {
                        version: '1.0.0',
                        default_model: 'gpt-4o-mini',
                        session_id: this.generateSessionId()
                    } }),
                nbformat: 4,
                nbformat_minor: 2
            };
            return new TextEncoder().encode(JSON.stringify(contents, null, 2));
        });
    }
    rawToNotebookCellData(cell) {
        const cellData = new vscode.NotebookCellData(this.getCellKind(cell.cell_type), Array.isArray(cell.source) ? cell.source.join('') : cell.source, this.getLanguageId(cell));
        // Set cell metadata
        cellData.metadata = Object.assign(Object.assign({}, cell.metadata), { cellType: cell.cell_type, llmCell: cell.cell_type === 'llm' });
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
    notebookCellDataToRaw(cell) {
        var _a, _b, _c;
        const cellType = ((_a = cell.metadata) === null || _a === void 0 ? void 0 : _a.cellType) || this.inferCellType(cell);
        return {
            cell_type: cellType,
            language_id: cellType === 'llm' ? 'llm-query' : cell.languageId,
            source: cell.value.split('\n').map((line, i, arr) => i === arr.length - 1 ? line : line + '\n'),
            metadata: cell.metadata || {},
            execution_count: ((_b = cell.executionSummary) === null || _b === void 0 ? void 0 : _b.executionOrder) || null,
            outputs: ((_c = cell.outputs) === null || _c === void 0 ? void 0 : _c.map(output => this.convertOutputToRaw(output))) || []
        };
    }
    getCellKind(cellType) {
        switch (cellType) {
            case 'markdown':
                return vscode.NotebookCellKind.Markup;
            case 'llm':
            case 'code':
            default:
                return vscode.NotebookCellKind.Code;
        }
    }
    getLanguageId(cell) {
        if (cell.cell_type === 'llm') {
            return 'llm-query';
        }
        if (cell.cell_type === 'markdown') {
            return 'markdown';
        }
        return cell.language_id || 'python';
    }
    inferCellType(cell) {
        var _a;
        if (((_a = cell.metadata) === null || _a === void 0 ? void 0 : _a.llmCell) || cell.languageId === 'llm-query') {
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
    convertOutput(output) {
        const items = [];
        if (output.output_type === 'execute_result' || output.output_type === 'display_data') {
            // Handle various MIME types
            for (const [mimeType, data] of Object.entries(output.data || {})) {
                if (mimeType === 'application/llm-response') {
                    items.push(vscode.NotebookCellOutputItem.json(data, mimeType));
                }
                else if (mimeType === 'text/plain') {
                    const text = Array.isArray(data) ? data.join('') : data;
                    items.push(vscode.NotebookCellOutputItem.text(text));
                }
                else if (mimeType === 'text/html') {
                    const html = Array.isArray(data) ? data.join('') : data;
                    items.push(vscode.NotebookCellOutputItem.text(html, 'text/html'));
                }
                else {
                    items.push(vscode.NotebookCellOutputItem.json(data, mimeType));
                }
            }
        }
        else if (output.output_type === 'stream') {
            const text = Array.isArray(output.text) ? output.text.join('') : output.text;
            items.push(vscode.NotebookCellOutputItem.stdout(text));
        }
        else if (output.output_type === 'error') {
            items.push(vscode.NotebookCellOutputItem.error({
                name: output.ename || 'Error',
                message: output.evalue || 'Unknown error',
                stack: Array.isArray(output.traceback) ? output.traceback.join('\n') : output.traceback
            }));
        }
        return new vscode.NotebookCellOutput(items, output.metadata);
    }
    convertOutputToRaw(output) {
        const outputData = {
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
            }
            else if (item.mime === 'application/vnd.code.notebook.stdout') {
                return {
                    output_type: 'stream',
                    name: 'stdout',
                    text: new TextDecoder().decode(item.data)
                };
            }
            else if (item.mime === 'application/vnd.code.notebook.stderr') {
                return {
                    output_type: 'stream',
                    name: 'stderr',
                    text: new TextDecoder().decode(item.data)
                };
            }
            else {
                try {
                    const data = item.mime.startsWith('application/')
                        ? JSON.parse(new TextDecoder().decode(item.data))
                        : new TextDecoder().decode(item.data);
                    outputData.data[item.mime] = data;
                }
                catch (_a) {
                    outputData.data[item.mime] = new TextDecoder().decode(item.data);
                }
            }
        }
        return outputData;
    }
    generateSessionId() {
        return `llm_session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }
    // Helper method to create new LLM notebook
    static createNewLLMNotebook() {
        const welcomeCell = new vscode.NotebookCellData(vscode.NotebookCellKind.Code, 'Welcome to LLM Enhanced Notebook!\n\nThis notebook supports:\n- Regular code cells (Python, R, Julia, etc.)\n- LLM query cells for AI assistance\n- Mixed execution with context sharing\n\nPress Ctrl+Shift+L to add an LLM query cell.', 'llm-query');
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
exports.LLMNotebookSerializer = LLMNotebookSerializer;
