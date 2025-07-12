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
exports.NotebookService = void 0;
const vscode = __importStar(require("vscode"));
class NotebookService {
    constructor() { }
    executeCell(cell, document) {
        var _a, _b, _c, _d;
        return __awaiter(this, void 0, void 0, function* () {
            // Mock implementation for compilation - would need actual controller
            this.currentExecution = {
                start: (time) => { },
                end: (success, time) => { },
                replaceOutput: (outputs) => { }
            };
            (_a = this.currentExecution) === null || _a === void 0 ? void 0 : _a.start(Date.now());
            try {
                // Simulate cell execution logic
                const result = yield this.runCellCode(cell.document.getText());
                (_b = this.currentExecution) === null || _b === void 0 ? void 0 : _b.replaceOutput([
                    new vscode.NotebookCellOutput([
                        vscode.NotebookCellOutputItem.text(result, 'text/plain')
                    ])
                ]);
            }
            catch (error) {
                (_c = this.currentExecution) === null || _c === void 0 ? void 0 : _c.replaceOutput([
                    new vscode.NotebookCellOutput([
                        vscode.NotebookCellOutputItem.error({
                            name: error instanceof Error ? error.constructor.name : 'Error',
                            message: error instanceof Error ? error.message : String(error)
                        })
                    ])
                ]);
            }
            finally {
                (_d = this.currentExecution) === null || _d === void 0 ? void 0 : _d.end(true, Date.now());
            }
        });
    }
    runCellCode(code) {
        return __awaiter(this, void 0, void 0, function* () {
            // Placeholder for actual execution logic
            return new Promise((resolve) => {
                setTimeout(() => {
                    resolve(`Executed: ${code}`);
                }, 1000);
            });
        });
    }
    resetNotebook(document) {
        // Logic to reset the notebook state
        // Note: Direct cell output clearing not available in VS Code API
        // This would need to be implemented through the notebook controller
        vscode.window.showInformationMessage('Notebook reset functionality needs controller implementation');
    }
    saveNotebookState() {
        return __awaiter(this, void 0, void 0, function* () {
            // Mock implementation for saving notebook state
            return Promise.resolve();
        });
    }
    loadNotebookState() {
        return __awaiter(this, void 0, void 0, function* () {
            // Mock implementation for loading notebook state
            return Promise.resolve();
        });
    }
}
exports.NotebookService = NotebookService;
