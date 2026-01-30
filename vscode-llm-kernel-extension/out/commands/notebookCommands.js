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
exports.registerNotebookCommands = void 0;
const vscode = __importStar(require("vscode"));
const notebookService_1 = require("../services/notebookService");
function registerNotebookCommands(kernelProvider) {
    const notebookService = new notebookService_1.NotebookService();
    const subscriptions = [];
    subscriptions.push(vscode.commands.registerCommand('llm.notebook.executeCell', (cell) => __awaiter(this, void 0, void 0, function* () {
        var _a;
        try {
            const document = (_a = vscode.window.activeNotebookEditor) === null || _a === void 0 ? void 0 : _a.notebook;
            if (!document)
                throw new Error('No active notebook');
            yield notebookService.executeCell(cell, document);
            vscode.window.showInformationMessage('Cell executed successfully');
        }
        catch (error) {
            vscode.window.showErrorMessage(`Error executing cell: ${error instanceof Error ? error.message : String(error)}`);
        }
    })), vscode.commands.registerCommand('llm.notebook.saveState', () => __awaiter(this, void 0, void 0, function* () {
        try {
            yield notebookService.saveNotebookState();
            vscode.window.showInformationMessage('Notebook state saved successfully.');
        }
        catch (error) {
            vscode.window.showErrorMessage(`Error saving notebook state: ${error instanceof Error ? error.message : String(error)}`);
        }
    })), vscode.commands.registerCommand('llm.notebook.loadState', () => __awaiter(this, void 0, void 0, function* () {
        try {
            yield notebookService.loadNotebookState();
            vscode.window.showInformationMessage('Notebook state loaded successfully.');
        }
        catch (error) {
            vscode.window.showErrorMessage(`Error loading notebook state: ${error instanceof Error ? error.message : String(error)}`);
        }
    })));
    return vscode.Disposable.from(...subscriptions);
}
exports.registerNotebookCommands = registerNotebookCommands;
