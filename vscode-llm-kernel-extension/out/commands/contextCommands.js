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
exports.registerContextCommands = void 0;
const vscode = __importStar(require("vscode"));
const vscode_1 = require("vscode");
const contextProvider_1 = require("../providers/contextProvider");
const apiService_1 = require("../services/apiService");
function registerContextCommands(context) {
    const apiService = new apiService_1.ApiService();
    const contextProvider = new contextProvider_1.ContextProvider(apiService);
    const subscriptions = [];
    subscriptions.push(vscode_1.commands.registerCommand('llm.context.save', () => __awaiter(this, void 0, void 0, function* () {
        const fileName = yield vscode_1.window.showInputBox({ prompt: 'Enter context file name' });
        if (fileName) {
            yield contextProvider.saveContext(fileName);
            vscode_1.window.showInformationMessage(`Context saved as ${fileName}`);
        }
    })), vscode_1.commands.registerCommand('llm.context.load', () => __awaiter(this, void 0, void 0, function* () {
        const fileName = yield vscode_1.window.showInputBox({ prompt: 'Enter context file name to load' });
        if (fileName) {
            yield contextProvider.loadContext(fileName);
            vscode_1.window.showInformationMessage(`Context loaded from ${fileName}`);
        }
    })), vscode_1.commands.registerCommand('llm.context.reset', () => __awaiter(this, void 0, void 0, function* () {
        yield contextProvider.resetContext();
        vscode_1.window.showInformationMessage('Context has been reset');
    })), vscode_1.commands.registerCommand('llm.context.status', () => __awaiter(this, void 0, void 0, function* () {
        yield contextProvider.checkContextStatus();
    })));
    return vscode.Disposable.from(...subscriptions);
}
exports.registerContextCommands = registerContextCommands;
