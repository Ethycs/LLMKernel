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
exports.registerKernelCommands = exports.KernelCommands = void 0;
const vscode = __importStar(require("vscode"));
const kernelService_1 = require("../services/kernelService");
const apiService_1 = require("../services/apiService");
class KernelCommands {
    constructor(kernelService, apiService) {
        this.kernelService = kernelService;
        this.apiService = apiService;
    }
    registerCommands(context) {
        context.subscriptions.push(vscode.commands.registerCommand('llmKernel.start', this.startKernel.bind(this)), vscode.commands.registerCommand('llmKernel.stop', this.stopKernel.bind(this)), vscode.commands.registerCommand('llmKernel.interact', this.interactWithKernel.bind(this)));
    }
    startKernel() {
        return __awaiter(this, void 0, void 0, function* () {
            try {
                yield this.kernelService.startKernel();
                vscode.window.showInformationMessage('LLM Kernel started successfully.');
            }
            catch (error) {
                vscode.window.showErrorMessage(`Failed to start LLM Kernel: ${error instanceof Error ? error.message : String(error)}`);
            }
        });
    }
    stopKernel() {
        return __awaiter(this, void 0, void 0, function* () {
            try {
                yield this.kernelService.stopKernel();
                vscode.window.showInformationMessage('LLM Kernel stopped successfully.');
            }
            catch (error) {
                vscode.window.showErrorMessage(`Failed to stop LLM Kernel: ${error instanceof Error ? error.message : String(error)}`);
            }
        });
    }
    interactWithKernel() {
        return __awaiter(this, void 0, void 0, function* () {
            const input = yield vscode.window.showInputBox({ prompt: 'Enter your query for the LLM Kernel' });
            if (input) {
                try {
                    const response = yield this.apiService.sendKernelRequest(input);
                    vscode.window.showInformationMessage(`Response from LLM Kernel: ${response}`);
                }
                catch (error) {
                    vscode.window.showErrorMessage(`Error interacting with LLM Kernel: ${error instanceof Error ? error.message : String(error)}`);
                }
            }
        });
    }
}
exports.KernelCommands = KernelCommands;
function registerKernelCommands(kernelProvider) {
    const kernelService = new kernelService_1.KernelService();
    const apiService = new apiService_1.ApiService();
    const kernelCommands = new KernelCommands(kernelService, apiService);
    const context = {
        subscriptions: []
    };
    kernelCommands.registerCommands(context);
    return vscode.Disposable.from(...context.subscriptions);
}
exports.registerKernelCommands = registerKernelCommands;
