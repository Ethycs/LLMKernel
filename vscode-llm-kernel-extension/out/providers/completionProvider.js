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
Object.defineProperty(exports, "__esModule", { value: true });
exports.CompletionProvider = void 0;
const vscode = __importStar(require("vscode"));
class CompletionProvider {
    constructor(kernelService) {
        this.kernelService = kernelService;
    }
    provideCompletionItems(document, position, token, context) {
        const linePrefix = document.lineAt(position).text.substr(0, position.character);
        // Only provide completions if the line starts with a specific prefix
        if (!linePrefix.startsWith('%llm')) {
            return undefined;
        }
        // Call the kernel service to get completions
        return this.kernelService.getCompletions(linePrefix).then(completions => {
            return completions.map(completion => {
                const item = new vscode.CompletionItem(completion, vscode.CompletionItemKind.Text);
                item.detail = 'LLM kernel completion';
                item.documentation = `Auto-completion for ${completion}`;
                return item;
            });
        });
    }
}
exports.CompletionProvider = CompletionProvider;
