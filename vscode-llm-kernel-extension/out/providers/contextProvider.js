"use strict";
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
exports.ContextProvider = void 0;
const vscode_1 = require("vscode");
class ContextProvider {
    constructor(apiService) {
        this.apiService = apiService;
    }
    saveContext(contextName) {
        return __awaiter(this, void 0, void 0, function* () {
            try {
                const contextData = yield this.apiService.getCurrentContext();
                yield this.apiService.saveContext(contextName, contextData);
                vscode_1.window.showInformationMessage(`Context "${contextName}" saved successfully.`);
            }
            catch (error) {
                vscode_1.window.showErrorMessage(`Failed to save context: ${error instanceof Error ? error.message : String(error)}`);
            }
        });
    }
    loadContext(contextName) {
        return __awaiter(this, void 0, void 0, function* () {
            try {
                const contextData = yield this.apiService.loadContext(contextName);
                yield this.apiService.setContext(contextData);
                vscode_1.window.showInformationMessage(`Context "${contextName}" loaded successfully.`);
            }
            catch (error) {
                vscode_1.window.showErrorMessage(`Failed to load context: ${error instanceof Error ? error.message : String(error)}`);
            }
        });
    }
    resetContext(keepHidden = false) {
        return __awaiter(this, void 0, void 0, function* () {
            try {
                yield this.apiService.resetContext(keepHidden);
                vscode_1.window.showInformationMessage('Context reset successfully.');
            }
            catch (error) {
                vscode_1.window.showErrorMessage(`Failed to reset context: ${error instanceof Error ? error.message : String(error)}`);
            }
        });
    }
    checkContextStatus() {
        return __awaiter(this, void 0, void 0, function* () {
            try {
                const status = yield this.apiService.getContextStatus();
                vscode_1.window.showInformationMessage(`Context status: ${status}`);
            }
            catch (error) {
                vscode_1.window.showErrorMessage(`Failed to check context status: ${error instanceof Error ? error.message : String(error)}`);
            }
        });
    }
}
exports.ContextProvider = ContextProvider;
