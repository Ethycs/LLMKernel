import * as vscode from 'vscode';
import { KernelService } from '../services/kernelService';
import { ApiService } from '../services/apiService';
import { KernelProvider } from '../providers/kernelProvider';

export class KernelCommands {
    private kernelService: KernelService;
    private apiService: ApiService;

    constructor(kernelService: KernelService, apiService: ApiService) {
        this.kernelService = kernelService;
        this.apiService = apiService;
    }

    public registerCommands(context: vscode.ExtensionContext) {
        context.subscriptions.push(
            vscode.commands.registerCommand('llmKernel.start', this.startKernel.bind(this)),
            vscode.commands.registerCommand('llmKernel.stop', this.stopKernel.bind(this)),
            vscode.commands.registerCommand('llmKernel.interact', this.interactWithKernel.bind(this))
        );
    }

    private async startKernel() {
        try {
            await this.kernelService.startKernel();
            vscode.window.showInformationMessage('LLM Kernel started successfully.');
        } catch (error) {
            vscode.window.showErrorMessage(`Failed to start LLM Kernel: ${error instanceof Error ? error.message : String(error)}`);
        }
    }

    private async stopKernel() {
        try {
            await this.kernelService.stopKernel();
            vscode.window.showInformationMessage('LLM Kernel stopped successfully.');
        } catch (error) {
            vscode.window.showErrorMessage(`Failed to stop LLM Kernel: ${error instanceof Error ? error.message : String(error)}`);
        }
    }

    private async interactWithKernel() {
        const input = await vscode.window.showInputBox({ prompt: 'Enter your query for the LLM Kernel' });
        if (input) {
            try {
                const response = await this.apiService.sendKernelRequest(input);
                vscode.window.showInformationMessage(`Response from LLM Kernel: ${response}`);
            } catch (error) {
                vscode.window.showErrorMessage(`Error interacting with LLM Kernel: ${error instanceof Error ? error.message : String(error)}`);
            }
        }
    }
}

export function registerKernelCommands(kernelProvider: KernelProvider): vscode.Disposable {
    const kernelService = new KernelService();
    const apiService = new ApiService();
    const kernelCommands = new KernelCommands(kernelService, apiService);
    
    const context = {
        subscriptions: [] as vscode.Disposable[]
    };
    
    kernelCommands.registerCommands(context as vscode.ExtensionContext);
    
    return vscode.Disposable.from(...context.subscriptions);
}