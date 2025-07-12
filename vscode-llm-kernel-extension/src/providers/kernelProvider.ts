import * as vscode from 'vscode';
import { KernelService } from '../services/kernelService';

export class KernelProvider {
    private kernelService: KernelService;

    constructor() {
        this.kernelService = new KernelService();
    }

    public async startKernel(): Promise<void> {
        try {
            await this.kernelService.startKernel();
            vscode.window.showInformationMessage('LLM Kernel started successfully.');
        } catch (error) {
            vscode.window.showErrorMessage(`Failed to start LLM Kernel: ${error instanceof Error ? error.message : String(error)}`);
        }
    }

    public async stopKernel(): Promise<void> {
        try {
            await this.kernelService.stopKernel();
            vscode.window.showInformationMessage('LLM Kernel stopped successfully.');
        } catch (error) {
            vscode.window.showErrorMessage(`Failed to stop LLM Kernel: ${error instanceof Error ? error.message : String(error)}`);
        }
    }

    public async sendRequest(request: any): Promise<any> {
        try {
            const response = await this.kernelService.executeCode(JSON.stringify(request));
            return response;
        } catch (error) {
            vscode.window.showErrorMessage(`Error sending request to LLM Kernel: ${error instanceof Error ? error.message : String(error)}`);
            throw error;
        }
    }

    public async getKernelStatus(): Promise<string> {
        try {
            const status = await this.kernelService.getKernelStatus();
            return status;
        } catch (error) {
            vscode.window.showErrorMessage(`Error retrieving kernel status: ${error instanceof Error ? error.message : String(error)}`);
            throw error;
        }
    }
}