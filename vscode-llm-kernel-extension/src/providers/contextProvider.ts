import { workspace, window } from 'vscode';
import { ApiService } from '../services/apiService';

export class ContextProvider {
    private apiService: ApiService;

    constructor(apiService: ApiService) {
        this.apiService = apiService;
    }

    public async saveContext(contextName: string): Promise<void> {
        try {
            const contextData = await this.apiService.getCurrentContext();
            await this.apiService.saveContext(contextName, contextData);
            window.showInformationMessage(`Context "${contextName}" saved successfully.`);
        } catch (error) {
            window.showErrorMessage(`Failed to save context: ${error instanceof Error ? error.message : String(error)}`);
        }
    }

    public async loadContext(contextName: string): Promise<void> {
        try {
            const contextData = await this.apiService.loadContext(contextName);
            await this.apiService.setContext(contextData);
            window.showInformationMessage(`Context "${contextName}" loaded successfully.`);
        } catch (error) {
            window.showErrorMessage(`Failed to load context: ${error instanceof Error ? error.message : String(error)}`);
        }
    }

    public async resetContext(keepHidden: boolean = false): Promise<void> {
        try {
            await this.apiService.resetContext(keepHidden);
            window.showInformationMessage('Context reset successfully.');
        } catch (error) {
            window.showErrorMessage(`Failed to reset context: ${error instanceof Error ? error.message : String(error)}`);
        }
    }

    public async checkContextStatus(): Promise<void> {
        try {
            const status = await this.apiService.getContextStatus();
            window.showInformationMessage(`Context status: ${status}`);
        } catch (error) {
            window.showErrorMessage(`Failed to check context status: ${error instanceof Error ? error.message : String(error)}`);
        }
    }
}