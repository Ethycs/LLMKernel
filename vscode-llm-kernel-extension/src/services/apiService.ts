import axios from 'axios';

export class ApiService {
    private baseUrl: string;

    constructor(baseUrl: string = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
    }

    public async sendRequest(endpoint: string, data: any): Promise<any> {
        try {
            const response = await axios.post(`${this.baseUrl}/${endpoint}`, data);
            return response.data;
        } catch (error) {
            console.error('API request failed:', error);
            throw error;
        }
    }

    public async getRequest(endpoint: string): Promise<any> {
        try {
            const response = await axios.get(`${this.baseUrl}/${endpoint}`);
            return response.data;
        } catch (error) {
            console.error('API request failed:', error);
            throw error;
        }
    }

    // Context management methods
    public async getCurrentContext(): Promise<any> {
        return this.getRequest('context/current');
    }

    public async saveContext(name: string, data: any): Promise<void> {
        await this.sendRequest('context/save', { name, data });
    }

    public async loadContext(name: string): Promise<any> {
        return this.getRequest(`context/load/${name}`);
    }

    public async setContext(data: any): Promise<void> {
        await this.sendRequest('context/set', data);
    }

    public async resetContext(keepHidden: boolean = false): Promise<void> {
        await this.sendRequest('context/reset', { keepHidden });
    }

    public async getContextStatus(): Promise<string> {
        const response = await this.getRequest('context/status');
        return response.status;
    }

    // Kernel methods  
    public async sendKernelRequest(data: any): Promise<any> {
        return this.sendRequest('kernel/request', data);
    }
}