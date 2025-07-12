export interface KernelStatus {
    isRunning: boolean;
    errorMessage?: string;
}

export interface Context {
    id: string;
    name: string;
    data: any;
}

export interface NotebookCell {
    id: string;
    content: string;
    language: string;
    executionCount?: number;
}

export interface Command {
    id: string;
    title: string;
    category: string;
}

export interface Model {
    id: string;
    name: string;
    description: string;
}