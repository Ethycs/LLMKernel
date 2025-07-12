export function formatMessage(message: string): string {
    return message.trim().replace(/\s+/g, ' ');
}

export function isValidJson(jsonString: string): boolean {
    try {
        JSON.parse(jsonString);
        return true;
    } catch {
        return false;
    }
}

export function delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
}

export function getCurrentTimestamp(): string {
    return new Date().toISOString();
}