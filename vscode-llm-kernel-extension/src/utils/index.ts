import { commands } from 'vscode';

export * from './constants';
export * from './helpers';

// Register utility commands
export function registerUtilityCommands() {
    commands.registerCommand('extension.someUtilityCommand', () => {
        // Implementation of the utility command
    });
}