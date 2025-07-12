export const COMMAND_IDS = {
    START_KERNEL: 'llmKernel.start',
    STOP_KERNEL: 'llmKernel.stop',
    SAVE_CONTEXT: 'llmContext.save',
    LOAD_CONTEXT: 'llmContext.load',
    RESET_CONTEXT: 'llmContext.reset',
    EXECUTE_CELL: 'notebook.executeCell',
    MANAGE_NOTEBOOK: 'notebook.manage',
};

export const CONFIG_KEYS = {
    AUTO_LOAD_CONTEXT: 'llmKernel.autoLoadContext',
    DEBUG_MODE: 'llmKernel.debugMode',
};

export const DEFAULT_SETTINGS = {
    AUTO_LOAD_CONTEXT: true,
    DEBUG_MODE: false,
};