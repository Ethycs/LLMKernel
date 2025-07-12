"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.DEFAULT_SETTINGS = exports.CONFIG_KEYS = exports.COMMAND_IDS = void 0;
exports.COMMAND_IDS = {
    START_KERNEL: 'llmKernel.start',
    STOP_KERNEL: 'llmKernel.stop',
    SAVE_CONTEXT: 'llmContext.save',
    LOAD_CONTEXT: 'llmContext.load',
    RESET_CONTEXT: 'llmContext.reset',
    EXECUTE_CELL: 'notebook.executeCell',
    MANAGE_NOTEBOOK: 'notebook.manage',
};
exports.CONFIG_KEYS = {
    AUTO_LOAD_CONTEXT: 'llmKernel.autoLoadContext',
    DEBUG_MODE: 'llmKernel.debugMode',
};
exports.DEFAULT_SETTINGS = {
    AUTO_LOAD_CONTEXT: true,
    DEBUG_MODE: false,
};
