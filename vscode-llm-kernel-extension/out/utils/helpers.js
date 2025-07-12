"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.getCurrentTimestamp = exports.delay = exports.isValidJson = exports.formatMessage = void 0;
function formatMessage(message) {
    return message.trim().replace(/\s+/g, ' ');
}
exports.formatMessage = formatMessage;
function isValidJson(jsonString) {
    try {
        JSON.parse(jsonString);
        return true;
    }
    catch (_a) {
        return false;
    }
}
exports.isValidJson = isValidJson;
function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}
exports.delay = delay;
function getCurrentTimestamp() {
    return new Date().toISOString();
}
exports.getCurrentTimestamp = getCurrentTimestamp;
