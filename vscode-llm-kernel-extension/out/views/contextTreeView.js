"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || function (mod) {
    if (mod && mod.__esModule) return mod;
    var result = {};
    if (mod != null) for (var k in mod) if (k !== "default" && Object.prototype.hasOwnProperty.call(mod, k)) __createBinding(result, mod, k);
    __setModuleDefault(result, mod);
    return result;
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.ContextTreeView = void 0;
const vscode = __importStar(require("vscode"));
class ContextTreeView {
    constructor() {
        this.treeView = vscode.window.createTreeView('contextTreeView', {
            treeDataProvider: new ContextTreeDataProvider(),
            showCollapseAll: true,
        });
    }
    refresh() {
        // TreeView doesn't have direct refresh method or provider property
        // Would need to store the data provider reference separately
        // For now, this is a no-op
    }
}
exports.ContextTreeView = ContextTreeView;
class ContextTreeDataProvider {
    constructor() {
        this.contextItems = [];
        this.loadContextItems();
    }
    loadContextItems() {
        // Load context items from the LLM kernel or other sources
        // This is a placeholder for actual implementation
        this.contextItems = [
            new ContextTreeItem('Context Item 1', vscode.TreeItemCollapsibleState.None),
            new ContextTreeItem('Context Item 2', vscode.TreeItemCollapsibleState.None),
        ];
    }
    getTreeItem(element) {
        return element;
    }
    getChildren(element) {
        if (element) {
            return Promise.resolve([]); // No children for now
        }
        return Promise.resolve(this.contextItems);
    }
}
class ContextTreeItem extends vscode.TreeItem {
    constructor(label, collapsibleState) {
        super(label, collapsibleState);
    }
}
