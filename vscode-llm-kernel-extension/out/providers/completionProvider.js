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
exports.CompletionProvider = void 0;
const vscode = __importStar(require("vscode"));
/**
 * Static registry of all kernel magic commands.
 * Organized by module matching the kernel's magic command structure.
 */
const MAGIC_COMMANDS = [
    // === BaseMagics ===
    {
        trigger: '%%llm',
        label: '%%llm',
        detail: 'Query an LLM',
        documentation: 'Send a query to the active LLM model. Use --model=X to specify a model.',
        insertText: '%%llm --model=${1:gpt-4o}\n${2:Your query here}',
        kind: vscode.CompletionItemKind.Function,
    },
    {
        trigger: '%%llm_gpt4',
        label: '%%llm_gpt4',
        detail: 'Query GPT-4',
        documentation: 'Send a query directly to GPT-4.',
        insertText: '%%llm_gpt4\n${1:Your query here}',
        kind: vscode.CompletionItemKind.Function,
    },
    {
        trigger: '%%llm_claude',
        label: '%%llm_claude',
        detail: 'Query Claude',
        documentation: 'Send a query directly to Claude.',
        insertText: '%%llm_claude\n${1:Your query here}',
        kind: vscode.CompletionItemKind.Function,
    },
    {
        trigger: '%%llm_compare',
        label: '%%llm_compare',
        detail: 'Compare models',
        documentation: 'Send the same query to multiple models and compare responses.',
        insertText: '%%llm_compare ${1:gpt-4o} ${2:claude-3-sonnet}\n${3:Your query here}',
        kind: vscode.CompletionItemKind.Function,
    },
    {
        trigger: '%llm_chat',
        label: '%llm_chat',
        detail: 'Toggle chat mode',
        documentation: 'Toggle chat mode on/off. Use %llm_chat on or %llm_chat off.',
        insertText: '%llm_chat ${1|on,off|}',
        kind: vscode.CompletionItemKind.Function,
    },
    {
        trigger: '%llm_models',
        label: '%llm_models',
        detail: 'List available models',
        documentation: 'Show all available LLM models and their status.',
        insertText: '%llm_models',
        kind: vscode.CompletionItemKind.Property,
    },
    {
        trigger: '%llm_model',
        label: '%llm_model',
        detail: 'Switch active model',
        documentation: 'Switch the active LLM model. Usage: %llm_model gpt-4o',
        insertText: '%llm_model ${1:gpt-4o}',
        kind: vscode.CompletionItemKind.Function,
    },
    {
        trigger: '%llm_status',
        label: '%llm_status',
        detail: 'Show kernel status',
        documentation: 'Display current kernel status including model, tokens, cost, and context.',
        insertText: '%llm_status',
        kind: vscode.CompletionItemKind.Property,
    },
    {
        trigger: '%llm_clear',
        label: '%llm_clear',
        detail: 'Clear conversation',
        documentation: 'Clear the current conversation history and context.',
        insertText: '%llm_clear',
        kind: vscode.CompletionItemKind.Function,
    },
    {
        trigger: '%llm_display',
        label: '%llm_display',
        detail: 'Set display mode',
        documentation: 'Set output display mode (markdown, raw, html).',
        insertText: '%llm_display ${1|markdown,raw,html|}',
        kind: vscode.CompletionItemKind.Function,
    },
    {
        trigger: '%llm_debug',
        label: '%llm_debug',
        detail: 'Toggle debug mode',
        documentation: 'Toggle debug output for LLM queries.',
        insertText: '%llm_debug',
        kind: vscode.CompletionItemKind.Function,
    },
    // === ContextMagics ===
    {
        trigger: '%llm_context',
        label: '%llm_context',
        detail: 'Show context',
        documentation: 'Display the current conversation context.',
        insertText: '%llm_context',
        kind: vscode.CompletionItemKind.Property,
    },
    {
        trigger: '%%hide',
        label: '%%hide',
        detail: 'Hide cell from context',
        documentation: 'Mark this cell as hidden from LLM context.',
        insertText: '%%hide\n${1:code}',
        kind: vscode.CompletionItemKind.Function,
    },
    {
        trigger: '%llm_unhide',
        label: '%llm_unhide',
        detail: 'Unhide cell',
        documentation: 'Remove the hidden flag from a cell.',
        insertText: '%llm_unhide',
        kind: vscode.CompletionItemKind.Function,
    },
    {
        trigger: '%llm_hidden',
        label: '%llm_hidden',
        detail: 'List hidden cells',
        documentation: 'Show all cells currently hidden from context.',
        insertText: '%llm_hidden',
        kind: vscode.CompletionItemKind.Property,
    },
    {
        trigger: '%llm_context_save',
        label: '%llm_context_save',
        detail: 'Save context',
        documentation: 'Save the current context to a file.',
        insertText: '%llm_context_save ${1:filename}',
        kind: vscode.CompletionItemKind.Function,
    },
    {
        trigger: '%llm_context_load',
        label: '%llm_context_load',
        detail: 'Load context',
        documentation: 'Load a previously saved context.',
        insertText: '%llm_context_load ${1:filename}',
        kind: vscode.CompletionItemKind.Function,
    },
    {
        trigger: '%llm_context_reset',
        label: '%llm_context_reset',
        detail: 'Reset context',
        documentation: 'Reset all context and conversation history.',
        insertText: '%llm_context_reset',
        kind: vscode.CompletionItemKind.Function,
    },
    {
        trigger: '%llm_context_persist',
        label: '%llm_context_persist',
        detail: 'Toggle context persistence',
        documentation: 'Toggle whether context persists across kernel restarts.',
        insertText: '%llm_context_persist ${1|on,off|}',
        kind: vscode.CompletionItemKind.Function,
    },
    {
        trigger: '%llm_pin_cell',
        label: '%llm_pin_cell',
        detail: 'Pin cell to context',
        documentation: 'Pin a cell so it is always included in LLM context.',
        insertText: '%llm_pin_cell',
        kind: vscode.CompletionItemKind.Function,
    },
    {
        trigger: '%llm_unpin_cell',
        label: '%llm_unpin_cell',
        detail: 'Unpin cell',
        documentation: 'Remove the pin from a cell.',
        insertText: '%llm_unpin_cell',
        kind: vscode.CompletionItemKind.Function,
    },
    {
        trigger: '%llm_history',
        label: '%llm_history',
        detail: 'Show history',
        documentation: 'Show conversation history.',
        insertText: '%llm_history',
        kind: vscode.CompletionItemKind.Property,
    },
    {
        trigger: '%llm_prune',
        label: '%llm_prune',
        detail: 'Prune context',
        documentation: 'Prune context using a strategy: smart, semantic, recency.',
        insertText: '%llm_prune --strategy=${1|smart,semantic,recency|} ${2:--keep=10}',
        kind: vscode.CompletionItemKind.Function,
    },
    // === ConfigMagics ===
    {
        trigger: '%llm_config',
        label: '%llm_config',
        detail: 'Show/set configuration',
        documentation: 'Display or update kernel configuration.',
        insertText: '%llm_config',
        kind: vscode.CompletionItemKind.Property,
    },
    {
        trigger: '%llm_context_window',
        label: '%llm_context_window',
        detail: 'Show context window usage',
        documentation: 'Display context window size and usage.',
        insertText: '%llm_context_window',
        kind: vscode.CompletionItemKind.Property,
    },
    {
        trigger: '%llm_token_count',
        label: '%llm_token_count',
        detail: 'Show token count',
        documentation: 'Display token usage for the current session.',
        insertText: '%llm_token_count',
        kind: vscode.CompletionItemKind.Property,
    },
    {
        trigger: '%llm_cost',
        label: '%llm_cost',
        detail: 'Show session cost',
        documentation: 'Display the cost of the current session.',
        insertText: '%llm_cost',
        kind: vscode.CompletionItemKind.Property,
    },
    // === MCPMagics ===
    {
        trigger: '%llm_mcp_connect',
        label: '%llm_mcp_connect',
        detail: 'Connect MCP server',
        documentation: 'Connect to an MCP (Model Context Protocol) server.',
        insertText: '%llm_mcp_connect',
        kind: vscode.CompletionItemKind.Function,
    },
    {
        trigger: '%llm_mcp_disconnect',
        label: '%llm_mcp_disconnect',
        detail: 'Disconnect MCP server',
        documentation: 'Disconnect from an MCP server.',
        insertText: '%llm_mcp_disconnect',
        kind: vscode.CompletionItemKind.Function,
    },
    {
        trigger: '%llm_mcp_tools',
        label: '%llm_mcp_tools',
        detail: 'List MCP tools',
        documentation: 'List all available MCP tools.',
        insertText: '%llm_mcp_tools',
        kind: vscode.CompletionItemKind.Property,
    },
    {
        trigger: '%llm_mcp_call',
        label: '%llm_mcp_call',
        detail: 'Call MCP tool',
        documentation: 'Call a specific MCP tool.',
        insertText: '%llm_mcp_call ${1:tool_name}',
        kind: vscode.CompletionItemKind.Function,
    },
    {
        trigger: '%llm_mcp_config',
        label: '%llm_mcp_config',
        detail: 'MCP configuration',
        documentation: 'Show or update MCP server configuration.',
        insertText: '%llm_mcp_config',
        kind: vscode.CompletionItemKind.Property,
    },
    {
        trigger: '%%llm_mcp',
        label: '%%llm_mcp',
        detail: 'MCP query',
        documentation: 'Send a query using MCP tools.',
        insertText: '%%llm_mcp\n${1:Your query here}',
        kind: vscode.CompletionItemKind.Function,
    },
    // === RerankingMagics ===
    {
        trigger: '%llm_rerank',
        label: '%llm_rerank',
        detail: 'Rerank context',
        documentation: 'Rerank context cells by relevance.',
        insertText: '%llm_rerank',
        kind: vscode.CompletionItemKind.Function,
    },
    {
        trigger: '%llm_rerank_clear',
        label: '%llm_rerank_clear',
        detail: 'Clear reranking',
        documentation: 'Clear reranking scores and restore original order.',
        insertText: '%llm_rerank_clear',
        kind: vscode.CompletionItemKind.Function,
    },
    {
        trigger: '%llm_rerank_apply',
        label: '%llm_rerank_apply',
        detail: 'Apply reranking',
        documentation: 'Apply the current reranking to context.',
        insertText: '%llm_rerank_apply',
        kind: vscode.CompletionItemKind.Function,
    },
    {
        trigger: '%%meta',
        label: '%%meta',
        detail: 'Add cell metadata',
        documentation: 'Add metadata annotations to a cell.',
        insertText: '%%meta\n${1:metadata}',
        kind: vscode.CompletionItemKind.Function,
    },
    {
        trigger: '%llm_apply_meta',
        label: '%llm_apply_meta',
        detail: 'Apply metadata',
        documentation: 'Apply metadata annotations from %%meta cells.',
        insertText: '%llm_apply_meta',
        kind: vscode.CompletionItemKind.Function,
    },
    {
        trigger: '%llm_meta_list',
        label: '%llm_meta_list',
        detail: 'List metadata',
        documentation: 'List all cell metadata annotations.',
        insertText: '%llm_meta_list',
        kind: vscode.CompletionItemKind.Property,
    },
    // === MultimodalMagics ===
    {
        trigger: '%llm_paste',
        label: '%llm_paste',
        detail: 'Paste image from clipboard',
        documentation: 'Paste an image from the clipboard into the LLM context.',
        insertText: '%llm_paste',
        kind: vscode.CompletionItemKind.Function,
    },
    {
        trigger: '%llm_image',
        label: '%llm_image',
        detail: 'Add image to context',
        documentation: 'Add an image file to the LLM context.',
        insertText: '%llm_image ${1:path/to/image.png}',
        kind: vscode.CompletionItemKind.Function,
    },
    {
        trigger: '%llm_pdf',
        label: '%llm_pdf',
        detail: 'Add PDF to context',
        documentation: 'Add a PDF file to the LLM context.',
        insertText: '%llm_pdf ${1:path/to/document.pdf}',
        kind: vscode.CompletionItemKind.Function,
    },
    {
        trigger: '%llm_media_clear',
        label: '%llm_media_clear',
        detail: 'Clear media',
        documentation: 'Clear all media (images, PDFs) from context.',
        insertText: '%llm_media_clear',
        kind: vscode.CompletionItemKind.Function,
    },
    {
        trigger: '%llm_media_list',
        label: '%llm_media_list',
        detail: 'List media',
        documentation: 'List all media files in the current context.',
        insertText: '%llm_media_list',
        kind: vscode.CompletionItemKind.Property,
    },
    {
        trigger: '%%llm_vision',
        label: '%%llm_vision',
        detail: 'Vision query',
        documentation: 'Send a query with attached images for vision analysis.',
        insertText: '%%llm_vision\n${1:Describe what you see in this image}',
        kind: vscode.CompletionItemKind.Function,
    },
    {
        trigger: '%llm_cache_info',
        label: '%llm_cache_info',
        detail: 'Show cache info',
        documentation: 'Display media cache information.',
        insertText: '%llm_cache_info',
        kind: vscode.CompletionItemKind.Property,
    },
    {
        trigger: '%llm_cache_list',
        label: '%llm_cache_list',
        detail: 'List cache entries',
        documentation: 'List all entries in the media cache.',
        insertText: '%llm_cache_list',
        kind: vscode.CompletionItemKind.Property,
    },
    {
        trigger: '%llm_cache_clear',
        label: '%llm_cache_clear',
        detail: 'Clear cache',
        documentation: 'Clear the media cache.',
        insertText: '%llm_cache_clear',
        kind: vscode.CompletionItemKind.Function,
    },
];
class CompletionProvider {
    constructor() { }
    provideCompletionItems(document, position, _token, _context) {
        const lineText = document.lineAt(position).text;
        const linePrefix = lineText.substring(0, position.character);
        // Only activate when the line starts with % or %%
        if (!linePrefix.match(/^\s*%%?\S*$/)) {
            return undefined;
        }
        const typedText = linePrefix.trim();
        return MAGIC_COMMANDS
            .filter(cmd => cmd.trigger.startsWith(typedText) || typedText === '%' || typedText === '%%')
            .map(cmd => {
            const item = new vscode.CompletionItem(cmd.label, cmd.kind);
            item.detail = cmd.detail;
            item.documentation = new vscode.MarkdownString(cmd.documentation);
            item.insertText = new vscode.SnippetString(cmd.insertText);
            // Replace the entire line content when accepting a completion
            const firstNonSpace = lineText.search(/\S/);
            item.range = new vscode.Range(position.line, firstNonSpace >= 0 ? firstNonSpace : 0, position.line, lineText.length);
            item.sortText = cmd.trigger;
            return item;
        });
    }
}
exports.CompletionProvider = CompletionProvider;
