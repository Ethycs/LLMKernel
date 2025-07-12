"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.activate = void 0;
const activate = (context) => {
    return {
        renderOutputItem(outputItem, element) {
            const data = outputItem.data();
            try {
                const llmResponse = JSON.parse(new TextDecoder().decode(data));
                element.innerHTML = createLLMResponseHTML(llmResponse);
                // Set up event handlers
                setupEventHandlers(element, llmResponse, context);
                // Handle streaming updates if supported
                if (llmResponse.streaming && context.postMessage) {
                    setupStreamingUpdates(element, llmResponse, context);
                }
            }
            catch (error) {
                element.innerHTML = `<div class="error">Error rendering LLM response: ${error}</div>`;
            }
        }
    };
};
exports.activate = activate;
function createLLMResponseHTML(data) {
    const { model = 'Unknown', content = '', cost = 0, tokens = 0, streaming = false, timestamp = new Date().toISOString(), context_size = 0, completion_tokens = 0, error = null } = data;
    const modelBadgeClass = getModelBadgeClass(model);
    const costClass = cost > 0.10 ? 'cost-warning' : '';
    return `
        <div class="llm-response-container">
            <div class="response-header">
                <div class="model-info">
                    <span class="model-badge ${modelBadgeClass}">${escapeHtml(model)}</span>
                    <span class="timestamp">${formatTimestamp(timestamp)}</span>
                </div>
                <div class="metrics">
                    <span class="cost-badge ${costClass}">
                        <svg class="icon" viewBox="0 0 16 16" width="14" height="14">
                            <path fill="currentColor" d="M7.5 1a6.5 6.5 0 1 0 0 13 6.5 6.5 0 0 0 0-13zm0 11.5a5 5 0 1 1 0-10 5 5 0 0 1 0 10zm.5-8.5h1v1h-1V3zm0 2h1v4.5h-1V5z"/>
                        </svg>
                        $${cost.toFixed(4)}
                    </span>
                    <span class="tokens-badge" title="Context: ${context_size}, Completion: ${completion_tokens}">
                        <svg class="icon" viewBox="0 0 16 16" width="14" height="14">
                            <path fill="currentColor" d="M3.5 5.5l.5-.5h8l.5.5v5l-.5.5h-8l-.5-.5v-5zm1 .5v4h7V6h-7z"/>
                        </svg>
                        ${tokens} tokens
                    </span>
                </div>
            </div>
            
            ${error ? `
                <div class="response-error">
                    <svg class="icon error-icon" viewBox="0 0 16 16" width="16" height="16">
                        <path fill="currentColor" d="M8 1a7 7 0 1 0 0 14A7 7 0 0 0 8 1zm0 13a6 6 0 1 1 0-12 6 6 0 0 1 0 12zm0-9a1 1 0 0 1 1 1v3a1 1 0 0 1-2 0V6a1 1 0 0 1 1-1zm0 7a1 1 0 1 1 0-2 1 1 0 0 1 0 2z"/>
                    </svg>
                    <span>${escapeHtml(error)}</span>
                </div>
            ` : ''}
            
            <div class="response-content ${streaming ? 'streaming' : ''}" data-streaming="${streaming}">
                ${streaming && !content ? '<div class="typing-indicator"><span></span><span></span><span></span></div>' : ''}
                <div class="markdown-content">${content ? renderMarkdown(content) : ''}</div>
            </div>
            
            <div class="response-actions">
                <button class="action-button" data-action="regenerate" title="Regenerate response">
                    <svg class="icon" viewBox="0 0 16 16" width="16" height="16">
                        <path fill="currentColor" d="M12.5 6.5a4 4 0 1 1-8 0 4 4 0 0 1 8 0zm1.5 0a5.5 5.5 0 1 0-11 0 5.5 5.5 0 0 0 11 0zM8 4.5v2h2v1H7V4.5h1z"/>
                    </svg>
                    Regenerate
                </button>
                <button class="action-button" data-action="compare" title="Compare with other models">
                    <svg class="icon" viewBox="0 0 16 16" width="16" height="16">
                        <path fill="currentColor" d="M9 3v1h4v1h1V2.5l-.5-.5H11l-.5.5V3H6v1h3zm-4 7v1H1v1h1v2.5l.5.5H5l.5-.5V14h4v-1H5v-1H1v-1h4v-1z"/>
                    </svg>
                    Compare
                </button>
                <button class="action-button" data-action="copy" title="Copy response">
                    <svg class="icon" viewBox="0 0 16 16" width="16" height="16">
                        <path fill="currentColor" d="M10 3v1h3v9h-3v1h4V3h-4zM5 5v9h5V5H5zM4 5v10h7V5H4z"/>
                    </svg>
                    Copy
                </button>
                <button class="action-button" data-action="pin" title="Pin to context">
                    <svg class="icon" viewBox="0 0 16 16" width="16" height="16">
                        <path fill="currentColor" d="M9.5 1.5l-.5-.5h-2l-.5.5L6 3H4.5L4 3.5V5l.5.5h.585l1.06 6.356L4.5 13.5v1l.5.5h3v-1.539l1.5-1.46H11V15l.5-.5v-1l-1.645-1.644L10.915 5.5h.585l.5-.5V3.5l-.5-.5H10l-.5-1.5zM7 2h2l.5 1.5h2V5H9.396l-1.06 6.356a.5.5 0 0 0 .118.444L10 13.293V14H8.5l-1.5 1.46V14H6v-.707l1.545-1.493a.5.5 0 0 0 .119-.444L6.604 5H4.5V3.5h2L7 2z"/>
                    </svg>
                    Pin
                </button>
            </div>
        </div>
        
        <style>
            .llm-response-container {
                font-family: var(--vscode-font-family);
                color: var(--vscode-editor-foreground);
                background: var(--vscode-editor-background);
                border: 1px solid var(--vscode-editorWidget-border);
                border-radius: 4px;
                overflow: hidden;
            }
            
            .response-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 8px 12px;
                background: var(--vscode-editorWidget-background);
                border-bottom: 1px solid var(--vscode-editorWidget-border);
            }
            
            .model-info {
                display: flex;
                align-items: center;
                gap: 8px;
            }
            
            .model-badge {
                padding: 2px 8px;
                border-radius: 12px;
                font-size: 12px;
                font-weight: 500;
            }
            
            .model-badge.gpt-4 {
                background: #7c3aed;
                color: white;
            }
            
            .model-badge.claude {
                background: #0ea5e9;
                color: white;
            }
            
            .model-badge.local {
                background: #22c55e;
                color: white;
            }
            
            .model-badge.unknown {
                background: var(--vscode-badge-background);
                color: var(--vscode-badge-foreground);
            }
            
            .timestamp {
                font-size: 11px;
                color: var(--vscode-descriptionForeground);
            }
            
            .metrics {
                display: flex;
                gap: 12px;
                align-items: center;
            }
            
            .cost-badge, .tokens-badge {
                display: flex;
                align-items: center;
                gap: 4px;
                font-size: 12px;
                color: var(--vscode-descriptionForeground);
            }
            
            .cost-badge.cost-warning {
                color: var(--vscode-editorWarning-foreground);
            }
            
            .icon {
                flex-shrink: 0;
            }
            
            .response-error {
                display: flex;
                align-items: center;
                gap: 8px;
                padding: 8px 12px;
                background: var(--vscode-inputValidation-errorBackground);
                border: 1px solid var(--vscode-inputValidation-errorBorder);
                color: var(--vscode-errorForeground);
            }
            
            .error-icon {
                color: var(--vscode-editorError-foreground);
            }
            
            .response-content {
                padding: 16px;
                min-height: 60px;
                max-height: 600px;
                overflow-y: auto;
            }
            
            .response-content.streaming {
                position: relative;
            }
            
            .typing-indicator {
                display: flex;
                gap: 4px;
                padding: 8px;
            }
            
            .typing-indicator span {
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: var(--vscode-activityBar-activeBorder);
                animation: typing 1.4s infinite;
            }
            
            .typing-indicator span:nth-child(2) {
                animation-delay: 0.2s;
            }
            
            .typing-indicator span:nth-child(3) {
                animation-delay: 0.4s;
            }
            
            @keyframes typing {
                0%, 60%, 100% {
                    opacity: 0.3;
                    transform: translateY(0);
                }
                30% {
                    opacity: 1;
                    transform: translateY(-10px);
                }
            }
            
            .markdown-content {
                line-height: 1.6;
            }
            
            .markdown-content pre {
                background: var(--vscode-textBlockQuote-background);
                border: 1px solid var(--vscode-textBlockQuote-border);
                border-radius: 4px;
                padding: 12px;
                overflow-x: auto;
            }
            
            .markdown-content code {
                background: var(--vscode-textBlockQuote-background);
                padding: 2px 4px;
                border-radius: 3px;
                font-family: var(--vscode-editor-font-family);
                font-size: 90%;
            }
            
            .markdown-content pre code {
                background: none;
                padding: 0;
            }
            
            .response-actions {
                display: flex;
                gap: 8px;
                padding: 8px 12px;
                background: var(--vscode-editorWidget-background);
                border-top: 1px solid var(--vscode-editorWidget-border);
            }
            
            .action-button {
                display: flex;
                align-items: center;
                gap: 4px;
                padding: 4px 12px;
                background: var(--vscode-button-secondaryBackground);
                color: var(--vscode-button-secondaryForeground);
                border: 1px solid transparent;
                border-radius: 4px;
                font-size: 12px;
                cursor: pointer;
                transition: all 0.2s;
            }
            
            .action-button:hover {
                background: var(--vscode-button-secondaryHoverBackground);
            }
            
            .action-button:active {
                transform: translateY(1px);
            }
        </style>
    `;
}
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
function formatTimestamp(timestamp) {
    const date = new Date(timestamp);
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    if (diff < 60000) {
        return 'just now';
    }
    else if (diff < 3600000) {
        const minutes = Math.floor(diff / 60000);
        return `${minutes}m ago`;
    }
    else if (diff < 86400000) {
        const hours = Math.floor(diff / 3600000);
        return `${hours}h ago`;
    }
    else {
        return date.toLocaleDateString();
    }
}
function getModelBadgeClass(model) {
    const modelLower = model.toLowerCase();
    if (modelLower.includes('gpt-4')) {
        return 'gpt-4';
    }
    else if (modelLower.includes('claude')) {
        return 'claude';
    }
    else if (modelLower.includes('local') || modelLower.includes('llama')) {
        return 'local';
    }
    return 'unknown';
}
function renderMarkdown(content) {
    // Basic markdown rendering (in production, use a proper markdown library)
    let html = escapeHtml(content);
    // Code blocks
    html = html.replace(/```(\w+)?\n([\s\S]*?)```/g, (match, lang, code) => {
        return `<pre><code class="language-${lang || 'text'}">${code.trim()}</code></pre>`;
    });
    // Inline code
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
    // Bold
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    // Italic
    html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');
    // Links
    html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');
    // Line breaks
    html = html.replace(/\n/g, '<br>');
    return html;
}
function setupEventHandlers(element, data, context) {
    const buttons = element.querySelectorAll('.action-button');
    buttons.forEach(button => {
        button.addEventListener('click', (e) => {
            const action = e.currentTarget.getAttribute('data-action');
            if (context.postMessage) {
                context.postMessage({
                    type: 'llm-action',
                    action: action,
                    data: data
                });
            }
            // Handle copy action directly
            if (action === 'copy') {
                navigator.clipboard.writeText(data.content || '').then(() => {
                    const btn = e.currentTarget;
                    const originalText = btn.innerHTML;
                    btn.innerHTML = btn.innerHTML.replace('Copy', 'Copied!');
                    setTimeout(() => {
                        btn.innerHTML = originalText;
                    }, 2000);
                });
            }
        });
    });
}
function setupStreamingUpdates(element, data, context) {
    const contentDiv = element.querySelector('.markdown-content');
    if (!contentDiv)
        return;
    // Listen for streaming updates
    const messageHandler = (event) => {
        if (event.type === 'streaming-update' && event.id === data.id) {
            contentDiv.innerHTML = renderMarkdown(event.content);
            // Update metrics if provided
            if (event.tokens !== undefined) {
                const tokensBadge = element.querySelector('.tokens-badge');
                if (tokensBadge) {
                    tokensBadge.textContent = `${event.tokens} tokens`;
                }
            }
            if (event.cost !== undefined) {
                const costBadge = element.querySelector('.cost-badge');
                if (costBadge) {
                    costBadge.textContent = `$${event.cost.toFixed(4)}`;
                    if (event.cost > 0.10) {
                        costBadge.classList.add('cost-warning');
                    }
                }
            }
            // Remove typing indicator if present
            const typingIndicator = element.querySelector('.typing-indicator');
            if (typingIndicator) {
                typingIndicator.remove();
            }
        }
    };
    context.onDidReceiveMessage(messageHandler);
}
