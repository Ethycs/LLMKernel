# Seamless VS Code Notebook Integration for LLM Kernel

## Native Notebook Cell UI Enhancements

### 1. **Cell Toolbar Integration**

```typescript
// Add buttons directly to notebook cell toolbar
interface LLMCellToolbar {
  buttons: [
    {
      icon: "📎",
      tooltip: "Paste from clipboard",
      action: async (cell) => {
        const content = await getClipboardContent();
        if (content.type === 'image') {
          cell.attachMedia(content);
          cell.showPreview(content);
        }
      }
    },
    {
      icon: "🖼️",
      tooltip: "Add image",
      action: (cell) => showImagePicker(cell)
    },
    {
      icon: "👁️",
      tooltip: "Toggle in context",
      action: (cell) => cell.toggleContext()
    },
    {
      icon: "🤖",
      tooltip: "Send to LLM",
      action: (cell) => cell.executeLLM()
    }
  ]
}
```

### 2. **Native Paste Handling**

```typescript
// Override notebook paste behavior
class NotebookPasteHandler {
  async handlePaste(event: ClipboardEvent, cell: NotebookCell) {
    const items = event.clipboardData?.items;
    
    for (const item of items) {
      // Handle images
      if (item.type.startsWith('image/')) {
        event.preventDefault();
        
        const blob = item.getAsFile();
        const attachment = await this.processImage(blob);
        
        // Show inline preview
        cell.outputs.push({
          type: 'image/png',
          data: attachment.data
        });
        
        // Attach to cell metadata
        cell.metadata.llmAttachments = cell.metadata.llmAttachments || [];
        cell.metadata.llmAttachments.push(attachment);
        
        // Show toast
        vscode.window.showInformationMessage('Image attached to cell');
      }
    }
  }
}
```

### 3. **Drag & Drop Support**

```typescript
// Native drag & drop for files
class CellDropHandler {
  handleDrop(files: File[], cell: NotebookCell) {
    files.forEach(file => {
      if (file.type.startsWith('image/')) {
        cell.attachImage(file);
      } else if (file.name.endsWith('.pdf')) {
        this.showPdfOptions(file, cell);
      }
    });
  }
}
```

### 4. **Inline Context Indicators**

```typescript
// Visual indicators in cells
interface CellDecorations {
  // Gutter indicators
  contextIndicator: {
    included: "🟢",  // Green dot for included
    excluded: "🔴",  // Red dot for excluded
    hidden: "🙈"     // Hidden from LLM
  },
  
  // Attachment badges
  attachmentBadges: {
    show: true,
    position: 'top-right',
    items: ['📎 2 images', '📄 1 PDF']
  }
}
```

### 5. **Smart Cell Actions**

```typescript
// Context menu additions
interface CellContextMenu {
  items: [
    {
      label: "Send to LLM",
      keybinding: "Ctrl+Enter",
      action: (cell) => executeLLMQuery(cell)
    },
    {
      label: "Add to Context",
      keybinding: "Ctrl+Shift+C",
      action: (cell) => addToContext(cell)
    },
    {
      label: "Attach Media...",
      submenu: [
        "From Clipboard (Ctrl+V)",
        "From File...",
        "From URL...",
        "Screenshot Region..."
      ]
    }
  ]
}
```

## Seamless Chat Integration

### 1. **Inline LLM Responses**

```typescript
// LLM responses appear as cell outputs
interface LLMCellOutput {
  // User cell remains editable
  userCell: {
    source: "What's in this image?",
    attachments: [imageData],
    toolbar: [sendButton, attachButton]
  },
  
  // Response appears below
  outputCell: {
    type: 'llm-response',
    model: 'gpt-4o',
    content: "I can see a diagram showing...",
    actions: [copyButton, regenerateButton]
  }
}
```

### 2. **Natural Cell Flow**

```typescript
// Cells automatically flow like chat
class NotebookChatMode {
  onCellExecute(cell: NotebookCell) {
    if (cell.hasAttachments() || this.isQuestion(cell.source)) {
      // Automatically send to LLM
      const response = await this.llm.query(cell);
      
      // Insert response as next cell
      this.notebook.insertCell(cell.index + 1, {
        type: 'markdown',
        source: response,
        metadata: { isLLMResponse: true }
      });
      
      // Auto-focus next input cell
      this.notebook.createNewCell(cell.index + 2);
    }
  }
}
```

## Rich Media Handling

### 1. **Image Preview Gallery**

```typescript
// Attached images show as thumbnails
interface MediaGallery {
  attachments: [
    {
      type: 'image',
      thumbnail: base64Thumbnail,
      fullSize: base64Full,
      actions: ['remove', 'preview', 'replace']
    }
  ],
  
  layout: 'horizontal-scroll',
  maxHeight: '100px'
}
```

### 2. **PDF Page Selector**

```typescript
// Interactive PDF page selection
interface PDFSelector {
  showModal(pdfFile: File) {
    return {
      preview: true,
      multiSelect: true,
      options: {
        extractText: boolean,
        asImages: boolean,
        pages: number[]
      }
    };
  }
}
```

## Smart Context Management

### 1. **Visual Context Window**

```typescript
// Sidebar showing context
interface ContextPanel {
  sections: {
    included: Cell[],
    excluded: Cell[],
    attachments: Media[],
    tokens: {
      used: 5420,
      limit: 8000,
      visual: ProgressBar
    }
  },
  
  actions: {
    dragToReorder: true,
    clickToToggle: true,
    clearAll: Button
  }
}
```

### 2. **Auto-Context Detection**

```typescript
// Automatically manage context
class SmartContext {
  onCellChange(cell: NotebookCell) {
    // Auto-include cells with questions
    if (this.looksLikeQuestion(cell.source)) {
      cell.includeInContext = true;
    }
    
    // Auto-include cells with code outputs
    if (cell.outputs.length > 0 && cell.language === 'python') {
      cell.includeInContext = true;
    }
    
    // Warn if context too large
    if (this.contextTokens > this.tokenLimit * 0.9) {
      this.suggestContextPruning();
    }
  }
}
```

## Implementation Details

### Cell Metadata Extension
```json
{
  "llmkernel": {
    "includeInContext": true,
    "attachments": [
      {
        "type": "image/png",
        "data": "base64...",
        "name": "screenshot.png"
      }
    ],
    "cellRole": "user|assistant|system",
    "tokens": 142
  }
}
```

### Keybindings
```json
{
  "key": "ctrl+enter",
  "command": "llmkernel.sendToLLM",
  "when": "notebookCellType == 'code' || notebookCellType == 'markdown'"
},
{
  "key": "ctrl+v",
  "command": "llmkernel.pasteAsAttachment",
  "when": "notebookEditorFocused && llmkernel.chatMode"
},
{
  "key": "ctrl+shift+c",
  "command": "llmkernel.toggleCellContext",
  "when": "notebookCellFocused"
}
```

### Settings
```json
{
  "llmkernel.seamlessMode": true,
  "llmkernel.autoAttachImages": true,
  "llmkernel.showContextIndicators": true,
  "llmkernel.pasteAsAttachment": true,
  "llmkernel.autoSendToLLM": true,
  "llmkernel.inlineResponses": true
}
```

## The Experience

1. **Paste an image** → Automatically attached and previewed
2. **Type a question** → Send with Ctrl+Enter
3. **Get response** → Appears as next cell
4. **Drag cells** → Reorder context visually
5. **See indicators** → Know what's included at a glance

This creates a truly native, seamless notebook experience where LLM interaction feels like a natural part of the workflow!