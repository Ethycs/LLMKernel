# LLM Kernel TODO List

Based on analysis of the codebase vs README documentation.

## 🚨 Missing Features (High Priority)

### 1. Missing Magic Commands
- [ ] `%llm_graph` - Implement context visualization
  - Show dependency graph of cells
  - Visualize which cells are included in context
  - Use matplotlib/networkx for graph rendering
  
- [ ] `%llm_export_context <filename>` - Export context to JSON
  - Save conversation history
  - Save pinned cells
  - Save configuration state
  
- [ ] `%llm_import_context <filename>` - Import context from JSON
  - Restore conversation history
  - Restore pinned cells
  - Restore configuration

### 2. Cell Reorganization Context Bug
- [ ] Fix context tracking when cells are moved/reorganized in notebook
- [ ] Update execution tracker to handle cell reordering
- [ ] Ensure context window respects visual order vs execution order

## 🔧 Bug Fixes & Improvements

### 1. Async Event Loop Issues
- [x] Fixed: RuntimeError with event loop in Jupyter (added nest-asyncio)
- [ ] Test thoroughly in different environments (JupyterLab, VS Code, Classic Notebook)

### 2. Installation Issues
- [x] Fixed: Case-sensitive package detection (IPython vs ipython)
- [x] Fixed: Kernel installation with prefix flag
- [ ] Add better error messages for missing API keys
- [ ] Improve kernel discovery in VS Code

## ✨ Feature Enhancements

### 1. Visualization Features
- [ ] Implement context dependency graph visualization
  - Use networkx for graph structure
  - Use matplotlib or plotly for rendering
  - Show cell relationships and dependencies
  
- [ ] Add context preview widget
  - Show what's currently in context
  - Allow interactive context manipulation
  
### 2. Context Management Enhancements
- [ ] Implement context caching for performance
- [ ] Add semantic similarity using embeddings (already has TODO in code)
- [ ] Implement notebook-wide context search
- [ ] Add context size indicator in status

### 3. Export/Import Features
- [ ] Implement session save/restore
- [ ] Add context templates
- [ ] Support for sharing contexts between notebooks

## 🎯 Roadmap Features (from README)

### Phase 1: Core Improvements
- [ ] Enhanced semantic similarity with embeddings
- [ ] Visual context dependency graphs
- [ ] Context caching implementation

### Phase 2: Advanced Features  
- [ ] Notebook-wide context search
- [ ] Custom model fine-tuning integration
- [ ] Plugin system for custom providers

### Phase 3: Collaboration
- [ ] Collaborative context sharing
- [ ] Multi-user session support
- [ ] Context versioning

## 🚀 New Feature Ideas

### MCP (Model Context Protocol) Integration
- [ ] Implement MCP client for connecting to external tools
- [ ] Add `%llm_mcp_connect` command to connect to MCP servers
- [ ] Add `%llm_mcp_tools` to list available MCP tools
- [ ] Create `%%llm_mcp` cell magic for queries with MCP tools available
- [ ] Support for file access, database queries, API integrations via MCP

### GitHub Copilot Workspace Integration
- [ ] Scan workspace for relevant files and documentation
- [ ] Add `%llm_workspace_add <pattern>` to include files in context
- [ ] Implement `%llm_workspace_search` for semantic search in codebase
- [ ] Create document embeddings for workspace files
- [ ] Add `%%llm_with_workspace` for queries with workspace context

### Context Import/Export Enhancements
- [ ] Implement `%llm_export_context` with full session state
- [ ] Add `%llm_import_context` to restore previous sessions
- [ ] Create `%llm_context_merge` to combine multiple context files
- [ ] Support context templates and presets
- [ ] Add context versioning and diff capabilities

### Kernel Passthrough Mode
- [ ] Create wrapper mode to add LLM capabilities to ANY kernel
- [ ] Implement `%llm_passthrough <kernel>` command
- [ ] Add input preprocessing with LLM enhancement
- [ ] Add output postprocessing and error explanation
- [ ] Support for intercepting errors, warnings, and specific patterns
- [ ] Create `%llm_intercept` for configuring what to process

### Editable Context (Input AND Output)
- [ ] Implement `%llm_edit_cell <num>` to edit both input and output
- [ ] Add visual editor interface for modifying cell history
- [ ] Create `%llm_synthetic_cell` to add cells that never executed
- [ ] Track original vs edited history
- [ ] Allow rewriting conversation history for better context
- [ ] Add `%llm_rewrite_history` for bulk editing

## 📝 Documentation Updates

- [ ] Update README to reflect actual implemented features
- [ ] Add examples for each magic command
- [ ] Create API documentation
- [ ] Add troubleshooting guide for common issues
- [ ] Document context strategies in detail

## 🧪 Testing Improvements

- [ ] Add integration tests for VS Code
- [ ] Add tests for context reorganization
- [ ] Test with different notebook interfaces
- [ ] Add performance benchmarks
- [ ] Test with large notebooks (100+ cells)

## 🏗️ Architecture Improvements

- [ ] Refactor magic commands into separate module
- [ ] Improve error handling and user feedback
- [ ] Add progress indicators for long operations
- [ ] Implement proper logging throughout
- [ ] Add telemetry/usage analytics (opt-in)

## 🐛 Known Issues

1. **Context Window & Cell Order**
   - Moving cells in notebook doesn't update context properly
   - Need to track both execution order and visual order

2. **Graph Visualization**
   - `%llm_graph` command is documented but not implemented
   - No visual feedback for context dependencies

3. **Session Persistence**
   - No way to save/restore kernel state
   - Context is lost on kernel restart

## Priority Order

1. **Immediate**: 
   - Fix context tracking with cell reordering
   - Complete `%%hide` cell magic implementation ✅

2. **High**: 
   - Implement missing magic commands (`%llm_graph`, export/import)
   - MCP (Model Context Protocol) integration
   - Kernel passthrough mode

3. **Medium**: 
   - Add visualization features
   - GitHub Copilot Workspace integration
   - Editable context (input AND output editing)
   - Context import/export enhancements

4. **Low**: 
   - Advanced features from roadmap
   - Collaboration features
   - Plugin system

## Recently Completed ✅

- [x] Implemented `%%hide` cell magic to exclude cells from context
- [x] Added `%llm_unhide` command to unhide cells
- [x] Added `%llm_hidden` to show hidden cells
- [x] Implemented notebook context mode where cells ARE the context
- [x] Added `%llm_notebook_context` toggle
- [x] Created `%llm_context` to show current context window
- [x] Implemented `%llm_rerank` for LLM-based cell reranking by relevance
- [x] Added `%llm_rerank_clear` to restore original cell order
- [x] Implemented `%%meta` cell magic for custom context processing functions
- [x] Added `%llm_apply_meta` to apply custom filter/ranking/transform functions
- [x] Added `%llm_meta_list` to list defined meta functions
- [x] Implemented context persistence with save/load functionality
- [x] Added `%llm_context_save`, `%llm_context_load`, `%llm_context_reset`, `%llm_context_persist`

---

*Last updated: 2025-01-04 - Added reranking and context persistence to completed features*