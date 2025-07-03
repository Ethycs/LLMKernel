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

1. **Immediate**: Fix context tracking with cell reordering
2. **High**: Implement missing magic commands (`%llm_graph`, export/import)
3. **Medium**: Add visualization features
4. **Low**: Advanced features from roadmap

---

*Last updated: Based on codebase analysis*