# VS Code Extension Test Guide

This guide explains how to run tests for the LLM Kernel Universal VS Code extension.

## Test Structure

The test suite is organized as follows:

```
src/test/
├── runTest.ts          # Test runner entry point
└── suite/
    ├── index.ts        # Test suite loader
    ├── extension.test.ts    # Extension activation and command tests
    ├── chatMode.test.ts     # Chat mode functionality tests
    ├── llmOverlay.test.ts   # LLM overlay system tests
    ├── providers.test.ts    # Provider tests
    ├── services.test.ts     # Service tests
    ├── serializer.test.ts   # Notebook serializer tests
    └── helpers/
        └── mockVscode.ts    # Mock objects for testing
```

## Running Tests

### From Command Line

```bash
# Run all tests
npm test

# Run tests with compilation
npm run test:unit

# Run tests with coverage (if nyc is installed)
npm run test:coverage
```

### From VS Code

1. Open the Command Palette (Ctrl+Shift+P / Cmd+Shift+P)
2. Run "Debug: Select and Start Debugging"
3. Choose "Extension Tests"

Or use the Run and Debug view (Ctrl+Shift+D) and select "Extension Tests" from the dropdown.

## Test Categories

### 1. Extension Tests (`extension.test.ts`)
- Extension activation
- Command registration
- Configuration validation
- New notebook creation

### 2. Chat Mode Tests (`chatMode.test.ts`)
- Enabling/disabling chat mode
- Natural language detection
- Chat mode toggle functionality
- Metadata persistence

### 3. LLM Overlay Tests (`llmOverlay.test.ts`)
- Overlay activation/deactivation
- Multiple notebook support
- Metadata management
- Toggle functionality

### 4. Provider Tests (`providers.test.ts`)
- Kernel provider operations
- Context provider save/load
- Completion provider functionality

### 5. Service Tests (`services.test.ts`)
- Kernel service lifecycle
- API service methods
- Status bar manager
- Notebook service execution

### 6. Serializer Tests (`serializer.test.ts`)
- Notebook deserialization
- Notebook serialization
- Output handling
- Welcome notebook creation

## Writing New Tests

### Test Structure

```typescript
suite('Feature Test Suite', () => {
    setup(() => {
        // Setup before each test
    });

    teardown(() => {
        // Cleanup after each test
    });

    test('Should do something', async () => {
        // Test implementation
        assert.strictEqual(actual, expected);
    });
});
```

### Using Mock Objects

```typescript
import { createMockNotebook, createMockCell } from './helpers/mockVscode';

test('Test with mock notebook', async () => {
    const mockCell = createMockCell('print("test")');
    const mockNotebook = createMockNotebook([mockCell]);
    
    // Use mock objects in test
});
```

## Debugging Tests

1. Set breakpoints in test files
2. Run tests in debug mode using VS Code's debugger
3. Check the Debug Console for output

## Test Best Practices

1. **Isolation**: Each test should be independent
2. **Cleanup**: Always clean up resources (close editors, dispose objects)
3. **Assertions**: Use specific assertions (`strictEqual` over `equal`)
4. **Async**: Use `async/await` for asynchronous operations
5. **Mocking**: Mock external dependencies and VS Code APIs when needed

## Common Issues

### Tests Won't Run
- Ensure TypeScript compilation succeeded: `npm run compile`
- Check that all dependencies are installed: `npm install`

### VS Code APIs Not Available
- Tests must run in VS Code's Extension Host
- Use the provided launch configuration

### Timeout Errors
- Increase timeout in `src/test/suite/index.ts` if needed
- Default is 10 seconds per test

## CI/CD Integration

For GitHub Actions or other CI systems:

```yaml
- name: Run tests
  uses: GabrielBB/xvfb-action@v1
  with:
    run: npm test
```

Tests require a display server (xvfb) when running in headless environments.