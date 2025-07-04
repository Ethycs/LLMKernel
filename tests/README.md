# LLM Kernel Test Suite

This directory contains comprehensive tests for the LLM Kernel project, including all magic commands.

## Test Structure

```
tests/
├── .env.test              # Test environment variables
├── conftest.py            # Pytest configuration and fixtures
├── test_magic_commands.py # Unit tests for magic commands
├── test_integration_magic.py # Integration tests
└── README.md              # This file
```

## Running Tests with Pixi

The project uses pixi for environment management. To run tests:

### Setup Test Environment
```bash
# Install pixi if not already installed
curl -fsSL https://pixi.sh/install.sh | bash

# Create and activate test environment
pixi run -e test setup-test-env
```

### Run All Tests
```bash
# Using pixi tasks (recommended)
pixi run -e test test-all

# Or directly with pytest
pixi run -e test pytest tests/ -v
```

### Run Specific Test Suites
```bash
# Magic command tests only
pixi run -e test test-magic

# Unit tests only
pixi run -e test test-unit

# Integration tests only
pixi run -e test test-integration

# Quick tests (skip slow tests)
pixi run -e test test-quick
```

### Run Tests with Coverage
```bash
pixi run -e test test-coverage

# View coverage report
open htmlcov/index.html
```

### Advanced Testing Options
```bash
# Run tests in parallel
pixi run -e test test-parallel

# Run tests with debugger
pixi run -e test test-debug

# Watch for changes and re-run tests
pixi run -e test test-watch
```

## Running Tests without Pixi

If you prefer not to use pixi:

### Setup Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev,multimodal]"
pip install litellm fastmcp pytest-watch pytest-html
```

### Run Tests with Pytest
```bash
# All tests
pytest tests/ -v

# Magic tests only
pytest tests/ -m magic -v

# With coverage
pytest tests/ --cov=llm_kernel --cov-report=html -v

# Run tests in parallel
pytest tests/ -n auto -v

# Generate HTML report
pytest tests/ --html=report.html --self-contained-html -v
```

## Test Categories

Tests are organized with pytest markers:

- `@pytest.mark.unit` - Fast unit tests with mocked dependencies
- `@pytest.mark.integration` - Tests that use real kernel instances
- `@pytest.mark.slow` - Tests that take longer to run
- `@pytest.mark.magic` - Tests specific to magic commands
- `@pytest.mark.requires_api_key` - Tests that need real API keys

## Writing New Tests

### Unit Test Example
```python
def test_llm_model_command(mock_kernel):
    """Test %llm_model command."""
    # Setup
    mock_kernel.active_model = 'gpt-4o'
    
    # Create command instance
    cmd = ModelCommand(mock_kernel)
    
    # Test execution
    result = cmd.execute('')
    
    # Assertions
    assert "Current model: gpt-4o" in result
```

### Integration Test Example
```python
def test_real_model_switching(kernel_instance):
    """Test actual model switching."""
    # Execute magic command
    result = kernel_instance.do_execute(
        "%llm_model gpt-4o-mini",
        silent=False,
        store_history=True,
        user_expressions={},
        allow_stdin=False
    )
    
    # Verify
    assert result['status'] == 'ok'
    assert kernel_instance.active_model == 'gpt-4o-mini'
```

## Environment Variables

The test suite uses `.env.test` for configuration:

- `LLM_KERNEL_ENV=test` - Enables test mode
- `LLM_KERNEL_DEBUG=false` - Disables debug output
- Test API keys (non-functional)
- Test directories for isolation

## Continuous Integration

The test suite is designed to work with CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    pixi run -e test test-all
    pixi run -e test test-coverage
```

## Troubleshooting

### Missing Dependencies
```bash
# Ensure all test dependencies are installed
pixi install -e test
```

### Clean Test Artifacts
```bash
pixi run -e test clean-test
```

### Debug Failing Tests
```bash
# Run specific test with verbose output
pixi run -e test pytest tests/test_magic_commands.py::test_llm_model -vvs

# Run with debugger
pixi run -e test pytest tests/test_magic_commands.py::test_llm_model --pdb
```