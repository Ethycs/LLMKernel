"""
Pytest configuration and fixtures for LLM Kernel tests.

This file is automatically loaded by pytest and provides:
- Test environment setup
- Shared fixtures
- Test isolation
- Mock configurations
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import pytest
import json
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# ==================== Environment Setup ====================

@pytest.fixture(scope="session", autouse=True)
def test_environment():
    """Set up test environment for the entire test session."""
    # Save original environment
    original_env = os.environ.copy()
    
    # Set test environment variables
    os.environ['LLM_KERNEL_ENV'] = 'test'
    os.environ['LLM_KERNEL_LOGGING'] = 'false'  # Disable logging during tests
    os.environ['LLM_KERNEL_DEBUG'] = 'false'
    
    # Use test API keys (non-functional keys for testing)
    if not os.getenv('OPENAI_API_KEY'):
        os.environ['OPENAI_API_KEY'] = 'test-key-not-real'
    if not os.getenv('ANTHROPIC_API_KEY'):
        os.environ['ANTHROPIC_API_KEY'] = 'test-key-not-real'
    if not os.getenv('GOOGLE_API_KEY'):
        os.environ['GOOGLE_API_KEY'] = 'test-key-not-real'
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture(scope="function")
def isolated_env():
    """Create an isolated environment for each test."""
    # Create temporary directories
    temp_dir = tempfile.mkdtemp(prefix="llm_kernel_test_")
    temp_home = Path(temp_dir) / "home"
    temp_cache = Path(temp_dir) / "cache"
    temp_config = Path(temp_dir) / "config"
    
    temp_home.mkdir(parents=True)
    temp_cache.mkdir(parents=True)
    temp_config.mkdir(parents=True)
    
    # Override paths
    original_home = os.environ.get('HOME')
    original_cache = os.environ.get('LLM_KERNEL_CACHE_DIR')
    
    os.environ['HOME'] = str(temp_home)
    os.environ['LLM_KERNEL_CACHE_DIR'] = str(temp_cache)
    os.environ['LLM_KERNEL_CONFIG_DIR'] = str(temp_config)
    
    yield {
        'temp_dir': temp_dir,
        'home': temp_home,
        'cache': temp_cache,
        'config': temp_config
    }
    
    # Cleanup
    if original_home:
        os.environ['HOME'] = original_home
    else:
        os.environ.pop('HOME', None)
        
    if original_cache:
        os.environ['LLM_KERNEL_CACHE_DIR'] = original_cache
    else:
        os.environ.pop('LLM_KERNEL_CACHE_DIR', None)
    
    os.environ.pop('LLM_KERNEL_CONFIG_DIR', None)
    
    # Remove temporary directory
    shutil.rmtree(temp_dir, ignore_errors=True)


# ==================== Mock Fixtures ====================

@pytest.fixture
def mock_llm_response():
    """Mock LLM API responses."""
    def _mock_response(content="Test response", model="gpt-4o"):
        return Mock(
            choices=[
                Mock(
                    message=Mock(content=content),
                    finish_reason="stop"
                )
            ],
            usage=Mock(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30
            ),
            model=model
        )
    return _mock_response


@pytest.fixture
def mock_litellm(mock_llm_response):
    """Mock litellm module."""
    with patch('llm_kernel.kernel.litellm') as mock:
        # Mock completion function
        mock.completion = Mock(return_value=mock_llm_response())
        
        # Mock acompletion for async
        async def mock_acompletion(*args, **kwargs):
            return mock_llm_response()
        mock.acompletion = mock_acompletion
        
        yield mock


@pytest.fixture
def mock_api_clients():
    """Mock API client initialization."""
    clients = {}
    
    # Mock OpenAI client
    with patch('openai.OpenAI') as mock_openai:
        mock_client = Mock()
        mock_client.files = Mock()
        mock_client.files.create = Mock(return_value=Mock(
            id="file-test123",
            bytes=1000,
            created_at=1234567890,
            filename="test.pdf",
            status="processed"
        ))
        mock_openai.return_value = mock_client
        clients['openai'] = mock_client
    
    # Mock Anthropic client
    with patch('anthropic.Anthropic') as mock_anthropic:
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        clients['anthropic'] = mock_client
    
    # Mock Google client
    with patch('google.generativeai.configure') as mock_genai:
        clients['google'] = mock_genai
    
    yield clients


# ==================== Kernel Fixtures ====================

@pytest.fixture
def mock_kernel():
    """Create a mock kernel for unit tests."""
    from llm_kernel.kernel import LLMKernel
    
    kernel = Mock(spec=LLMKernel)
    
    # Set up attributes
    kernel.llm_clients = {
        'gpt-4o': 'gpt-4o',
        'gpt-4o-mini': 'gpt-4o-mini',
        'claude-3-opus': 'claude-3-opus-20240229',
        'gemini-2.5-pro': 'gemini/gemini-2.5-pro'
    }
    kernel.active_model = 'gpt-4o'
    kernel.conversation_history = []
    kernel.context_cells = []
    kernel.pinned_cells = set()
    kernel.chat_mode = False
    kernel.display_mode = 'inline'
    kernel.execution_count = 1
    kernel.session_exchanges = []
    kernel.session_costs = {'total': 0.0, 'by_model': {}}
    kernel.log = logging.getLogger('test')
    
    # Mock methods
    kernel.get_notebook_cells_as_context = Mock(return_value=[])
    kernel.query_llm_async = Mock(return_value="Test response")
    kernel.track_exchange = Mock()
    kernel.update_context = Mock()
    kernel.count_tokens = Mock(return_value=100)
    kernel.export_context = Mock(return_value={})
    kernel.import_context = Mock()
    
    return kernel


@pytest.fixture
def real_kernel(isolated_env, mock_litellm):
    """Create a real kernel instance for integration tests."""
    from llm_kernel.kernel import LLMKernel
    
    # Create kernel with mocked external dependencies
    kernel = LLMKernel()
    
    # Ensure kernel uses test environment
    kernel._test_mode = True
    
    yield kernel
    
    # Cleanup
    try:
        kernel.do_shutdown(restart=False)
    except:
        pass


# ==================== Test Data Fixtures ====================

@pytest.fixture
def sample_notebook_cells():
    """Sample notebook cells for testing."""
    return [
        {
            'cell_type': 'code',
            'source': 'import pandas as pd\ndata = pd.read_csv("test.csv")',
            'execution_count': 1
        },
        {
            'cell_type': 'markdown',
            'source': '# Test Notebook\nThis is a test.'
        },
        {
            'cell_type': 'code',
            'source': 'print("Hello, World!")',
            'execution_count': 2
        }
    ]


@pytest.fixture
def sample_conversation():
    """Sample conversation history."""
    return [
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a high-level programming language."},
        {"role": "user", "content": "Can you give an example?"},
        {"role": "assistant", "content": "Here's a simple example:\n```python\nprint('Hello, World!')\n```"}
    ]


@pytest.fixture
def temp_pdf_file():
    """Create a temporary PDF file for testing."""
    import io
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        
        # Create PDF in memory
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        c.drawString(100, 750, "Test PDF Document")
        c.drawString(100, 700, "This is a test PDF for LLM Kernel.")
        c.save()
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pdf', delete=False) as f:
            f.write(buffer.getvalue())
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        os.unlink(temp_path)
        
    except ImportError:
        # If reportlab not available, create a simple file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdf', delete=False) as f:
            f.write("%PDF-1.4\nTest PDF")
            temp_path = f.name
        
        yield temp_path
        
        os.unlink(temp_path)


@pytest.fixture
def temp_image_file():
    """Create a temporary image file for testing."""
    try:
        from PIL import Image
        
        # Create a simple image
        img = Image.new('RGB', (100, 100), color='red')
        
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.png', delete=False) as f:
            img.save(f, 'PNG')
            temp_path = f.name
        
        yield temp_path
        
        os.unlink(temp_path)
        
    except ImportError:
        # If PIL not available, create a simple file
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.png', delete=False) as f:
            f.write(b'PNG_TEST_DATA')
            temp_path = f.name
        
        yield temp_path
        
        os.unlink(temp_path)


# ==================== Helper Fixtures ====================

@pytest.fixture
def capture_output():
    """Capture stdout/stderr for testing print statements."""
    from io import StringIO
    import sys
    
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    
    stdout_capture = StringIO()
    stderr_capture = StringIO()
    
    sys.stdout = stdout_capture
    sys.stderr = stderr_capture
    
    yield {
        'stdout': stdout_capture,
        'stderr': stderr_capture
    }
    
    sys.stdout = old_stdout
    sys.stderr = old_stderr


@pytest.fixture
def mock_ipython_display():
    """Mock IPython display functions."""
    with patch('IPython.display.display') as mock_display:
        with patch('IPython.display.Markdown') as mock_markdown:
            with patch('IPython.display.HTML') as mock_html:
                with patch('IPython.display.Image') as mock_image:
                    yield {
                        'display': mock_display,
                        'markdown': mock_markdown,
                        'html': mock_html,
                        'image': mock_image
                    }


# ==================== Configuration ====================

def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Add custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "requires_api_key: marks tests that require real API keys"
    )
    config.addinivalue_line(
        "markers", "magic: marks tests related to magic commands"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add unit marker to tests in test_magic_commands.py
        if "test_magic_commands.py" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
            item.add_marker(pytest.mark.magic)
        
        # Add integration marker to tests in test_integration_magic.py
        if "test_integration_magic.py" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
            item.add_marker(pytest.mark.magic)
        
        # Mark tests that use real API calls
        if "test_llm_query" in item.name or "test_async_llm_query" in item.name:
            item.add_marker(pytest.mark.requires_api_key)