"""
Installation script for LLM Kernel

This script handles the installation and registration of the LLM kernel with Jupyter.
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from jupyter_client.kernelspec import KernelSpecManager


def get_kernel_spec():
    """Get the kernel specification."""
    return {
        "argv": [
            sys.executable,
            "-m",
            "llm_kernel",
            "-f",
            "{connection_file}"
        ],
        "display_name": "LLM Kernel",
        "language": "python",
        "metadata": {
            "debugger": True
        }
    }


def install_kernel(user=True, prefix=None, kernel_name="llm_kernel"):
    """Install the LLM kernel."""
    print("Installing LLM Kernel...")
    
    # Create kernel spec
    kernel_spec = get_kernel_spec()
    
    # Get kernel spec manager
    ksm = KernelSpecManager()
    
    # Create temporary directory for kernel spec
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        kernel_dir = Path(temp_dir) / kernel_name
        kernel_dir.mkdir()
        
        # Write kernel.json
        with open(kernel_dir / "kernel.json", "w") as f:
            json.dump(kernel_spec, f, indent=2)
        
        # Copy logo if it exists
        logo_path = Path(__file__).parent / "logo-64x64.png"
        if logo_path.exists():
            shutil.copy(logo_path, kernel_dir / "logo-64x64.png")
        
        # Install the kernel
        ksm.install_kernel_spec(
            str(kernel_dir),
            kernel_name=kernel_name,
            user=user,
            prefix=prefix
        )
    
    print(f"✅ LLM Kernel installed successfully as '{kernel_name}'")
    
    # Show installation location
    if user:
        kernel_dir = ksm.user_kernel_dir
    else:
        kernel_dir = ksm.system_kernel_dir if prefix is None else os.path.join(prefix, 'share', 'jupyter', 'kernels')
    
    print(f"📁 Kernel installed to: {os.path.join(kernel_dir, kernel_name)}")


def uninstall_kernel(kernel_name="llm_kernel"):
    """Uninstall the LLM kernel."""
    print(f"Uninstalling LLM Kernel '{kernel_name}'...")
    
    ksm = KernelSpecManager()
    
    try:
        ksm.remove_kernel_spec(kernel_name)
        print(f"✅ LLM Kernel '{kernel_name}' uninstalled successfully")
    except Exception as e:
        print(f"❌ Error uninstalling kernel: {e}")
        return False
    
    return True


def list_kernels():
    """List all installed kernels."""
    print("📋 Installed Jupyter kernels:")
    
    ksm = KernelSpecManager()
    kernels = ksm.get_all_specs()
    
    for name, spec in kernels.items():
        display_name = spec.get('spec', {}).get('display_name', name)
        language = spec.get('spec', {}).get('language', 'unknown')
        location = spec.get('resource_dir', 'unknown')
        
        marker = "🤖" if name == "llm_kernel" else "📓"
        print(f"  {marker} {name}: {display_name} ({language})")
        if name == "llm_kernel":
            print(f"    📁 {location}")


def check_dependencies():
    """Check if required dependencies are installed."""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'ipykernel',
        'ipython',
        'ipywidgets',
        'litellm',
        'python-dotenv',
        'requests'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install them with:")
        print(f"  pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ All dependencies are installed")
    return True


def setup_environment():
    """Set up the environment for the LLM kernel."""
    print("🔧 Setting up LLM Kernel environment...")
    
    # Create user config directory
    config_dir = Path.home() / '.llm-kernel'
    config_dir.mkdir(exist_ok=True)
    print(f"📁 Created config directory: {config_dir}")
    
    # Create example .env file
    env_example = config_dir / '.env.example'
    if not env_example.exists():
        env_content = """# LLM Kernel Environment Configuration
# Copy this file to .env and fill in your API keys

# OpenAI API Key (get from https://platform.openai.com/api-keys)
OPENAI_API_KEY=your_openai_key_here

# Anthropic API Key (get from https://console.anthropic.com/)
ANTHROPIC_API_KEY=your_anthropic_key_here

# Google API Key (get from https://console.cloud.google.com/)
GOOGLE_API_KEY=your_google_key_here

# Optional: Default model to use
DEFAULT_LLM_MODEL=gpt-4o-mini

# Optional: Enable debug logging
LLM_KERNEL_DEBUG=false

# Optional: Context management settings
LLM_KERNEL_MAX_TOKENS=4000
LLM_KERNEL_CONTEXT_STRATEGY=smart
LLM_KERNEL_AUTO_PRUNE=true
"""
        with open(env_example, 'w') as f:
            f.write(env_content)
        print(f"📝 Created example environment file: {env_example}")
    
    # Create example config file
    from .config_manager import ConfigManager
    config_manager = ConfigManager()
    config_manager.create_example_configs()
    
    print("✅ Environment setup complete")
    print("\n📋 Next steps:")
    print("1. Copy .env.example to .env and add your API keys")
    print("2. Start Jupyter and select 'LLM Kernel' from the kernel list")
    print("3. Use magic commands like %llm_models and %%llm to get started")


def main():
    """Main installation script."""
    parser = argparse.ArgumentParser(description="Install LLM Kernel for Jupyter")
    parser.add_argument(
        'action',
        choices=['install', 'uninstall', 'list', 'check', 'setup'],
        help='Action to perform'
    )
    parser.add_argument(
        '--user',
        action='store_true',
        default=True,
        help='Install for current user only (default)'
    )
    parser.add_argument(
        '--system',
        action='store_true',
        help='Install system-wide'
    )
    parser.add_argument(
        '--prefix',
        help='Install to specific prefix'
    )
    parser.add_argument(
        '--name',
        default='llm_kernel',
        help='Kernel name (default: llm_kernel)'
    )
    
    args = parser.parse_args()
    
    if args.action == 'install':
        # Check dependencies first
        if not check_dependencies():
            print("\n❌ Please install missing dependencies before proceeding")
            return 1
        
        user = not args.system
        install_kernel(user=user, prefix=args.prefix, kernel_name=args.name)
        
        # Offer to set up environment
        response = input("\n🔧 Would you like to set up the environment? (y/N): ")
        if response.lower() in ('y', 'yes'):
            setup_environment()
    
    elif args.action == 'uninstall':
        uninstall_kernel(args.name)
    
    elif args.action == 'list':
        list_kernels()
    
    elif args.action == 'check':
        check_dependencies()
    
    elif args.action == 'setup':
        setup_environment()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
