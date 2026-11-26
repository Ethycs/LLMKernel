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
from typing import Any, Dict, Optional
from jupyter_client.kernelspec import KernelSpecManager

from llm_kernel.wire import versions as _versions


def get_llmnb_kernel_spec(python_executable: Optional[str] = None) -> Dict[str, Any]:
    exe = python_executable or sys.executable
    return {
        "argv": [
            exe,
            "-m",
            "llm_kernel.jupyter_kernel",
            "-f",
            "{connection_file}",
        ],
        "display_name": "LLMNB",
        "language": "llmnb",
        "interrupt_mode": "message",
        "metadata": {
            "debugger": False,
            "llmnb": {
                "comm_target": _versions.COMM_TARGET,
                "wire_version": _versions.WIRE_VERSION,
                "schema_version": _versions.SCHEMA_VERSION,
            },
        },
    }


def get_kernel_spec(kernel_name: str = "llm_kernel") -> Dict[str, Any]:
    """Get the kernel specification."""
    if kernel_name == "llmnb":
        return get_llmnb_kernel_spec()
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


def install_kernel(
    user: bool = True,
    prefix: Optional[str] = None,
    kernel_name: str = "llm_kernel",
    spec: Optional[Dict[str, Any]] = None,
) -> str:
    if prefix:
        user = False
    kernel_spec = spec if spec is not None else get_kernel_spec(kernel_name)
    ksm = KernelSpecManager()

    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        kernel_dir = Path(temp_dir) / kernel_name
        kernel_dir.mkdir()

        with open(kernel_dir / "kernel.json", "w", encoding="utf-8") as f:
            json.dump(kernel_spec, f, indent=2)

        logo_path = Path(__file__).parent / "logo-64x64.png"
        if logo_path.exists():
            shutil.copy(logo_path, kernel_dir / "logo-64x64.png")

        dest = ksm.install_kernel_spec(
            str(kernel_dir),
            kernel_name=kernel_name,
            user=user,
            prefix=prefix,
        )

    print(f"Installed kernel '{kernel_name}' to: {dest}")
    return dest


def install_llmnb_kernel(
    user: bool = True,
    prefix: Optional[str] = None,
    kernel_name: str = "llmnb",
    python_executable: Optional[str] = None,
) -> str:
    spec = get_llmnb_kernel_spec(python_executable=python_executable)
    return install_kernel(user=user, prefix=prefix, kernel_name=kernel_name, spec=spec)


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
        
        marker = "🤖" if name in ("llm_kernel", "llmnb") else "📓"
        print(f"  {marker} {name}: {display_name} ({language})")
        if name in ("llm_kernel", "llmnb"):
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
            # Handle special cases
            if package.lower() == 'ipython':
                __import__('IPython')
            elif package == 'python-dotenv':
                __import__('dotenv')
            else:
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
    from llm_kernel.config_manager import ConfigManager
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
        '--sys-prefix',
        action='store_true',
        help='Install into sys.prefix (active virtualenv / pixi env)'
    )
    parser.add_argument(
        '--prefix',
        help='Install to specific prefix'
    )
    parser.add_argument(
        '--name',
        default='llmnb',
        help='Kernel name (default: llmnb; use llm_kernel for legacy IPython kernel)'
    )
    parser.add_argument(
        '--no-setup',
        action='store_true',
        help='Skip interactive environment setup prompt'
    )

    args = parser.parse_args()

    if args.action == 'install':
        if args.name != 'llmnb' and not check_dependencies():
            print("\nPlease install missing dependencies before proceeding")
            return 1

        prefix = sys.prefix if args.sys_prefix else args.prefix
        user = not (args.system or bool(prefix))
        install_kernel(user=user, prefix=prefix, kernel_name=args.name)

        if not args.no_setup and sys.stdin is not None and sys.stdin.isatty():
            response = input("\nWould you like to set up the environment? (y/N): ")
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
