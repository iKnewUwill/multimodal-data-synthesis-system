"""Web UI - Entry point for Gradio interface (金融财务数据合成)

This file serves as a backward-compatible entry point that delegates to the
refactored modular architecture in web_ui/ package.

For development and customization, see:
- web_ui/app.py - Main UI orchestration
- web_ui/components.py - UI component builders
- web_ui/handlers.py - Event handlers
- web_ui/styles.py - CSS styles
- services/html_generator.py - HTML generation
"""

# Import the new modular UI
from web_ui.app import MultimodalSynthesisUI, launch_ui

# Re-export for backward compatibility
__all__ = ["MultimodalSynthesisUI", "launch_ui"]


if __name__ == "__main__":
    launch_ui()
