# SlicerKimiAgent

An AI-powered assistant extension for [3D Slicer](https://www.slicer.org/) using the [KIMI API](https://platform.moonshot.cn/). This extension allows you to control 3D Slicer using natural language prompts, with the AI generating and executing Python code based on Slicer's comprehensive skill knowledge base.

![SlicerKimiAgent Screenshot](screenshots/screenshot1.png)

## Features

- **Natural Language Interface**: Control Slicer using plain English prompts
- **Intelligent Code Generation**: AI generates accurate Python code using Slicer's best practices
- **Streaming Responses with Thinking**: See the model's reasoning process in real time as it streams into the conversation box
- **Skill-Guided Responses**: Leverages Slicer's skill knowledge base for accurate API usage
- **Safe Execution**: AST-based code validation and sandboxed execution environment
- **Multi-turn Conversations**: Context-aware conversations with conversation history
- **Cost Tracking**: Monitor token usage and estimated API costs
- **Scene Awareness**: AI understands your current Slicer scene context

## Requirements

- 3D Slicer 5.2 or later
- A KIMI API key from [Moonshot AI Platform](https://platform.moonshot.cn/)

## Installation

### From Extension Manager (when published)

1. Open 3D Slicer
2. Go to `View` -> `Extension Manager`
3. Search for "SlicerKimiAgent"
4. Click `Install`
5. Restart 3D Slicer

### Manual Installation

1. Download or clone this repository
2. Copy the `SlicerKimiAgent` folder to your Slicer extensions directory:
   - Windows: `%USERPROFILE%\.slicer.org\Extensions-<version>`
   - macOS: `~/Library/Application Support/NA-MIC/Extensions-<version>`
   - Linux: `~/.config/NA-MIC/Extensions-<version>`
3. Restart 3D Slicer

## Setup

1. Open the SlicerKimiAgent module from the AI category
2. Click on `Settings` to expand the settings panel
3. Enter your KIMI API key
4. Select your preferred model (default: `kimi-k2.5`)
5. Click `Save Settings`

## Usage

### Basic Usage

1. Type your request in the chat box, for example:
   - "Load a sample volume"
   - "Create a volume rendering of the CT scan"
   - "Segment the lungs using thresholding"
   - "Export the model to STL"

2. The AI will analyze your request, stream its thinking into the conversation box, and generate Python code

3. Review the generated code in the code display area

4. Click `Execute Code` to run it (you'll be prompted to confirm for destructive operations)

### Example Prompts

```
Load the MRHead sample data and display it in 3D
```

```
Create a segmentation and add a segment called "Tumor"
```

```
Apply Gaussian smoothing to the current volume with sigma 2.0
```

```
Get the numpy array from the current volume and print its shape
```

```
Create fiducial points at specific coordinates: (10, 20, 30), (40, 50, 60)
```

## Architecture

The extension consists of several key components:

- **KimiClient**: HTTP client for KIMI API communication with SSE streaming and reasoning capture
- **SkillContextManager**: RAG-based retrieval of relevant Slicer knowledge
- **CodeValidator**: AST-based static analysis for security validation
- **SafeExecutor**: Sandboxed execution environment with output capture
- **ConversationStore**: Persistent conversation history management
- **SlicerCodeTemplates**: Library of common Slicer code patterns

## Security

The extension implements multiple security layers:

1. **Static Analysis**: All generated code is parsed and analyzed before execution
2. **Import Restrictions**: Dangerous modules (os, subprocess, socket, etc.) are blocked
3. **Function Blacklist**: Dangerous functions (eval, exec, etc.) are blocked
4. **User Confirmation**: Destructive operations (deleting nodes, overwriting files) require confirmation
5. **Execution Timeout**: Code execution is limited to prevent infinite loops

## Configuration

Settings are stored in Slicer's settings and include:

- **API Key**: Your KIMI API key (stored securely)
- **Model**: The KIMI model to use
- **Execution Timeout**: Maximum execution time for generated code
- **Safety Settings**: Toggle user confirmations for destructive operations

Note: Kimi API requests no longer use a 30-second client-side timeout. Generated Python code execution still has its own safety timeout.

## Troubleshooting

### "API key not configured" error
Make sure you've entered and saved your KIMI API key in the Settings panel.

### "Code validation failed" message
The generated code may contain operations that are blocked for security reasons. Review the error message and modify your prompt.

### Execution timeout
Generated Python code still has a safety timeout to prevent Slicer from hanging during execution.

### API requests take a long time
Kimi API requests no longer have a 30-second client-side timeout in this extension. If a response is slow, you should still see streamed thinking and partial answer content appear in the conversation box while the request is running.

## Development

### Project Structure

```
SlicerKimiAgent/
├── CMakeLists.txt              # Extension build configuration
├── SlicerKimiAgent.py         # Main module entry point
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── Resources/
│   ├── UI/
│   │   └── SlicerKimiAgent.ui  # Qt Designer UI file
│   ├── Icons/
│   │   └── SlicerKimiAgent.png # Module icon
│   └── Skills/
│       ├── skill_index.json    # Skill knowledge index
│       ├── api_patterns.json   # API documentation
│       ├── pitfalls.json       # Common pitfalls
│       └── ...
├── SlicerKimiAgentLib/         # Python library modules
│   ├── __init__.py
│   ├── KimiClient.py
│   ├── SkillContextManager.py
│   ├── CodeValidator.py
│   ├── SafeExecutor.py
│   ├── ConversationStore.py
│   └── SlicerCodeTemplates.py
└── Testing/
    └── SlicerKimiAgentTest.py  # Unit tests
```

### Running Tests

```python
import unittest
loader = unittest.TestLoader()
suite = loader.loadTestsFromName('SlicerKimiAgentTest')
unittest.TextTestRunner(verbosity=2).run(suite)
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [3D Slicer](https://www.slicer.org/) community for the comprehensive skill knowledge base
- [Moonshot AI](https://www.moonshot.cn/) for the KIMI API
- The Slicer Discourse community for API examples and best practices

## Support

For issues and feature requests, please use the [GitHub issue tracker](https://github.com/yourusername/SlicerKimiAgent/issues).

For general 3D Slicer questions, visit the [Slicer Discourse forum](https://discourse.slicer.org/).
