# Layercode Examples

Production-ready examples for building voice AI agents with [Layercode](https://layercode.com). These examples demonstrate best practices for integrating Layercode's voice AI platform with Python backends.

## About Layercode

**What is Layercode?** Layercode makes it easy for developers to build low-latency, production-ready voice AI agents. Connect your custom text-based backend to audio pipelines and turn your AI agent into a voice agent.

- **Website**: [layercode.com](https://layercode.com)
- **Documentation**: [layercode.mintlify.app/tutorials/getting-started](https://layercode.mintlify.app/tutorials/getting-started)
- **Dashboard**: [dash.layercode.com](https://dash.layercode.com/)

## Examples

### 1. Python FastAPI Backend

**[python-fastapi-backend/](./python-fastapi-backend/)**

A production-ready FastAPI backend that integrates with Layercode to build voice-powered AI agents. This example focuses on the backend webhook implementation without a frontend.

**Perfect for:**
- Building headless voice agents (phone, mobile apps, custom frontends)
- Understanding webhook handling and streaming responses
- Production deployments with FastAPI

**Features:**
- FastAPI webhook endpoints
- Pydantic-AI agent integration with streaming
- Signature verification for security
- Conversation history management
- Logfire observability
- Full type safety with Pydantic

**Quick start:**
```bash
cd python-fastapi-backend
make server  # Terminal 1
make tunnel  # Terminal 2
```

### 2. Python FastHTML Voice Chatbot

**[python-fasthtml-chatbot/](./python-fasthtml-chatbot/)**

A complete voice chat application with a web interface built using FastHTML. This example includes both frontend and backend in a single application, demonstrating the full stack integration.

**Perfect for:**
- Building web-based voice chat applications
- Rapid prototyping with FastHTML
- Understanding the complete voice agent flow
- Self-contained deployments

**Features:**
- FastHTML web interface with microphone support
- Real-time voice and text chat
- Pydantic-AI agent integration
- Session management and authorization
- Logfire observability
- Conversation history with async locks

**Quick start:**
```bash
cd python-fasthtml-chatbot
make app     # Terminal 1
make tunnel  # Terminal 2
# Open http://localhost:8000
```

## Prerequisites

All examples require:
- Python 3.10+
- [uv](https://github.com/astral-sh/uv) - Fast Python package installer
- A Layercode account - [Sign up here](https://dash.layercode.com/)
- OpenAI API key (or another LLM provider)
- Logfire token (free tier available at [logfire.pydantic.dev](https://logfire.pydantic.dev/))

## Getting Started

1. Choose an example from above
2. Follow the README in that example's directory
3. Each example includes:
   - Detailed setup instructions
   - Environment variable configuration
   - Testing guidelines
   - Deployment recommendations

## Common Setup Steps

All examples follow a similar pattern:

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone and enter the example
git clone <your-repo>
cd layercode-examples/<example-name>

# 3. Install dependencies
uv sync

# 4. Configure .env file
cp .env.example .env
# Edit .env with your credentials

# 5. Run the application (see example-specific README)
```

## Resources

- [Layercode Website](https://layercode.com)
- [Layercode Documentation](https://layercode.mintlify.app/tutorials/getting-started)
- [Layercode Dashboard](https://dash.layercode.com/)
- [Pydantic-AI Documentation](https://ai.pydantic.dev/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [FastHTML Documentation](https://docs.fastht.ml/)
- [uv Documentation](https://docs.astral.sh/uv/)
- [Logfire](https://logfire.pydantic.dev/)

## Support

For issues with:
- **These examples**: Open an issue in this repository
- **Layercode platform**: Check the [docs](https://layercode.mintlify.app/tutorials/getting-started) or visit [layercode.com](https://layercode.com)
- **Pydantic-AI**: Check [Pydantic-AI docs](https://ai.pydantic.dev/)

## License

MIT
