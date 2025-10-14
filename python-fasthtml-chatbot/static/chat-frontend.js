// Simple chat interface using official Layercode JS SDK
// Handles connection, audio, VAD automatically

// Load config from page
const configElement = document.getElementById('layercode-config');
let config = {};
try {
  config = configElement ? JSON.parse(configElement.textContent) : {};
} catch (error) {
  console.error('Failed to parse Layercode config', error);
  config = {};
}

// UI elements
const UI = {
  connectBtn: document.getElementById('connect-btn'),
  muteBtn: document.getElementById('mute-btn'),
  statusLabel: document.getElementById('status-label'),
  messageList: document.getElementById('message-list'),
  textForm: document.getElementById('text-form'),
  textInput: document.getElementById('text-input'),
  errorBox: document.getElementById('error-box'),
};

// Status label classes
const LABEL_CLASSES = {
  disconnected: ['uk-label-destructive'],
  connecting: ['uk-label-secondary'],
  connected: ['bg-green-600', 'text-white', 'rounded-md', 'px-2', 'py-1', 'text-xs'],
  error: ['uk-label-destructive'],
};

const ALL_LABEL_CLASSES = [
  'uk-label-destructive', 'uk-label-secondary', 'uk-label-primary',
  'bg-green-600', 'bg-red-600', 'text-white'
];

// State
let layercodeClient = null;
let isConnected = false;
let messages = [];

// =============================================================================
// UI Updates
// =============================================================================

function setStatus(status, message) {
  if (UI.statusLabel) {
    UI.statusLabel.textContent = message;
    ALL_LABEL_CLASSES.forEach((cls) => {
      UI.statusLabel.classList.remove(cls);
    });
    const labelClasses = LABEL_CLASSES[status] || LABEL_CLASSES.error;
    labelClasses.forEach((cls) => {
      UI.statusLabel.classList.add(cls);
    });
  }

  if (UI.connectBtn) {
    if (status === 'connected') {
      UI.connectBtn.textContent = 'Disconnect';
      UI.connectBtn.disabled = false;
    } else if (status === 'connecting') {
      UI.connectBtn.textContent = 'Connecting…';
      UI.connectBtn.disabled = true;
    } else if (status === 'error') {
      UI.connectBtn.textContent = 'Reconnect';
      UI.connectBtn.disabled = false;
    } else {
      UI.connectBtn.textContent = 'Connect';
      UI.connectBtn.disabled = false;
    }
  }

  if (UI.muteBtn) {
    UI.muteBtn.disabled = status !== 'connected';
  }
}

function setError(message) {
  if (!UI.errorBox) return;
  if (message) {
    UI.errorBox.textContent = message;
    UI.errorBox.classList.remove('hidden');
  } else {
    UI.errorBox.textContent = '';
    UI.errorBox.classList.add('hidden');
  }
}

function addMessage(role, text) {
  if (!text.trim()) return;

  messages.push({ role, text });
  renderMessages();
}

function renderMessages() {
  if (!UI.messageList) return;

  UI.messageList.innerHTML = '';
  const fragment = document.createDocumentFragment();

  for (const message of messages) {
    const wrapper = document.createElement('div');
    wrapper.className = message.role === 'assistant'
      ? 'flex flex-col gap-1 items-start'
      : message.role === 'system'
        ? 'flex flex-col gap-1 items-center'
        : 'flex flex-col gap-1 items-end';

    const bubble = document.createElement('div');
    bubble.textContent = message.text;
    bubble.className = message.role === 'assistant'
      ? 'max-w-[80%] rounded-2xl bg-gray-200 dark:bg-gray-800 text-gray-900 dark:text-gray-100 px-4 py-2 text-sm'
      : message.role === 'system'
        ? 'max-w-[80%] rounded-xl bg-gray-100 dark:bg-gray-900 text-gray-600 dark:text-gray-400 px-3 py-2 text-xs uppercase tracking-wide'
        : 'max-w-[80%] rounded-2xl bg-indigo-600 text-white px-4 py-2 text-sm';

    wrapper.appendChild(bubble);
    fragment.appendChild(wrapper);
  }

  UI.messageList.appendChild(fragment);
  UI.messageList.scrollTop = UI.messageList.scrollHeight;
}

// =============================================================================
// Layercode Client
// =============================================================================

function initLayercodeClient() {
  if (!window.LayercodeClient) {
    setError('Layercode SDK not loaded. Check your internet connection.');
    return null;
  }

  if (!config.agentId) {
    setError('Missing agent ID. Set LAYERCODE_AGENT_ID in .env file.');
    return null;
  }

  try {
    const client = new window.LayercodeClient({
      agentId: config.agentId,
      authorizeSessionEndpoint: config.authorizePath || '/api/authorize',
      metadata: config.metadata || {},

      // Callbacks
      onConnect: ({ conversationId, config: agentConfig }) => {
        console.log('Connected to Layercode', conversationId, agentConfig);
        isConnected = true;
        setStatus('connected', 'Connected');
        setError('');
        addMessage('system', 'Connected! Speak or type a message.');
      },

      onDisconnect: () => {
        console.log('Disconnected from Layercode');
        isConnected = false;
        setStatus('disconnected', 'Disconnected');
      },

      onError: (error) => {
        console.error('Layercode error:', error);
        setStatus('error', 'Error');
        setError(error.message || 'Connection error');
      },

      onStatusChange: (status) => {
        console.log('Status changed:', status);
      },

      onMessage: (data) => {
        console.log('[onMessage]', data);

        switch (data?.type) {
          case 'turn.start':
            if (data.role === 'assistant') {
              // Assistant is starting to respond
              console.log('Assistant turn started');
            } else if (data.role === 'user') {
              // User is starting to speak
              console.log('User turn started');
            }
            break;

          case 'turn.end':
            console.log('Turn ended:', data.turn_id);
            break;

          case 'vad_events':
            // Voice activity detection
            console.log('VAD event:', data.event);
            break;

          case 'user.transcript':
            // Final user transcript (after speaking or text input)
            addMessage('user', data.content || '');
            break;

          case 'user.transcript.interim_delta':
          case 'user.transcript.delta':
            // Partial transcript while user is speaking
            // Could show interim text here if desired
            console.log('User speaking:', data.content);
            break;

          case 'response.text':
            // Assistant's text response
            addMessage('assistant', data.content || '');
            break;

          case 'response.data':
            // Custom data from agent (tool calls, etc)
            console.log('Agent data:', data.content);
            break;

          default:
            console.log('Unhandled message type:', data?.type);
        }
      },

      onDataMessage: (message) => {
        // This is for custom data messages only (tool calls, etc)
        console.log('Data message:', message);
      },

      onUserAmplitudeChange: (amplitude) => {
        // Could show mic level here if needed
      },

      onAgentAmplitudeChange: (amplitude) => {
        // Could show speaker level here if needed
      },
    });

    return client;
  } catch (error) {
    console.error('Failed to initialize Layercode client:', error);
    setError(error.message || 'Failed to initialize');
    return null;
  }
}

// =============================================================================
// Connection Management
// =============================================================================

async function connectOrDisconnect() {
  if (isConnected && layercodeClient) {
    // Disconnect
    try {
      await layercodeClient.disconnect();
    } catch (error) {
      console.error('Error disconnecting:', error);
    }
    layercodeClient = null;
    return;
  }

  // Connect
  setError('');
  setStatus('connecting', 'Connecting…');

  if (!layercodeClient) {
    layercodeClient = initLayercodeClient();
  }

  if (!layercodeClient) {
    setStatus('error', 'Initialization failed');
    return;
  }

  try {
    await layercodeClient.connect();
  } catch (error) {
    console.error('Failed to connect:', error);
    setStatus('error', 'Connection failed');
    setError(error.message || 'Failed to connect');
  }
}

// =============================================================================
// Text Input
// =============================================================================

function handleTextSubmit(event) {
  event.preventDefault();

  const value = UI.textInput ? UI.textInput.value.trim() : '';
  if (!value) {
    return;
  }

  if (!isConnected || !layercodeClient) {
    setError('Connect before sending a text message.');
    return;
  }

  try {
    layercodeClient.sendClientResponseText(value);
    addMessage('user', value);
    UI.textInput.value = '';
  } catch (error) {
    console.error('Failed to send text:', error);
    setError('Failed to send message');
  }
}

// =============================================================================
// Mute Toggle
// =============================================================================

function toggleMute() {
  // The SDK handles muting internally via triggerUserTurnStarted/Finished
  // For now, just show a message
  addMessage('system', 'Mute functionality coming soon');
}

// =============================================================================
// Event Handlers
// =============================================================================

function setupEventHandlers() {
  if (UI.connectBtn) {
    UI.connectBtn.addEventListener('click', connectOrDisconnect);
  }

  if (UI.muteBtn) {
    UI.muteBtn.addEventListener('click', toggleMute);
  }

  if (UI.textForm) {
    UI.textForm.addEventListener('submit', handleTextSubmit);
  }

  window.addEventListener('beforeunload', () => {
    if (layercodeClient && isConnected) {
      layercodeClient.disconnect().catch(() => {});
    }
  });
}

// =============================================================================
// Initialization
// =============================================================================

function init() {
  setStatus('disconnected', 'Disconnected');
  setupEventHandlers();

  // Check for secure context
  if (!window.isSecureContext) {
    const isIpAddress = window.location.hostname.match(/^\d+\.\d+\.\d+\.\d+$/);
    if (isIpAddress || window.location.hostname === '0.0.0.0') {
      const port = window.location.port || '8000';
      const suggestedUrl = `http://localhost:${port}`;
      setError(`⚠️ Microphone requires localhost. Try: ${suggestedUrl}`);
      console.warn(`For microphone access, use: ${suggestedUrl}`);
    }
  }
}

// Start when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
