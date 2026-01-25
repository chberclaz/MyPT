/**
 * MyPT Web Application - Vanilla JavaScript
 * Self-contained - No external dependencies
 * Version: 2026-01-25-v2 (mode selector fix)
 */

// ============================================================================
// State Management
// ============================================================================

const AppState = {
    chat: {
        messages: [],
        inputText: '',
        isLoading: false,
        selectedModel: '',
        models: [],
        mode: 'agentic',  // 'conversation' or 'agentic'
        verbose: false,
        documents: [],
        indexStatus: { chunks: 0, lastUpdated: null }
    },
    training: {
        mode: 'pretrain',
        isTraining: false,
        progress: { step: 0, maxSteps: 5000, trainLoss: null, valLoss: null, eta: null },
        logs: [],
        config: {
            modelSize: '150M',
            baseModel: '',
            datasetDir: '',
            outputName: '',
            maxIters: 5000,
            evalInterval: 50,
            learningRate: '3e-4',
            batchSize: 'auto',
            warmupIters: '0'
        }
    }
};

// ============================================================================
// Utility Functions
// ============================================================================

function $(selector) {
    return document.querySelector(selector);
}

function $$(selector) {
    return document.querySelectorAll(selector);
}

function formatTime(timestamp) {
    if (!timestamp) return '';
    const date = new Date(timestamp);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ============================================================================
// Chat Page Functions
// ============================================================================

async function initChatPage() {
    console.log('Initializing chat page...');
    
    // Load models
    await loadModels();
    
    // Load workspace info
    await loadWorkspaceInfo();
    
    // Setup event listeners
    setupChatEventListeners();
    
    // Render initial state
    renderChatUI();
    
    // Update system status in header
    updateSystemStatus();
}

function updateSystemStatus() {
    console.log('[DEBUG] updateSystemStatus() called');
    
    const statusDot = $('#system-status-dot');
    const statusText = $('#system-status-text');
    const statusBadge = $('#system-status-badge');
    
    console.log('[DEBUG] Header elements found:', {
        statusDot: !!statusDot,
        statusText: !!statusText,
        statusBadge: !!statusBadge
    });
    
    const hasModels = AppState.chat.models.length > 0;
    const hasIndex = (AppState.chat.indexStatus.chunks || 0) > 0;
    
    console.log('[DEBUG] Status check:', { hasModels, hasIndex, models: AppState.chat.models.length, chunks: AppState.chat.indexStatus.chunks });
    
    if (statusText && statusDot) {
        if (hasModels && hasIndex) {
            console.log('[DEBUG] Setting status to Ready');
            statusText.textContent = 'Ready';
            statusDot.classList.add('active');
            if (statusBadge) statusBadge.setAttribute('data-tooltip', 'Models loaded, index ready');
        } else if (hasModels) {
            console.log('[DEBUG] Setting status to No Index');
            statusText.textContent = 'No Index';
            statusDot.classList.add('warning');
            if (statusBadge) statusBadge.setAttribute('data-tooltip', 'Models loaded, no index');
        } else {
            console.log('[DEBUG] Setting status to No Models');
            statusText.textContent = 'No Models';
            statusDot.classList.remove('active', 'warning');
            if (statusBadge) statusBadge.setAttribute('data-tooltip', 'No models available');
        }
    } else {
        console.log('[DEBUG] Header status elements NOT FOUND!');
    }
}

async function loadModels() {
    try {
        const response = await fetch('/api/chat/models');
        const data = await response.json();
        AppState.chat.models = data.models || [];
        if (AppState.chat.models.length > 0) {
            AppState.chat.selectedModel = AppState.chat.models[0];
        }
        renderModelSelector();
    } catch (error) {
        console.error('Failed to load models:', error);
    }
}

async function loadWorkspaceInfo() {
    console.log('[DEBUG] loadWorkspaceInfo() called');
    try {
        const response = await fetch('/api/workspace/info');
        const data = await response.json();
        console.log('[DEBUG] Workspace API response:', data);
        
        AppState.chat.documents = data.documents || [];
        AppState.chat.indexStatus = {
            chunks: data.num_chunks || 0,
            hasIndex: data.has_index || false,
            lastUpdated: data.last_updated
        };
        console.log('[DEBUG] AppState.chat.indexStatus:', AppState.chat.indexStatus);
        
        renderWorkspaceInfo();
    } catch (error) {
        console.error('Failed to load workspace info:', error);
    }
}

function setupChatEventListeners() {
    // Send button
    const sendBtn = $('#send-btn');
    if (sendBtn) {
        sendBtn.addEventListener('click', sendMessage);
    }
    
    // Input field - Enter to send
    const chatInput = $('#chat-input');
    if (chatInput) {
        chatInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
    }
    
    // Clear button
    const clearBtn = $('#clear-btn');
    if (clearBtn) {
        clearBtn.addEventListener('click', clearChat);
    }
    
    // Rebuild index button
    const rebuildBtn = $('#rebuild-index-btn');
    if (rebuildBtn) {
        rebuildBtn.addEventListener('click', rebuildIndex);
    }
    
    // Model selector
    const modelSelect = $('#model-select');
    if (modelSelect) {
        modelSelect.addEventListener('change', (e) => {
            AppState.chat.selectedModel = e.target.value;
        });
    }
    
    // Verbose checkbox
    const verboseCheck = $('#verbose-check');
    if (verboseCheck) {
        verboseCheck.addEventListener('change', (e) => {
            AppState.chat.verbose = e.target.checked;
        });
    }
    
    // Mode selector
    const modeSelect = $('#mode-select');
    console.log('[DEBUG] mode-select element:', modeSelect);
    if (modeSelect) {
        // Sync initial value from HTML to AppState
        AppState.chat.mode = modeSelect.value;
        console.log(`[MODE INIT] Starting with: ${AppState.chat.mode} (from dropdown)`);
        
        modeSelect.addEventListener('change', (e) => {
            AppState.chat.mode = e.target.value;
            console.log(`[MODE CHANGED] Now using: ${AppState.chat.mode}`);
            updateModeDescription();
        });
    } else {
        console.warn('[MODE] mode-select element NOT FOUND!');
    }
}

function updateModeDescription() {
    const desc = $('#mode-description');
    if (!desc) return;
    
    if (AppState.chat.mode === 'conversation') {
        desc.textContent = 'Simple Q&A without tools (Phase 3a)';
    } else {
        desc.textContent = 'Uses workspace tools to search documents';
    }
}

async function sendMessage() {
    const chatInput = $('#chat-input');
    const message = chatInput.value.trim();
    
    if (!message || AppState.chat.isLoading) return;
    
    // Clear input
    chatInput.value = '';
    
    // Add user message
    AppState.chat.messages.push({
        role: 'user',
        content: message,
        timestamp: new Date().toISOString()
    });
    renderMessages();
    
    // Show loading
    AppState.chat.isLoading = true;
    updateLoadingState();
    
    try {
        const requestBody = {
            message: message,
            model: AppState.chat.selectedModel,
            mode: AppState.chat.mode,
            verbose: AppState.chat.verbose
        };
        console.log('[SEND] Request:', requestBody);
        
        const response = await fetch('/api/chat/send', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestBody)
        });
        
        const data = await response.json();
        
        // Add tool calls if any
        if (data.tool_calls && data.tool_calls.length > 0) {
            for (const tc of data.tool_calls) {
                AppState.chat.messages.push({
                    role: 'tool_call',
                    content: JSON.stringify({ name: tc.name, arguments: tc.arguments }, null, 2),
                    timestamp: new Date().toISOString()
                });
                AppState.chat.messages.push({
                    role: 'tool_result',
                    content: JSON.stringify(tc.result, null, 2),
                    timestamp: new Date().toISOString()
                });
            }
        }
        
        // Add assistant response
        AppState.chat.messages.push({
            role: 'assistant',
            content: data.content,
            timestamp: new Date().toISOString()
        });
        
    } catch (error) {
        AppState.chat.messages.push({
            role: 'system',
            content: `Error: ${error.message}`,
            timestamp: new Date().toISOString()
        });
    } finally {
        AppState.chat.isLoading = false;
        updateLoadingState();
        renderMessages();
        scrollToBottom();
    }
}

async function clearChat() {
    AppState.chat.messages = [];
    renderMessages();
    await fetch('/api/chat/clear', { method: 'POST' });
}

async function rebuildIndex() {
    const rebuildBtn = $('#rebuild-index-btn');
    if (rebuildBtn) {
        rebuildBtn.disabled = true;
        rebuildBtn.innerHTML = '<span class="loading-spinner" style="width:16px;height:16px;"></span> Building...';
    }
    
    try {
        const response = await fetch('/api/workspace/rebuild-index', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ docs_dir: 'workspace/docs' })
        });
        const data = await response.json();
        
        if (data.success) {
            await loadWorkspaceInfo();
            updateSystemStatus();  // Update header status
            AppState.chat.messages.push({
                role: 'system',
                content: `Index rebuilt: ${data.num_chunks} chunks indexed`,
                timestamp: new Date().toISOString()
            });
            renderMessages();
        }
    } catch (error) {
        console.error('Failed to rebuild index:', error);
    } finally {
        if (rebuildBtn) {
            rebuildBtn.disabled = false;
            rebuildBtn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M23 4v6h-6"/><path d="M1 20v-6h6"/>
                <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/>
            </svg> Rebuild`;
        }
    }
}

function renderChatUI() {
    renderModelSelector();
    renderWorkspaceInfo();
    renderMessages();
}

function renderModelSelector() {
    const select = $('#model-select');
    if (!select) return;
    
    select.innerHTML = '';
    
    if (AppState.chat.models.length === 0) {
        select.innerHTML = '<option disabled>No models available</option>';
    } else {
        AppState.chat.models.forEach(model => {
            const option = document.createElement('option');
            option.value = model;
            option.textContent = model;
            if (model === AppState.chat.selectedModel) {
                option.selected = true;
            }
            select.appendChild(option);
        });
    }
}

function renderWorkspaceInfo() {
    console.log('[DEBUG] renderWorkspaceInfo() called');
    
    const docsCount = $('#docs-count');
    const chunksCount = $('#chunks-count');
    const docsList = $('#documents-list');
    const statusText = $('#index-status-text');
    const statusDot = $('#status-dot');
    const toggleBtn = $('#toggle-files-btn');
    
    console.log('[DEBUG] Elements found:', {
        docsCount: !!docsCount,
        chunksCount: !!chunksCount,
        statusText: !!statusText,
        statusDot: !!statusDot
    });
    
    const numDocs = AppState.chat.documents.length;
    const numChunks = AppState.chat.indexStatus.chunks || 0;
    const hasIndex = numChunks > 0;
    
    console.log('[DEBUG] Values:', { numDocs, numChunks, hasIndex });
    
    // Update status display
    if (docsCount) {
        docsCount.textContent = numDocs.toString();
    }
    
    if (chunksCount) {
        chunksCount.textContent = numChunks.toString();
    }
    
    // Update status indicator
    if (statusText) {
        if (hasIndex) {
            statusText.textContent = 'Ready';
        } else if (numDocs > 0) {
            statusText.textContent = 'Not indexed';
        } else {
            statusText.textContent = 'Empty';
        }
    }
    
    if (statusDot) {
        statusDot.classList.remove('active', 'warning', 'error');
        if (hasIndex) {
            statusDot.classList.add('active');
        } else if (numDocs > 0) {
            statusDot.classList.add('warning');
        }
    }
    
    // Setup toggle button for file list
    if (toggleBtn && !toggleBtn._hasListener) {
        toggleBtn._hasListener = true;
        toggleBtn.addEventListener('click', () => {
            const list = $('#documents-list');
            const isExpanded = list.classList.contains('expanded');
            
            if (isExpanded) {
                list.classList.remove('expanded');
                list.classList.add('collapsed');
                toggleBtn.classList.remove('expanded');
            } else {
                list.classList.remove('collapsed');
                list.classList.add('expanded');
                toggleBtn.classList.add('expanded');
            }
        });
    }
    
    // Update file list (informational, not clickable)
    if (docsList) {
        if (numDocs === 0) {
            docsList.innerHTML = `
                <div class="text-muted text-sm">
                    No files in workspace/docs/
                </div>`;
        } else {
            docsList.innerHTML = AppState.chat.documents.map(doc => `
                <div class="file-item">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                        <path d="M14 2v6h6"/>
                    </svg>
                    <span class="file-name">${escapeHtml(doc.title || doc.path || 'Unknown')}</span>
                </div>
            `).join('');
        }
    }
}

function renderMessages() {
    const container = $('#chat-messages');
    if (!container) return;
    
    if (AppState.chat.messages.length === 0) {
        container.innerHTML = `
            <div class="message system">
                <strong>Welcome to MyPT Workspace Assistant</strong><br>
                I can help you search and analyze documents in your workspace.
                Try asking questions about your indexed documents.
            </div>`;
        return;
    }
    
    container.innerHTML = AppState.chat.messages.map(msg => {
        let content = '';
        
        if (msg.role === 'tool_call') {
            content = `
                <div style="color: var(--accent-warning); margin-bottom: 0.25rem;">
                    🔧 Tool Call
                </div>
                <pre style="margin: 0; white-space: pre-wrap;">${escapeHtml(msg.content)}</pre>`;
        } else if (msg.role === 'tool_result') {
            content = `
                <div style="color: var(--accent-success); margin-bottom: 0.25rem;">
                    📋 Result
                </div>
                <pre style="margin: 0; white-space: pre-wrap; max-height: 200px; overflow-y: auto;">${escapeHtml(msg.content)}</pre>`;
        } else {
            content = `
                <div style="white-space: pre-wrap;">${escapeHtml(msg.content)}</div>
                <div class="message-meta">${formatTime(msg.timestamp)}</div>`;
        }
        
        return `<div class="message ${msg.role}">${content}</div>`;
    }).join('');
    
    scrollToBottom();
}

function updateLoadingState() {
    const sendBtn = $('#send-btn');
    const loadingIndicator = $('#loading-indicator');
    
    if (sendBtn) {
        sendBtn.disabled = AppState.chat.isLoading;
    }
    
    if (loadingIndicator) {
        loadingIndicator.style.display = AppState.chat.isLoading ? 'flex' : 'none';
    }
}

function scrollToBottom() {
    const container = $('#chat-messages');
    if (container) {
        container.scrollTop = container.scrollHeight;
    }
}

// ============================================================================
// Training Page Functions
// ============================================================================

async function initTrainingPage() {
    console.log('Initializing training page...');
    
    // Load available models for fine-tuning
    await loadAvailableModels();
    
    // Check current training status
    await loadTrainingStatus();
    
    // Setup event listeners
    setupTrainingEventListeners();
    
    // Render initial state
    renderTrainingUI();
}

async function loadAvailableModels() {
    try {
        const response = await fetch('/api/chat/models');
        const data = await response.json();
        renderBaseModelSelector(data.models || []);
    } catch (error) {
        console.error('Failed to load models:', error);
    }
}

async function loadTrainingStatus() {
    try {
        const response = await fetch('/api/training/status');
        const data = await response.json();
        AppState.training.isTraining = data.is_training || false;
        if (data.progress) {
            AppState.training.progress = data.progress;
        }
        if (data.logs) {
            AppState.training.logs = data.logs;
        }
        renderTrainingProgress();
        renderLogs();
    } catch (error) {
        console.error('Failed to load training status:', error);
    }
}

function setupTrainingEventListeners() {
    // Training mode selection
    $$('.training-mode-option').forEach(option => {
        option.addEventListener('click', () => {
            const mode = option.dataset.mode;
            selectTrainingMode(mode);
        });
    });
    
    // Start/Stop buttons
    const startBtn = $('#start-training-btn');
    if (startBtn) {
        startBtn.addEventListener('click', startTraining);
    }
    
    const stopBtn = $('#stop-training-btn');
    if (stopBtn) {
        stopBtn.addEventListener('click', stopTraining);
    }
    
    // Config inputs
    const configInputs = ['modelSize', 'baseModel', 'datasetDir', 'outputName', 
                          'maxIters', 'evalInterval', 'learningRate', 'batchSize', 'warmupIters'];
    configInputs.forEach(name => {
        const input = $(`#config-${name}`);
        if (input) {
            input.addEventListener('change', (e) => {
                AppState.training.config[name] = e.target.value;
            });
        }
    });
}

function selectTrainingMode(mode) {
    AppState.training.mode = mode;
    
    // Update UI
    $$('.training-mode-option').forEach(opt => {
        opt.classList.toggle('selected', opt.dataset.mode === mode);
    });
    
    // Update learning rate default
    const lrInput = $('#config-learningRate');
    if (lrInput) {
        lrInput.value = mode === 'pretrain' ? '3e-4' : '3e-5';
        AppState.training.config.learningRate = lrInput.value;
    }
    
    // Show/hide base model selector
    const baseModelGroup = $('#base-model-group');
    if (baseModelGroup) {
        baseModelGroup.style.display = mode === 'pretrain' ? 'none' : 'block';
    }
}

async function startTraining() {
    if (AppState.training.isTraining) return;
    
    const config = AppState.training.config;
    
    // Validate
    if (!config.outputName) {
        addLog('error', 'Please specify an output model name');
        return;
    }
    
    AppState.training.isTraining = true;
    AppState.training.progress = { step: 0, maxSteps: parseInt(config.maxIters), trainLoss: null, valLoss: null, eta: null };
    AppState.training.logs = [];
    
    renderTrainingControls();
    addLog('info', `Starting ${AppState.training.mode} training...`);
    
    try {
        const response = await fetch('/api/training/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                mode: AppState.training.mode,
                ...config
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            addLog('success', 'Training started');
            connectTrainingWebSocket();
        } else {
            addLog('error', data.error || 'Failed to start training');
            AppState.training.isTraining = false;
            renderTrainingControls();
        }
    } catch (error) {
        addLog('error', `Error: ${error.message}`);
        AppState.training.isTraining = false;
        renderTrainingControls();
    }
}

async function stopTraining() {
    try {
        await fetch('/api/training/stop', { method: 'POST' });
        addLog('warning', 'Training stopped by user');
        AppState.training.isTraining = false;
        renderTrainingControls();
    } catch (error) {
        console.error('Failed to stop training:', error);
    }
}

let trainingWs = null;

function connectTrainingWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    trainingWs = new WebSocket(`${protocol}//${window.location.host}/api/training/ws`);
    
    trainingWs.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        if (data.type === 'progress') {
            AppState.training.progress.step = data.step;
            AppState.training.progress.trainLoss = data.train_loss;
            AppState.training.progress.valLoss = data.val_loss;
            AppState.training.progress.eta = data.eta;
            renderTrainingProgress();
        }
        
        if (data.type === 'log') {
            addLog(data.level, data.message);
        }
        
        if (data.type === 'complete') {
            AppState.training.isTraining = false;
            addLog('success', 'Training complete!');
            renderTrainingControls();
            loadAvailableModels();
        }
        
        if (data.type === 'error') {
            AppState.training.isTraining = false;
            addLog('error', data.message);
            renderTrainingControls();
        }
    };
    
    trainingWs.onclose = () => {
        if (AppState.training.isTraining) {
            setTimeout(connectTrainingWebSocket, 2000);
        }
    };
}

function addLog(level, message) {
    AppState.training.logs.push({ level, message, timestamp: new Date().toISOString() });
    if (AppState.training.logs.length > 100) {
        AppState.training.logs.shift();
    }
    renderLogs();
}

function renderTrainingUI() {
    renderTrainingControls();
    renderTrainingProgress();
    renderLogs();
}

function renderBaseModelSelector(models) {
    const select = $('#config-baseModel');
    if (!select) return;
    
    select.innerHTML = '<option value="">None (start fresh)</option>';
    models.forEach(model => {
        const option = document.createElement('option');
        option.value = model;
        option.textContent = model;
        select.appendChild(option);
    });
}

function renderTrainingControls() {
    const startBtn = $('#start-training-btn');
    const stopBtn = $('#stop-training-btn');
    
    if (startBtn && stopBtn) {
        startBtn.style.display = AppState.training.isTraining ? 'none' : 'inline-flex';
        stopBtn.style.display = AppState.training.isTraining ? 'inline-flex' : 'none';
    }
}

function renderTrainingProgress() {
    const { step, maxSteps, trainLoss, valLoss, eta } = AppState.training.progress;
    const percent = maxSteps > 0 ? Math.round((step / maxSteps) * 100) : 0;
    
    const stepDisplay = $('#progress-step');
    const maxStepsDisplay = $('#progress-max-steps');
    const percentDisplay = $('#progress-percent');
    const progressFill = $('#progress-fill');
    const trainLossDisplay = $('#stat-train-loss');
    const valLossDisplay = $('#stat-val-loss');
    const etaDisplay = $('#stat-eta');
    
    if (stepDisplay) stepDisplay.textContent = step;
    if (maxStepsDisplay) maxStepsDisplay.textContent = maxSteps;
    if (percentDisplay) percentDisplay.textContent = percent;
    if (progressFill) progressFill.style.width = `${percent}%`;
    if (trainLossDisplay) trainLossDisplay.textContent = trainLoss !== null ? trainLoss.toFixed(4) : '—';
    if (valLossDisplay) valLossDisplay.textContent = valLoss !== null ? valLoss.toFixed(4) : '—';
    if (etaDisplay) etaDisplay.textContent = eta || '—';
}

function renderLogs() {
    const logOutput = $('#log-output');
    if (!logOutput) return;
    
    if (AppState.training.logs.length === 0) {
        logOutput.innerHTML = '<span class="text-muted">No logs yet. Start training to see output.</span>';
    } else {
        logOutput.innerHTML = AppState.training.logs.map(log => 
            `<div class="log-${log.level}">${escapeHtml(log.message)}</div>`
        ).join('');
    }
    
    logOutput.scrollTop = logOutput.scrollHeight;
}

// ============================================================================
// Page Initialization
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    console.log('MyPT Web Application initialized (v2026-01-25-v2)');
    
    // Detect which page we're on and initialize
    const path = window.location.pathname;
    
    if (path === '/' || path === '/chat') {
        initChatPage();
    } else if (path === '/training') {
        initTrainingPage();
    }
});
