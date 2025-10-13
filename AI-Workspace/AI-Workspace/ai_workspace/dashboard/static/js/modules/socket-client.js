/**
 * SocketClient - WebSocket communication and real-time updates
 *
 * Handles Socket.IO connection, event subscriptions, and reconnection logic
 * with exponential backoff strategy.
 *
 * @module SocketClient
 */

export class SocketClient {
    /**
     * Create a SocketClient instance
     * @param {Function} updateCallback - Callback function for workspace updates
     */
    constructor(updateCallback) {
        this.socket = io();
        this.updateCallback = updateCallback;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.setupEventHandlers();
    }

    /**
     * Setup all Socket.IO event handlers
     * @private
     */
    setupEventHandlers() {
        // Connection events
        this.socket.on('connect', () => this.handleConnect());
        this.socket.on('disconnect', () => this.handleDisconnect());
        this.socket.on('reconnect_attempt', () => this.handleReconnectAttempt());
        this.socket.on('reconnect_failed', () => this.handleReconnectFailed());

        // Data update events (from handoff 013)
        this.socket.on('reuse_check_update', (payload) => this.handleReuseCheckUpdate(payload));
        this.socket.on('gate_update', (payload) => this.handleGateUpdate(payload));
        this.socket.on('metrics_update', (payload) => this.handleMetricsUpdate(payload));
        this.socket.on('manifest_update', (payload) => this.handleManifestUpdate(payload));
        this.socket.on('workspace_health', (payload) => this.handleWorkspaceHealth(payload));
    }

    /**
     * Handle successful connection
     * @private
     */
    handleConnect() {
        console.log('[SocketClient] Connected to workspace monitor');
        this.reconnectAttempts = 0;
        this.updateConnectionStatus(true, false);

        // Trigger full data refresh on reconnection
        if (this.updateCallback) {
            this.updateCallback('reconnect', { refresh: true });
        }
    }

    /**
     * Handle disconnection
     * @private
     */
    handleDisconnect() {
        console.log('[SocketClient] Disconnected from workspace monitor');
        this.updateConnectionStatus(false, false);
    }

    /**
     * Handle reconnection attempt
     * @private
     */
    handleReconnectAttempt() {
        this.reconnectAttempts++;
        console.log(`[SocketClient] Reconnection attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts}`);
        this.updateConnectionStatus(false, false);
    }

    /**
     * Handle reconnection failure
     * @private
     */
    handleReconnectFailed() {
        console.error('[SocketClient] Failed to reconnect after maximum attempts');
        this.updateConnectionStatus(false, true);

        // Notify user of connection failure
        if (this.updateCallback) {
            this.updateCallback('connection_failed', {
                message: 'Unable to connect to workspace monitor. Please refresh the page.'
            });
        }
    }

    /**
     * Handle reuse check updates
     * Event from handoff 013: reuse_check_update
     *
     * @private
     * @param {Object} payload - Reuse check update payload
     * @param {Array} payload.data - Array of ReuseAnalysisDict objects
     * @param {number} payload.total_checks - Total number of checks
     * @param {string} payload.timestamp - Update timestamp
     */
    handleReuseCheckUpdate(payload) {
        console.log('[SocketClient] Reuse check update:', payload.total_checks, 'checks');

        if (this.updateCallback) {
            this.updateCallback('reuse_check', payload);
        }
    }

    /**
     * Handle gate updates
     * Event from handoff 013: gate_update
     *
     * @private
     * @param {Object} payload - Gate update payload
     * @param {Object} payload.data - PhaseGateDict object
     * @param {string} payload.gate_name - Gate identifier
     * @param {string} payload.phase - Current phase
     * @param {string} payload.status - Gate status
     * @param {number} payload.completion - Completion percentage
     */
    handleGateUpdate(payload) {
        console.log('[SocketClient] Gate update:', payload.gate_name, `${payload.completion}% complete`);

        if (this.updateCallback) {
            this.updateCallback('gate', payload);
        }
    }

    /**
     * Handle metrics updates
     * Event from handoff 013: metrics_update
     *
     * @private
     * @param {Object} payload - Metrics update payload
     * @param {Object} payload.data - MetricsSummaryDict object
     * @param {number} payload.quality_score - Overall quality score
     */
    handleMetricsUpdate(payload) {
        console.log('[SocketClient] Metrics update: quality score', payload.quality_score);

        if (this.updateCallback) {
            this.updateCallback('metrics', payload);
        }
    }

    /**
     * Handle manifest updates
     * Event from handoff 013: manifest_update
     *
     * @private
     * @param {Object} payload - Manifest update payload
     * @param {Object} payload.data - ProjectManifestDict object
     * @param {string} payload.project - Project name
     * @param {string} payload.current_phase - Current phase
     */
    handleManifestUpdate(payload) {
        console.log('[SocketClient] Manifest update:', payload.project, 'phase:', payload.current_phase);

        if (this.updateCallback) {
            this.updateCallback('manifest', payload);
        }
    }

    /**
     * Handle workspace health checks
     * Event from handoff 013: workspace_health (30s heartbeat)
     *
     * @private
     * @param {Object} payload - Health status payload
     * @param {string} payload.status - 'connected' | 'disconnected'
     * @param {boolean} payload.monitor_active - Monitor running status
     */
    handleWorkspaceHealth(payload) {
        console.log('[SocketClient] Workspace health:', payload.status, 'monitor:', payload.monitor_active);
        this.updateConnectionStatus(payload.status === 'connected', false, payload.monitor_active);
    }

    /**
     * Update connection status indicator in UI
     *
     * @private
     * @param {boolean} connected - Connection status
     * @param {boolean} failed - Reconnection failed
     * @param {boolean} [monitorActive=true] - Monitor running status
     */
    updateConnectionStatus(connected, failed, monitorActive = true) {
        const statusEl = document.getElementById('connection-status');

        if (!statusEl) return;

        if (failed) {
            // Reconnection failed
            statusEl.innerHTML = `
                <span class="inline-block w-2 h-2 bg-red-400 rounded-full mr-1"></span>
                <span class="text-red-200">Connection Failed</span>
            `;
        } else if (connected && monitorActive) {
            // Connected and monitor active
            statusEl.innerHTML = `
                <span class="inline-block w-2 h-2 bg-green-400 rounded-full mr-1 animate-pulse"></span>
                <span>Connected</span>
            `;
        } else if (connected && !monitorActive) {
            // Connected but monitor inactive
            statusEl.innerHTML = `
                <span class="inline-block w-2 h-2 bg-yellow-400 rounded-full mr-1"></span>
                <span class="text-yellow-200">Monitor Paused</span>
            `;
        } else {
            // Disconnected/reconnecting
            statusEl.innerHTML = `
                <span class="inline-block w-2 h-2 bg-yellow-400 rounded-full mr-1"></span>
                <span class="text-yellow-200">Reconnecting...</span>
            `;
        }
    }

    /**
     * Subscribe to custom event
     *
     * @param {string} event - Event name
     * @param {Function} handler - Event handler function
     */
    on(event, handler) {
        this.socket.on(event, handler);
    }

    /**
     * Disconnect from WebSocket
     */
    disconnect() {
        console.log('[SocketClient] Disconnecting...');
        this.socket.disconnect();
    }

    /**
     * Get connection status
     *
     * @returns {boolean} - True if connected
     */
    isConnected() {
        return this.socket.connected;
    }
}
