/**
 * DashboardApp - Main application initialization and coordination
 *
 * Orchestrates all modules and handles application lifecycle.
 * Entry point for the AI Workspace Dashboard.
 *
 * @module DashboardApp
 */

import { APIClient } from './api-client.js';
import { SocketClient } from './socket-client.js';
import { ChartManager } from './chart-manager.js';
import { StateManager } from './state-manager.js';
import { UIController } from './ui-controller.js';
import { initializeSearch } from './search.js';

/**
 * Main Dashboard Application Class
 */
class DashboardApp {
    /**
     * Create DashboardApp instance
     */
    constructor() {
        this.state = null;
        this.api = null;
        this.socket = null;
        this.charts = null;
        this.ui = null;
        this.search = null;
        this.initialized = false;
    }

    /**
     * Initialize dashboard application
     * Sets up all modules and loads initial data
     */
    async init() {
        try {
            console.log('[DashboardApp] Initializing...');

            // Performance mark
            performance.mark('dashboard-init-start');

            // Initialize modules
            this.initializeModules();

            // Setup error handlers
            this.setupErrorHandlers();

            // Load initial data
            await this.loadInitialData();

            // Initialize charts
            this.initializeCharts();

            // Performance measurement
            performance.mark('dashboard-init-end');
            performance.measure('dashboard-init', 'dashboard-init-start', 'dashboard-init-end');

            const initTime = performance.getEntriesByName('dashboard-init')[0].duration;
            console.log(`[DashboardApp] Initialized in ${initTime.toFixed(2)}ms`);

            this.initialized = true;

            // Announce to screen readers
            this.announceToScreenReader('Dashboard loaded successfully');

        } catch (error) {
            console.error('[DashboardApp] Initialization failed:', error);
            this.handleInitializationError(error);
        }
    }

    /**
     * Initialize all application modules
     *
     * @private
     */
    initializeModules() {
        console.log('[DashboardApp] Initializing modules...');

        // State manager (first - others depend on it)
        this.state = new StateManager();

        // API client
        this.api = new APIClient(this);

        // Chart manager
        this.charts = new ChartManager();

        // UI controller
        this.ui = new UIController(this.state);

        // Search manager (Phase 3)
        this.search = initializeSearch(this.api);

        // WebSocket client (last - triggers updates)
        this.socket = new SocketClient((eventType, data) => {
            this.handleRealtimeUpdate(eventType, data);
        });

        console.log('[DashboardApp] Modules initialized');
    }

    /**
     * Load initial dashboard data
     * Parallel API calls for optimal performance
     *
     * @private
     */
    async loadInitialData() {
        console.log('[DashboardApp] Loading initial data...');

        // Set loading states
        this.state.setLoading('metrics', true);
        this.state.setLoading('reuse', true);
        this.state.setLoading('gates', true);
        this.state.setLoading('trends', true);

        try {
            // Parallel API calls for performance
            const [metrics, reuse, gates, trends] = await Promise.all([
                this.api.fetchMetricsSummary().catch(err => {
                    console.error('Failed to fetch metrics:', err);
                    return null;
                }),
                this.api.fetchReuseMetrics().catch(err => {
                    console.error('Failed to fetch reuse metrics:', err);
                    return null;
                }),
                this.api.fetchGateStatus().catch(err => {
                    console.error('Failed to fetch gate status:', err);
                    return null;
                }),
                this.api.fetchTrends().catch(err => {
                    console.error('Failed to fetch trends:', err);
                    return null;
                })
            ]);

            // Update state with fetched data
            if (metrics) this.state.update('metrics', metrics);
            if (reuse) this.state.update('reuse', reuse);
            if (gates) this.state.update('gates', gates);
            if (trends) this.state.update('trends', trends);

            console.log('[DashboardApp] Initial data loaded');

        } catch (error) {
            console.error('[DashboardApp] Error loading initial data:', error);
            throw error;

        } finally {
            // Clear loading states
            this.state.setLoading('metrics', false);
            this.state.setLoading('reuse', false);
            this.state.setLoading('gates', false);
            this.state.setLoading('trends', false);
        }
    }

    /**
     * Initialize charts with current data
     *
     * @private
     */
    initializeCharts() {
        const reuse = this.state.get('reuse');
        const gates = this.state.get('gates');

        // Create reuse distribution chart
        if (reuse && reuse.distribution) {
            this.charts.createReuseChart('reuseChart', reuse.distribution);
        }

        // Create phase gates chart
        if (gates && gates.phase_counts) {
            this.charts.createGatesChart('gatesChart', gates.phase_counts);
        }

        console.log('[DashboardApp] Charts initialized');
    }

    /**
     * Handle real-time updates from WebSocket
     * Routes updates to appropriate handlers
     *
     * @private
     * @param {string} eventType - Event type
     * @param {Object} data - Event payload
     */
    handleRealtimeUpdate(eventType, data) {
        console.log(`[DashboardApp] Real-time update: ${eventType}`, data);

        switch (eventType) {
            case 'reuse_check':
                this.handleReuseCheckUpdate(data);
                break;

            case 'gate':
                this.handleGateUpdate(data);
                break;

            case 'metrics':
                this.handleMetricsUpdate(data);
                break;

            case 'manifest':
                this.handleManifestUpdate(data);
                break;

            case 'reconnect':
                this.handleReconnect();
                break;

            case 'connection_failed':
                this.handleConnectionFailed(data);
                break;

            default:
                console.warn(`[DashboardApp] Unknown event type: ${eventType}`);
        }
    }

    /**
     * Handle reuse check updates
     *
     * @private
     * @param {Object} payload - Reuse check update payload
     */
    async handleReuseCheckUpdate(payload) {
        // Refetch reuse metrics
        try {
            const reuse = await this.api.fetchReuseMetrics();
            this.state.update('reuse', reuse);

            // Update chart
            if (reuse.distribution) {
                this.charts.updateReuseData('reuseChart', reuse.distribution);
            }

            // Show notification
            this.ui.showToast(
                `Reuse checks updated: ${payload.total_checks} checks`,
                'success',
                3000
            );

        } catch (error) {
            console.error('[DashboardApp] Error handling reuse check update:', error);
        }
    }

    /**
     * Handle gate updates
     *
     * @private
     * @param {Object} payload - Gate update payload
     */
    async handleGateUpdate(payload) {
        // Refetch gate status
        try {
            const gates = await this.api.fetchGateStatus();
            this.state.update('gates', gates);

            // Update chart
            if (gates.phase_counts) {
                this.charts.updateGatesData('gatesChart', gates.phase_counts);
            }

            // Show notification
            this.ui.showToast(
                `Gate updated: ${payload.gate_name} (${payload.completion}% complete)`,
                'info',
                3000
            );

        } catch (error) {
            console.error('[DashboardApp] Error handling gate update:', error);
        }
    }

    /**
     * Handle metrics updates
     *
     * @private
     * @param {Object} payload - Metrics update payload
     */
    handleMetricsUpdate(payload) {
        // Update metrics state
        this.state.update('metrics', payload.data);

        console.log('[DashboardApp] Metrics updated');
    }

    /**
     * Handle manifest updates
     *
     * @private
     * @param {Object} payload - Manifest update payload
     */
    handleManifestUpdate(payload) {
        // Update manifest state
        this.state.update('manifest', payload.data);

        // Show notification
        this.ui.showToast(
            `Project updated: ${payload.project}`,
            'info',
            3000
        );
    }

    /**
     * Handle reconnection
     * Refetch all data to ensure consistency
     *
     * @private
     */
    async handleReconnect() {
        console.log('[DashboardApp] Reconnected, resyncing data...');

        this.ui.showToast('Reconnected to workspace monitor', 'success', 3000);

        // Clear cache and refetch
        this.api.clearCache();
        await this.loadInitialData();

        // Refresh charts
        const reuse = this.state.get('reuse');
        const gates = this.state.get('gates');

        if (reuse && reuse.distribution) {
            this.charts.updateReuseData('reuseChart', reuse.distribution);
        }

        if (gates && gates.phase_counts) {
            this.charts.updateGatesData('gatesChart', gates.phase_counts);
        }
    }

    /**
     * Handle connection failure
     *
     * @private
     * @param {Object} data - Failure data
     */
    handleConnectionFailed(data) {
        this.ui.showToast(
            data.message || 'Connection to workspace monitor failed',
            'error',
            10000
        );
    }

    /**
     * Setup global error handlers
     *
     * @private
     */
    setupErrorHandlers() {
        // Global error handler
        window.addEventListener('error', (event) => {
            console.error('[DashboardApp] Global error:', event.error);
            this.handleGlobalError(event.error);
        });

        // Unhandled promise rejection handler
        window.addEventListener('unhandledrejection', (event) => {
            console.error('[DashboardApp] Unhandled rejection:', event.reason);
            this.handleGlobalError(event.reason);
        });
    }

    /**
     * Handle global errors
     *
     * @private
     * @param {Error} error - Error object
     */
    handleGlobalError(error) {
        // Log error details
        console.error('[DashboardApp] Error details:', {
            message: error.message,
            stack: error.stack,
            timestamp: new Date().toISOString()
        });

        // Show user-friendly error message
        if (this.ui) {
            this.ui.showToast(
                'An unexpected error occurred. Please refresh the page.',
                'error',
                10000
            );
        }
    }

    /**
     * Handle API errors (ErrorHandler interface)
     *
     * @param {Error} error - API error
     */
    handleApiError(error) {
        console.error('[DashboardApp] API error:', error);

        if (this.ui) {
            this.ui.showToast(
                error.message || 'Failed to load data from API',
                'error',
                5000
            );
        }
    }

    /**
     * Handle initialization errors
     *
     * @private
     * @param {Error} error - Initialization error
     */
    handleInitializationError(error) {
        const errorContainer = document.createElement('div');
        errorContainer.className = 'error-boundary bg-red-50 border border-red-200 rounded-lg p-6 m-8';
        errorContainer.setAttribute('role', 'alert');
        errorContainer.innerHTML = `
            <h2 class="text-xl font-bold text-red-800 mb-2">Dashboard Initialization Failed</h2>
            <p class="text-red-700 mb-4">${this.escapeHtml(error.message)}</p>
            <button onclick="window.location.reload()" class="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700">
                Reload Dashboard
            </button>
        `;

        // Replace main content with error
        const main = document.querySelector('main');
        if (main) {
            main.innerHTML = '';
            main.appendChild(errorContainer);
        }
    }

    /**
     * Announce message to screen readers
     *
     * @private
     * @param {string} message - Message to announce
     */
    announceToScreenReader(message) {
        const announcer = document.createElement('div');
        announcer.setAttribute('role', 'status');
        announcer.setAttribute('aria-live', 'polite');
        announcer.className = 'sr-only';
        announcer.textContent = message;

        document.body.appendChild(announcer);

        // Remove after announcement
        setTimeout(() => announcer.remove(), 1000);
    }

    /**
     * Escape HTML for safe rendering
     *
     * @private
     * @param {string} unsafe - Unsafe string
     * @returns {string} - Escaped string
     */
    escapeHtml(unsafe) {
        const div = document.createElement('div');
        div.textContent = unsafe;
        return div.innerHTML;
    }

    /**
     * Cleanup and destroy dashboard
     * For SPA navigation or testing
     */
    destroy() {
        console.log('[DashboardApp] Destroying...');

        // Disconnect WebSocket
        if (this.socket) {
            this.socket.disconnect();
        }

        // Destroy charts
        if (this.charts) {
            this.charts.destroyAll();
        }

        // Clear state
        if (this.state) {
            this.state.reset();
        }

        // Clear API cache
        if (this.api) {
            this.api.clearCache();
        }

        this.initialized = false;

        console.log('[DashboardApp] Destroyed');
    }
}

// Create and initialize dashboard on DOM ready
const app = new DashboardApp();

document.addEventListener('DOMContentLoaded', () => {
    app.init();
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (app.initialized) {
        app.destroy();
    }
});

// Expose app to global scope for debugging
window.DashboardApp = app;

export default DashboardApp;
