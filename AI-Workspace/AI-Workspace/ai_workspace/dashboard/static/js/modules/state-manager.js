/**
 * StateManager - Observable state container
 *
 * Centralized application state with observer pattern for reactive updates.
 * Single source of truth for dashboard data with state change notifications.
 *
 * @module StateManager
 */

export class StateManager {
    /**
     * Create a StateManager instance
     */
    constructor() {
        this.state = this.getInitialState();
        this.observers = [];
    }

    /**
     * Get initial state shape
     *
     * @private
     * @returns {Object} - Initial state object
     */
    getInitialState() {
        return {
            // Metrics data
            metrics: null,

            // Reuse analysis data
            reuse: null,

            // Phase gate data
            gates: null,

            // Trends and activity data
            trends: null,

            // Project manifest data
            manifest: null,

            // Loading states
            loading: {
                metrics: false,
                reuse: false,
                gates: false,
                trends: false,
                manifest: false
            },

            // Error states
            errors: {},

            // Connection state
            connection: {
                status: 'disconnected',
                lastUpdate: null,
                monitorActive: false
            }
        };
    }

    /**
     * Update state value
     * Notifies all observers of the change
     *
     * @param {string} key - State key to update
     * @param {*} value - New value
     */
    update(key, value) {
        const oldValue = this.state[key];

        // Update state
        this.state[key] = value;

        // Update last update timestamp
        if (key !== 'loading' && key !== 'errors') {
            this.state.connection.lastUpdate = new Date().toISOString();
        }

        // Notify observers
        this.notify(key, value, oldValue);

        console.log(`[StateManager] Updated: ${key}`, value);
    }

    /**
     * Get state value
     *
     * @param {string} key - State key to retrieve
     * @returns {*} - State value
     */
    get(key) {
        return this.state[key];
    }

    /**
     * Get entire state
     *
     * @returns {Object} - Complete state object
     */
    getAll() {
        return { ...this.state };
    }

    /**
     * Set loading state
     *
     * @param {string} key - Loading key (metrics, reuse, gates, trends, manifest)
     * @param {boolean} loading - Loading status
     */
    setLoading(key, loading) {
        this.state.loading[key] = loading;
        this.notify('loading', this.state.loading);
    }

    /**
     * Set error state
     *
     * @param {string} key - Error key
     * @param {Error|null} error - Error object or null to clear
     */
    setError(key, error) {
        if (error) {
            this.state.errors[key] = {
                message: error.message,
                timestamp: new Date().toISOString(),
                error: error
            };
        } else {
            delete this.state.errors[key];
        }

        this.notify('errors', this.state.errors);
    }

    /**
     * Update connection state
     *
     * @param {string} status - Connection status ('connected', 'disconnected', 'reconnecting')
     * @param {boolean} [monitorActive=false] - Monitor running status
     */
    updateConnection(status, monitorActive = false) {
        this.state.connection.status = status;
        this.state.connection.monitorActive = monitorActive;
        this.state.connection.lastUpdate = new Date().toISOString();

        this.notify('connection', this.state.connection);
    }

    /**
     * Subscribe to state changes
     * Observer pattern implementation
     *
     * @param {Function} observer - Observer callback function(key, value, oldValue)
     * @returns {Function} - Unsubscribe function
     */
    subscribe(observer) {
        if (typeof observer !== 'function') {
            throw new TypeError('Observer must be a function');
        }

        this.observers.push(observer);

        console.log('[StateManager] Observer subscribed. Total:', this.observers.length);

        // Return unsubscribe function
        return () => {
            const index = this.observers.indexOf(observer);
            if (index > -1) {
                this.observers.splice(index, 1);
                console.log('[StateManager] Observer unsubscribed. Total:', this.observers.length);
            }
        };
    }

    /**
     * Notify all observers of state change
     *
     * @private
     * @param {string} key - Changed state key
     * @param {*} value - New value
     * @param {*} [oldValue] - Previous value
     */
    notify(key, value, oldValue = null) {
        this.observers.forEach(observer => {
            try {
                observer(key, value, oldValue);
            } catch (error) {
                console.error('[StateManager] Observer error:', error);
            }
        });
    }

    /**
     * Reset state to initial values
     */
    reset() {
        const oldState = this.state;
        this.state = this.getInitialState();

        console.log('[StateManager] State reset');

        // Notify observers of reset
        this.notify('reset', this.state, oldState);
    }

    /**
     * Clear all errors
     */
    clearErrors() {
        this.state.errors = {};
        this.notify('errors', this.state.errors);
    }

    /**
     * Check if data is loaded
     *
     * @param {string} key - Data key to check
     * @returns {boolean} - True if data exists
     */
    hasData(key) {
        return this.state[key] !== null && this.state[key] !== undefined;
    }

    /**
     * Check if currently loading
     *
     * @param {string} key - Loading key to check
     * @returns {boolean} - True if loading
     */
    isLoading(key) {
        return this.state.loading[key] === true;
    }

    /**
     * Check if error exists
     *
     * @param {string} key - Error key to check
     * @returns {boolean} - True if error exists
     */
    hasError(key) {
        return !!this.state.errors[key];
    }

    /**
     * Get error for key
     *
     * @param {string} key - Error key
     * @returns {Object|null} - Error object or null
     */
    getError(key) {
        return this.state.errors[key] || null;
    }

    /**
     * Merge data into existing state
     * Useful for partial updates from WebSocket
     *
     * @param {string} key - State key
     * @param {Object} data - Data to merge
     */
    merge(key, data) {
        const current = this.state[key];

        if (!current || typeof current !== 'object') {
            // Can't merge, just update
            this.update(key, data);
            return;
        }

        const merged = {
            ...current,
            ...data
        };

        this.update(key, merged);
    }

    /**
     * Get state summary for debugging
     *
     * @returns {Object} - State summary
     */
    getSummary() {
        return {
            hasMetrics: this.hasData('metrics'),
            hasReuse: this.hasData('reuse'),
            hasGates: this.hasData('gates'),
            hasTrends: this.hasData('trends'),
            hasManifest: this.hasData('manifest'),
            loadingCount: Object.values(this.state.loading).filter(Boolean).length,
            errorCount: Object.keys(this.state.errors).length,
            connectionStatus: this.state.connection.status,
            lastUpdate: this.state.connection.lastUpdate,
            observers: this.observers.length
        };
    }

    /**
     * Log state summary to console
     */
    logSummary() {
        console.table(this.getSummary());
    }
}
