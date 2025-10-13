/**
 * APIClient - HTTP API communication with retry logic
 *
 * Centralized API client for all dashboard endpoints with automatic
 * retry, error handling, and response caching (5-minute TTL).
 *
 * @module APIClient
 */

export class APIClient {
    /**
     * Create an APIClient instance
     * @param {Object} errorHandler - Error handler instance
     */
    constructor(errorHandler) {
        this.errorHandler = errorHandler;
        this.baseUrl = '';
        this.defaultRetries = 3;
        this.retryDelay = 1000; // Base delay in ms
        this.timeout = 30000; // 30 second timeout
        this.cache = new Map(); // Response cache
        this.cacheTTL = 5 * 60 * 1000; // 5 minutes
    }

    /**
     * Fetch metrics summary
     * Endpoint: GET /api/metrics/summary
     *
     * @returns {Promise<Object>} - MetricsSummary object
     */
    async fetchMetricsSummary() {
        return this.fetchWithRetry('/api/metrics/summary');
    }

    /**
     * Fetch reuse metrics
     * Endpoint: GET /api/metrics/reuse
     *
     * @returns {Promise<Object>} - ReuseMetrics object
     */
    async fetchReuseMetrics() {
        return this.fetchWithRetry('/api/metrics/reuse');
    }

    /**
     * Fetch quality metrics
     * Endpoint: GET /api/metrics/quality
     *
     * @returns {Promise<Object>} - QualityMetrics object
     */
    async fetchQualityMetrics() {
        return this.fetchWithRetry('/api/metrics/quality');
    }

    /**
     * Fetch trends data
     * Endpoint: GET /api/metrics/trends
     *
     * @returns {Promise<Object>} - Trends object
     */
    async fetchTrends() {
        return this.fetchWithRetry('/api/metrics/trends');
    }

    /**
     * Fetch gate status
     * Endpoint: GET /api/gates/status
     *
     * @returns {Promise<Object>} - GateStatus object
     */
    async fetchGateStatus() {
        return this.fetchWithRetry('/api/gates/status');
    }

    /**
     * Fetch specific phase details
     * Endpoint: GET /api/gates/<phase>?task=<name>
     *
     * @param {string} phase - Phase name
     * @param {string} [taskName] - Optional task name
     * @returns {Promise<Object>} - PhaseDetails object
     */
    async fetchPhaseDetails(phase, taskName = null) {
        const url = taskName
            ? `/api/gates/${phase}?task=${encodeURIComponent(taskName)}`
            : `/api/gates/${phase}`;

        return this.fetchWithRetry(url);
    }

    /**
     * Fetch reuse checks
     * Endpoint: GET /api/reuse/checks?limit=50
     *
     * @param {number} [limit=50] - Maximum number of checks to return
     * @returns {Promise<Array>} - Array of ReuseCheck objects
     */
    async fetchReuseChecks(limit = 50) {
        return this.fetchWithRetry(`/api/reuse/checks?limit=${limit}`);
    }

    /**
     * Fetch reuse violations
     * Endpoint: GET /api/reuse/violations
     *
     * @returns {Promise<Array>} - Array of Violation objects
     */
    async fetchReuseViolations() {
        return this.fetchWithRetry('/api/reuse/violations');
    }

    /**
     * Fetch with automatic retry logic and caching
     *
     * @private
     * @param {string} url - API endpoint URL
     * @param {number} [retries=3] - Number of retry attempts
     * @returns {Promise<Object>} - Parsed JSON response
     * @throws {APIError} - On non-retriable errors or max retries exceeded
     */
    async fetchWithRetry(url, retries = this.defaultRetries) {
        // Check cache first
        const cached = this.getFromCache(url);
        if (cached) {
            console.log(`[APIClient] Cache hit: ${url}`);
            return cached;
        }

        let lastError = null;

        for (let attempt = 0; attempt <= retries; attempt++) {
            try {
                const response = await this.fetchWithTimeout(url, this.timeout);

                if (!response.ok) {
                    const error = new APIError(
                        response.status,
                        response.statusText,
                        url
                    );

                    // Check if error is retriable
                    if (!this.isRetriable(error) || attempt === retries) {
                        throw error;
                    }

                    lastError = error;
                    await this.sleep(this.retryDelay * Math.pow(2, attempt));
                    continue;
                }

                const data = await response.json();

                // Cache successful response
                this.addToCache(url, data);

                return data;

            } catch (error) {
                if (error instanceof APIError) {
                    lastError = error;

                    if (!this.isRetriable(error) || attempt === retries) {
                        // Handle error through error handler
                        if (this.errorHandler) {
                            this.errorHandler.handleApiError(error);
                        }
                        throw error;
                    }
                } else {
                    // Network error or timeout
                    lastError = error;

                    if (attempt === retries) {
                        if (this.errorHandler) {
                            this.errorHandler.handleApiError(error);
                        }
                        throw error;
                    }
                }

                // Exponential backoff
                await this.sleep(this.retryDelay * Math.pow(2, attempt));
            }
        }

        throw lastError;
    }

    /**
     * Fetch with timeout
     *
     * @private
     * @param {string} url - URL to fetch
     * @param {number} timeout - Timeout in milliseconds
     * @returns {Promise<Response>} - Fetch response
     */
    async fetchWithTimeout(url, timeout) {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), timeout);

        try {
            const response = await fetch(url, { signal: controller.signal });
            clearTimeout(timeoutId);
            return response;
        } catch (error) {
            clearTimeout(timeoutId);
            throw error;
        }
    }

    /**
     * Check if error is retriable
     *
     * @private
     * @param {APIError|Error} error - Error to check
     * @returns {boolean} - True if error should be retried
     */
    isRetriable(error) {
        if (!(error instanceof APIError)) {
            // Network errors are retriable
            return true;
        }

        // Retriable HTTP status codes
        const retriableStatuses = [408, 429, 500, 502, 503, 504];
        return retriableStatuses.includes(error.status);
    }

    /**
     * Sleep for specified milliseconds
     *
     * @private
     * @param {number} ms - Milliseconds to sleep
     * @returns {Promise<void>}
     */
    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    /**
     * Get cached response
     *
     * @private
     * @param {string} url - Cache key (URL)
     * @returns {Object|null} - Cached data or null
     */
    getFromCache(url) {
        const cached = this.cache.get(url);

        if (!cached) return null;

        const now = Date.now();
        if (now - cached.timestamp > this.cacheTTL) {
            // Cache expired
            this.cache.delete(url);
            return null;
        }

        return cached.data;
    }

    /**
     * Add response to cache
     *
     * @private
     * @param {string} url - Cache key (URL)
     * @param {Object} data - Data to cache
     */
    addToCache(url, data) {
        this.cache.set(url, {
            data: data,
            timestamp: Date.now()
        });
    }

    /**
     * Clear cache
     */
    clearCache() {
        this.cache.clear();
        console.log('[APIClient] Cache cleared');
    }

    /**
     * Clear expired cache entries
     */
    clearExpiredCache() {
        const now = Date.now();
        let cleared = 0;

        for (const [url, cached] of this.cache.entries()) {
            if (now - cached.timestamp > this.cacheTTL) {
                this.cache.delete(url);
                cleared++;
            }
        }

        if (cleared > 0) {
            console.log(`[APIClient] Cleared ${cleared} expired cache entries`);
        }
    }
}

/**
 * Custom API Error class
 */
export class APIError extends Error {
    /**
     * Create an APIError
     *
     * @param {number} status - HTTP status code
     * @param {string} statusText - HTTP status text
     * @param {string} url - Request URL
     */
    constructor(status, statusText, url) {
        super(`API Error ${status}: ${statusText}`);
        this.name = 'APIError';
        this.status = status;
        this.statusText = statusText;
        this.url = url;
    }
}
