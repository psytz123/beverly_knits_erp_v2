/**
 * UIController - DOM manipulation and UI updates
 *
 * Handles all UI updates, DOM manipulation, and rendering with
 * cached element references and ARIA live regions for accessibility.
 *
 * @module UIController
 */

export class UIController {
    /**
     * Create a UIController instance
     *
     * @param {StateManager} stateManager - State manager instance
     */
    constructor(stateManager) {
        this.state = stateManager;
        this.elements = this.cacheElements();

        // Subscribe to state changes
        this.state.subscribe((key, value) => this.handleStateChange(key, value));
    }

    /**
     * Cache DOM element references
     * Performance optimization: Query DOM once, reuse references
     *
     * @private
     * @returns {Object} - Cached element references
     */
    cacheElements() {
        return {
            // Metric cards
            avgReuse: document.getElementById('avg-reuse'),
            gatesComplete: document.getElementById('gates-complete'),
            primaryLang: document.getElementById('primary-lang'),
            totalChecks: document.getElementById('total-checks'),

            // Quality bars
            reuseCompliance: document.getElementById('reuse-compliance'),
            reuseBar: document.getElementById('reuse-bar'),
            checkCompliance: document.getElementById('check-compliance'),
            checkBar: document.getElementById('check-bar'),
            gateCompliance: document.getElementById('gate-compliance'),
            gateBar: document.getElementById('gate-bar'),
            overallScore: document.getElementById('overall-score'),

            // Activity
            recentActivity: document.getElementById('recent-activity'),

            // Connection status
            connectionStatus: document.getElementById('connection-status')
        };
    }

    /**
     * Handle state changes
     * Router for state updates to appropriate UI update methods
     *
     * @private
     * @param {string} key - State key that changed
     * @param {*} value - New value
     */
    handleStateChange(key, value) {
        switch (key) {
            case 'metrics':
                if (value) this.updateMetricCards(value);
                break;

            case 'reuse':
                if (value) this.updateReuseMetrics(value);
                break;

            case 'gates':
                if (value) this.updateGateMetrics(value);
                break;

            case 'trends':
                if (value && value.recent_activity) {
                    this.updateRecentActivity(value.recent_activity);
                }
                break;

            case 'loading':
                this.updateLoadingStates(value);
                break;

            case 'errors':
                this.updateErrorStates(value);
                break;

            default:
                // Ignore other state changes
                break;
        }
    }

    /**
     * Update metric cards from summary data
     *
     * @param {Object} metrics - MetricsSummary object
     */
    updateMetricCards(metrics) {
        // Average reuse percentage
        if (this.elements.avgReuse && metrics.reuse) {
            this.setText(
                this.elements.avgReuse,
                `${metrics.reuse.avg_percentage.toFixed(1)}%`
            );
        }

        // Gates completed
        if (this.elements.gatesComplete && metrics.gates) {
            this.setText(
                this.elements.gatesComplete,
                `${metrics.gates.completed}/${metrics.gates.total}`
            );
        }

        // Primary language
        if (this.elements.primaryLang && metrics.project) {
            this.setText(
                this.elements.primaryLang,
                metrics.project.primary_language || 'Unknown'
            );
        }

        // Total checks
        if (this.elements.totalChecks && metrics.reuse) {
            this.setText(
                this.elements.totalChecks,
                metrics.reuse.total_checks || 0
            );
        }

        // Update quality metrics if available
        if (metrics.quality) {
            this.updateQualityMetrics(metrics.quality);
        }

        console.log('[UIController] Updated metric cards');
    }

    /**
     * Update quality compliance bars
     *
     * @param {Object} quality - QualityMetrics object
     */
    updateQualityMetrics(quality) {
        // Reuse compliance
        this.setMetric(
            this.elements.reuseCompliance,
            this.elements.reuseBar,
            quality.reuse_compliance
        );

        // Check before create compliance
        this.setMetric(
            this.elements.checkCompliance,
            this.elements.checkBar,
            quality.check_before_create_compliance
        );

        // Gate compliance
        this.setMetric(
            this.elements.gateCompliance,
            this.elements.gateBar,
            quality.gate_compliance
        );

        // Overall score
        if (this.elements.overallScore) {
            this.setText(
                this.elements.overallScore,
                `${quality.overall_quality_score.toFixed(1)}%`
            );
        }

        console.log('[UIController] Updated quality metrics');
    }

    /**
     * Update reuse metrics
     *
     * @param {Object} reuse - ReuseMetrics object
     */
    updateReuseMetrics(reuse) {
        // Update handled by chart manager
        // Could add reuse violations display here if needed
        console.log('[UIController] Reuse metrics updated');
    }

    /**
     * Update gate metrics
     *
     * @param {Object} gates - GateStatus object
     */
    updateGateMetrics(gates) {
        // Update handled by chart manager
        // Could add gate details display here if needed
        console.log('[UIController] Gate metrics updated');
    }

    /**
     * Update recent activity feed
     *
     * @param {Array} activities - Array of activity objects
     */
    updateRecentActivity(activities) {
        if (!this.elements.recentActivity) return;

        if (!activities || activities.length === 0) {
            this.elements.recentActivity.innerHTML = `
                <div class="empty-state text-center py-8">
                    <p class="text-gray-500 text-sm">No recent activity</p>
                </div>
            `;
            return;
        }

        const activityHTML = activities.map(activity => {
            const timestamp = this.formatTimestamp(activity.timestamp);
            const icon = this.getActivityIcon(activity.type);

            return `
                <div class="flex items-start space-x-3 py-2 border-b border-gray-200">
                    <span class="text-xl flex-shrink-0" role="img" aria-label="${activity.type}">${icon}</span>
                    <div class="flex-1 min-w-0">
                        <p class="text-sm text-gray-800">${this.escapeHtml(activity.description)}</p>
                        <p class="text-xs text-gray-500 mt-1">${timestamp}</p>
                    </div>
                </div>
            `;
        }).join('');

        this.elements.recentActivity.innerHTML = activityHTML;

        console.log('[UIController] Updated recent activity');
    }

    /**
     * Update loading states
     *
     * @private
     * @param {Object} loadingStates - Loading state object
     */
    updateLoadingStates(loadingStates) {
        // Show/hide loading skeletons or spinners
        // Implementation depends on loading UI design
        console.log('[UIController] Loading states:', loadingStates);
    }

    /**
     * Update error states
     *
     * @private
     * @param {Object} errors - Error state object
     */
    updateErrorStates(errors) {
        // Show/hide error messages
        // Implementation depends on error UI design
        if (Object.keys(errors).length > 0) {
            console.warn('[UIController] Errors:', errors);
        }
    }

    /**
     * Set metric value and progress bar
     *
     * @private
     * @param {HTMLElement} textEl - Text element for percentage
     * @param {HTMLElement} barEl - Progress bar element
     * @param {number} value - Percentage value (0-100)
     */
    setMetric(textEl, barEl, value) {
        if (!textEl || !barEl) return;

        // Update text
        this.setText(textEl, `${value.toFixed(1)}%`);

        // Update bar width with animation
        barEl.style.width = `${value}%`;

        // Update bar color based on value
        const colorClass = this.getQualityColor(value);
        barEl.className = `h-2 rounded-full transition-all duration-500 ${colorClass}`;
    }

    /**
     * Set text content safely
     *
     * @private
     * @param {HTMLElement} element - Element to update
     * @param {string|number} text - Text content
     */
    setText(element, text) {
        if (!element) return;

        element.textContent = String(text);
    }

    /**
     * Get quality color class based on percentage
     *
     * @private
     * @param {number} percentage - Quality percentage
     * @returns {string} - Tailwind color class
     */
    getQualityColor(percentage) {
        if (percentage >= 90) return 'bg-green-600';   // High quality
        if (percentage >= 70) return 'bg-yellow-600';  // Medium quality
        return 'bg-red-600';                            // Low quality
    }

    /**
     * Get activity icon based on type
     *
     * @private
     * @param {string} type - Activity type
     * @returns {string} - Emoji icon
     */
    getActivityIcon(type) {
        const icons = {
            'reuse_check': '🔍',
            'gate_complete': '✅',
            'gate_update': '🔄',
            'manifest_update': '📋',
            'error': '❌',
            'warning': '⚠️',
            'info': 'ℹ️'
        };

        return icons[type] || '📌';
    }

    /**
     * Format timestamp for display
     *
     * @private
     * @param {string} timestamp - ISO 8601 timestamp
     * @returns {string} - Formatted timestamp
     */
    formatTimestamp(timestamp) {
        try {
            const date = new Date(timestamp);
            const now = new Date();
            const diffMs = now - date;
            const diffMins = Math.floor(diffMs / 60000);

            if (diffMins < 1) return 'Just now';
            if (diffMins < 60) return `${diffMins} minute${diffMins === 1 ? '' : 's'} ago`;

            const diffHours = Math.floor(diffMins / 60);
            if (diffHours < 24) return `${diffHours} hour${diffHours === 1 ? '' : 's'} ago`;

            const diffDays = Math.floor(diffHours / 24);
            if (diffDays < 7) return `${diffDays} day${diffDays === 1 ? '' : 's'} ago`;

            // Fallback to locale string
            return date.toLocaleString();

        } catch (error) {
            return timestamp;
        }
    }

    /**
     * Escape HTML to prevent XSS
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
     * Refresh all UI elements
     * Useful after reconnection or manual refresh
     */
    refresh() {
        const state = this.state.getAll();

        if (state.metrics) this.updateMetricCards(state.metrics);
        if (state.reuse) this.updateReuseMetrics(state.reuse);
        if (state.gates) this.updateGateMetrics(state.gates);
        if (state.trends && state.trends.recent_activity) {
            this.updateRecentActivity(state.trends.recent_activity);
        }

        console.log('[UIController] UI refreshed');
    }

    /**
     * Show toast notification
     *
     * @param {string} message - Toast message
     * @param {string} [type='info'] - Toast type (success, error, warning, info)
     * @param {number} [duration=5000] - Duration in milliseconds
     */
    showToast(message, type = 'info', duration = 5000) {
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.setAttribute('role', 'alert');
        toast.setAttribute('aria-live', 'polite');

        const icon = this.getToastIcon(type);

        toast.innerHTML = `
            <div class="toast__icon">${icon}</div>
            <div class="toast__content">
                <p class="toast__message">${this.escapeHtml(message)}</p>
            </div>
            <button class="toast__dismiss" aria-label="Dismiss notification">&times;</button>
        `;

        document.body.appendChild(toast);

        // Auto-dismiss
        const timeoutId = setTimeout(() => {
            toast.classList.add('toast-exit');
            setTimeout(() => toast.remove(), 300);
        }, duration);

        // Manual dismiss
        const dismissBtn = toast.querySelector('.toast__dismiss');
        dismissBtn.addEventListener('click', () => {
            clearTimeout(timeoutId);
            toast.classList.add('toast-exit');
            setTimeout(() => toast.remove(), 300);
        });
    }

    /**
     * Get toast icon
     *
     * @private
     * @param {string} type - Toast type
     * @returns {string} - Icon HTML or emoji
     */
    getToastIcon(type) {
        const icons = {
            'success': '✅',
            'error': '❌',
            'warning': '⚠️',
            'info': 'ℹ️'
        };

        return icons[type] || 'ℹ️';
    }
}
