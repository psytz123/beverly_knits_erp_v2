/**
 * SearchManager - Search and filter functionality
 *
 * Manages search queries, filters, and result display with debouncing,
 * keyboard shortcuts, and search history persistence.
 *
 * @module SearchManager
 */

export class SearchManager {
    /**
     * Create SearchManager instance
     *
     * @param {Object} apiClient - API client for search requests
     */
    constructor(apiClient) {
        this.api = apiClient;

        /** @type {string} Current search query */
        this.currentQuery = '';

        /** @type {Object} Active filters */
        this.activeFilters = {
            dateRange: null,
            status: [],
            type: [],
            category: [],
            reuseRange: null
        };

        /** @type {Array<string>} Search history (max 10) */
        this.searchHistory = this.loadSearchHistory();

        /** @type {number} Debounce timeout ID */
        this.debounceTimeout = null;

        /** @type {number} Debounce delay in milliseconds */
        this.debounceDelay = 300;

        /** @type {Array} Current search results */
        this.results = [];

        /** @type {number} Current page number */
        this.currentPage = 1;

        /** @type {number} Results per page */
        this.pageSize = 20;

        /** @type {number} Total results count */
        this.totalResults = 0;

        /** @type {boolean} Loading state */
        this.isLoading = false;

        this.initializeSearch();
    }

    /**
     * Initialize search system
     * Sets up event listeners and keyboard shortcuts
     *
     * @private
     */
    initializeSearch() {
        this.setupSearchInput();
        this.setupClearButton();
        this.setupKeyboardShortcuts();
        this.setupFilterControls();
        this.setupSearchHistory();

        console.log('[SearchManager] Initialized');
    }

    /**
     * Setup search input event listeners
     *
     * @private
     */
    setupSearchInput() {
        const searchInput = document.getElementById('search-input');
        if (!searchInput) return;

        // Input event with debouncing
        searchInput.addEventListener('input', (e) => {
            this.handleSearchInput(e.target.value);
        });

        // Focus event - show history
        searchInput.addEventListener('focus', () => {
            this.showSearchHistory();
        });

        // Blur event - hide history after delay
        searchInput.addEventListener('blur', () => {
            setTimeout(() => this.hideSearchHistory(), 200);
        });

        console.log('[SearchManager] Search input configured');
    }

    /**
     * Setup clear button functionality
     *
     * @private
     */
    setupClearButton() {
        const clearButton = document.getElementById('search-clear');
        if (!clearButton) return;

        clearButton.addEventListener('click', () => {
            this.clearSearch();
        });

        console.log('[SearchManager] Clear button configured');
    }

    /**
     * Setup keyboard shortcuts
     * Ctrl+K: Focus search
     * Escape: Clear search
     *
     * @private
     */
    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Ctrl+K or Cmd+K: Focus search
            if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
                e.preventDefault();
                this.focusSearch();
            }

            // Escape: Clear search if focused
            if (e.key === 'Escape') {
                const searchInput = document.getElementById('search-input');
                if (document.activeElement === searchInput) {
                    e.preventDefault();
                    this.clearSearch();
                }
            }
        });

        console.log('[SearchManager] Keyboard shortcuts enabled');
    }

    /**
     * Setup filter control event listeners
     *
     * @private
     */
    setupFilterControls() {
        // Date range filter
        const dateRange = document.getElementById('filter-date-range');
        if (dateRange) {
            dateRange.addEventListener('change', (e) => {
                this.setFilter('dateRange', e.target.value);
            });
        }

        // Status checkboxes
        const statusCheckboxes = document.querySelectorAll('[data-filter-status]');
        statusCheckboxes.forEach(checkbox => {
            checkbox.addEventListener('change', (e) => {
                this.toggleFilter('status', e.target.value, e.target.checked);
            });
        });

        // Type checkboxes
        const typeCheckboxes = document.querySelectorAll('[data-filter-type]');
        typeCheckboxes.forEach(checkbox => {
            checkbox.addEventListener('change', (e) => {
                this.toggleFilter('type', e.target.value, e.target.checked);
            });
        });

        // Category multi-select
        const categorySelect = document.getElementById('filter-category');
        if (categorySelect) {
            categorySelect.addEventListener('change', (e) => {
                const selected = Array.from(e.target.selectedOptions).map(opt => opt.value);
                this.setFilter('category', selected);
            });
        }

        // Reuse percentage slider
        const reuseSlider = document.getElementById('filter-reuse-range');
        if (reuseSlider) {
            reuseSlider.addEventListener('input', (e) => {
                this.setFilter('reuseRange', parseInt(e.target.value));
                this.updateReuseRangeDisplay(e.target.value);
            });
        }

        // Reset filters button
        const resetButton = document.getElementById('reset-filters');
        if (resetButton) {
            resetButton.addEventListener('click', () => {
                this.resetFilters();
            });
        }

        // Preset filter buttons
        const presetButtons = document.querySelectorAll('[data-filter-preset]');
        presetButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                this.applyFilterPreset(e.target.dataset.filterPreset);
            });
        });

        console.log('[SearchManager] Filter controls configured');
    }

    /**
     * Setup search history dropdown
     *
     * @private
     */
    setupSearchHistory() {
        const historyContainer = document.getElementById('search-history');
        if (!historyContainer) return;

        // History item click handler (delegated)
        historyContainer.addEventListener('click', (e) => {
            const historyItem = e.target.closest('[data-history-query]');
            if (historyItem) {
                const query = historyItem.dataset.historyQuery;
                this.selectHistoryItem(query);
            }
        });

        console.log('[SearchManager] Search history configured');
    }

    /**
     * Handle search input with debouncing
     *
     * @private
     * @param {string} query - Search query
     */
    handleSearchInput(query) {
        // Update current query
        this.currentQuery = query.trim();

        // Update clear button visibility
        this.updateClearButtonVisibility();

        // Clear existing timeout
        if (this.debounceTimeout) {
            clearTimeout(this.debounceTimeout);
        }

        // Debounce search
        this.debounceTimeout = setTimeout(() => {
            this.performSearch();
        }, this.debounceDelay);
    }

    /**
     * Perform search with current query and filters
     */
    async performSearch() {
        if (!this.currentQuery && this.isFiltersEmpty()) {
            this.clearResults();
            return;
        }

        this.isLoading = true;
        this.updateLoadingState(true);

        try {
            console.log('[SearchManager] Searching:', this.currentQuery, this.activeFilters);

            // Make API request
            const response = await this.api.post('/api/search', {
                query: this.currentQuery,
                filters: this.activeFilters,
                page: this.currentPage,
                page_size: this.pageSize
            });

            // Update results
            this.results = response.results || [];
            this.totalResults = response.total || 0;

            // Display results
            this.displayResults();

            // Update pagination
            this.updatePagination();

            // Add to search history
            if (this.currentQuery) {
                this.addToSearchHistory(this.currentQuery);
            }

            console.log(`[SearchManager] Found ${this.totalResults} results`);

        } catch (error) {
            console.error('[SearchManager] Search failed:', error);
            this.displayError('Search failed. Please try again.');

        } finally {
            this.isLoading = false;
            this.updateLoadingState(false);
        }
    }

    /**
     * Display search results
     *
     * @private
     */
    displayResults() {
        const resultsContainer = document.getElementById('search-results');
        if (!resultsContainer) return;

        if (this.results.length === 0) {
            resultsContainer.innerHTML = `
                <div class="empty-state text-center py-12 bg-gray-50 rounded-lg">
                    <span class="text-4xl mb-4 block" aria-hidden="true">🔍</span>
                    <p class="text-gray-600">No results found</p>
                    <p class="text-sm text-gray-500 mt-2">Try adjusting your search or filters</p>
                </div>
            `;
            return;
        }

        const resultsHTML = this.results.map(result => this.renderResultItem(result)).join('');

        resultsContainer.innerHTML = `
            <div class="mb-4 text-sm text-gray-600">
                Found ${this.totalResults} result${this.totalResults === 1 ? '' : 's'}
            </div>
            <div class="space-y-4">
                ${resultsHTML}
            </div>
        `;

        // Announce to screen readers
        this.announceToScreenReader(`Found ${this.totalResults} results`);
    }

    /**
     * Render single result item
     *
     * @private
     * @param {Object} result - Result object
     * @returns {string} - HTML string
     */
    renderResultItem(result) {
        const icon = this.getResultIcon(result.type);
        const timestamp = this.formatTimestamp(result.timestamp);
        const highlightedDescription = this.highlightMatches(result.description, this.currentQuery);
        const statusBadge = this.getStatusBadge(result.status);

        return `
            <div class="bg-white border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow">
                <div class="flex items-start space-x-3">
                    <span class="text-2xl flex-shrink-0" role="img" aria-label="${result.type}">${icon}</span>
                    <div class="flex-1 min-w-0">
                        <div class="flex items-center justify-between mb-2">
                            <h4 class="text-sm font-semibold text-gray-800">${this.escapeHtml(result.title || 'Untitled')}</h4>
                            ${statusBadge}
                        </div>
                        <p class="text-sm text-gray-700 mb-2">${highlightedDescription}</p>
                        <div class="flex items-center space-x-4 text-xs text-gray-500">
                            <span>${timestamp}</span>
                            ${result.category ? `<span class="px-2 py-1 bg-blue-100 text-blue-700 rounded">${this.escapeHtml(result.category)}</span>` : ''}
                            ${result.reuse_percentage !== undefined ? `<span class="px-2 py-1 bg-green-100 text-green-700 rounded">${result.reuse_percentage}% reuse</span>` : ''}
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    /**
     * Highlight search query matches in text
     *
     * @private
     * @param {string} text - Text to highlight
     * @param {string} query - Search query
     * @returns {string} - HTML with highlights
     */
    highlightMatches(text, query) {
        if (!query || !text) return this.escapeHtml(text);

        const escapedText = this.escapeHtml(text);
        const escapedQuery = this.escapeHtml(query);

        // Case-insensitive highlighting
        const regex = new RegExp(`(${escapedQuery})`, 'gi');
        return escapedText.replace(regex, '<mark class="bg-yellow-200 font-semibold">$1</mark>');
    }

    /**
     * Get icon for result type
     *
     * @private
     * @param {string} type - Result type
     * @returns {string} - Emoji icon
     */
    getResultIcon(type) {
        const icons = {
            'reuse_check': '🔍',
            'phase_gate': '🚪',
            'agent_activity': '🤖',
            'metric': '📊',
            'error': '❌',
            'warning': '⚠️'
        };

        return icons[type] || '📌';
    }

    /**
     * Get status badge HTML
     *
     * @private
     * @param {string} status - Result status
     * @returns {string} - HTML badge
     */
    getStatusBadge(status) {
        const badges = {
            'passed': '<span class="px-2 py-1 text-xs font-semibold bg-green-100 text-green-800 rounded">Passed</span>',
            'failed': '<span class="px-2 py-1 text-xs font-semibold bg-red-100 text-red-800 rounded">Failed</span>',
            'in_progress': '<span class="px-2 py-1 text-xs font-semibold bg-blue-100 text-blue-800 rounded">In Progress</span>',
            'pending': '<span class="px-2 py-1 text-xs font-semibold bg-yellow-100 text-yellow-800 rounded">Pending</span>'
        };

        return badges[status] || '';
    }

    /**
     * Clear search results
     *
     * @private
     */
    clearResults() {
        const resultsContainer = document.getElementById('search-results');
        if (resultsContainer) {
            resultsContainer.innerHTML = '';
        }

        this.results = [];
        this.totalResults = 0;
        this.currentPage = 1;
    }

    /**
     * Display error message
     *
     * @private
     * @param {string} message - Error message
     */
    displayError(message) {
        const resultsContainer = document.getElementById('search-results');
        if (!resultsContainer) return;

        resultsContainer.innerHTML = `
            <div class="bg-red-50 border border-red-200 rounded-lg p-4" role="alert">
                <div class="flex items-center space-x-2">
                    <span class="text-xl" aria-hidden="true">❌</span>
                    <p class="text-sm text-red-800">${this.escapeHtml(message)}</p>
                </div>
            </div>
        `;
    }

    /**
     * Update pagination controls
     *
     * @private
     */
    updatePagination() {
        const paginationContainer = document.getElementById('search-pagination');
        if (!paginationContainer) return;

        const totalPages = Math.ceil(this.totalResults / this.pageSize);

        if (totalPages <= 1) {
            paginationContainer.innerHTML = '';
            return;
        }

        const prevDisabled = this.currentPage === 1;
        const nextDisabled = this.currentPage === totalPages;

        paginationContainer.innerHTML = `
            <div class="flex items-center justify-between mt-6">
                <button
                    id="prev-page"
                    class="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                    ${prevDisabled ? 'disabled' : ''}
                    aria-label="Previous page">
                    Previous
                </button>
                <span class="text-sm text-gray-600">
                    Page ${this.currentPage} of ${totalPages}
                </span>
                <button
                    id="next-page"
                    class="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                    ${nextDisabled ? 'disabled' : ''}
                    aria-label="Next page">
                    Next
                </button>
            </div>
        `;

        // Add event listeners
        document.getElementById('prev-page')?.addEventListener('click', () => {
            this.goToPage(this.currentPage - 1);
        });

        document.getElementById('next-page')?.addEventListener('click', () => {
            this.goToPage(this.currentPage + 1);
        });
    }

    /**
     * Navigate to specific page
     *
     * @param {number} page - Page number
     */
    goToPage(page) {
        this.currentPage = page;
        this.performSearch();
    }

    /**
     * Set filter value
     *
     * @param {string} filterKey - Filter key
     * @param {*} value - Filter value
     */
    setFilter(filterKey, value) {
        this.activeFilters[filterKey] = value;
        this.updateActiveFiltersDisplay();
        this.currentPage = 1; // Reset to first page
        this.performSearch();
    }

    /**
     * Toggle filter value in array
     *
     * @param {string} filterKey - Filter key
     * @param {string} value - Value to toggle
     * @param {boolean} checked - Whether value is checked
     */
    toggleFilter(filterKey, value, checked) {
        if (checked) {
            if (!this.activeFilters[filterKey].includes(value)) {
                this.activeFilters[filterKey].push(value);
            }
        } else {
            this.activeFilters[filterKey] = this.activeFilters[filterKey].filter(v => v !== value);
        }

        this.updateActiveFiltersDisplay();
        this.currentPage = 1;
        this.performSearch();
    }

    /**
     * Apply filter preset
     *
     * @param {string} presetName - Preset name
     */
    applyFilterPreset(presetName) {
        const presets = {
            'recent-failures': {
                dateRange: 'last-7-days',
                status: ['failed'],
                type: [],
                category: [],
                reuseRange: null
            },
            'high-reuse': {
                dateRange: null,
                status: [],
                type: ['reuse_check'],
                category: [],
                reuseRange: 70
            },
            'pending-gates': {
                dateRange: null,
                status: ['pending', 'in_progress'],
                type: ['phase_gate'],
                category: [],
                reuseRange: null
            }
        };

        const preset = presets[presetName];
        if (preset) {
            this.activeFilters = { ...preset };
            this.syncFiltersToUI();
            this.updateActiveFiltersDisplay();
            this.currentPage = 1;
            this.performSearch();
        }
    }

    /**
     * Reset all filters
     */
    resetFilters() {
        this.activeFilters = {
            dateRange: null,
            status: [],
            type: [],
            category: [],
            reuseRange: null
        };

        this.syncFiltersToUI();
        this.updateActiveFiltersDisplay();
        this.currentPage = 1;
        this.performSearch();
    }

    /**
     * Sync filter state to UI controls
     *
     * @private
     */
    syncFiltersToUI() {
        // Date range
        const dateRange = document.getElementById('filter-date-range');
        if (dateRange) dateRange.value = this.activeFilters.dateRange || '';

        // Status checkboxes
        document.querySelectorAll('[data-filter-status]').forEach(checkbox => {
            checkbox.checked = this.activeFilters.status.includes(checkbox.value);
        });

        // Type checkboxes
        document.querySelectorAll('[data-filter-type]').forEach(checkbox => {
            checkbox.checked = this.activeFilters.type.includes(checkbox.value);
        });

        // Category select
        const categorySelect = document.getElementById('filter-category');
        if (categorySelect) {
            Array.from(categorySelect.options).forEach(option => {
                option.selected = this.activeFilters.category.includes(option.value);
            });
        }

        // Reuse slider
        const reuseSlider = document.getElementById('filter-reuse-range');
        if (reuseSlider) {
            reuseSlider.value = this.activeFilters.reuseRange || 0;
            this.updateReuseRangeDisplay(this.activeFilters.reuseRange || 0);
        }
    }

    /**
     * Update active filters display
     *
     * @private
     */
    updateActiveFiltersDisplay() {
        const container = document.getElementById('active-filters');
        if (!container) return;

        const activeFilterTags = [];

        // Date range
        if (this.activeFilters.dateRange) {
            activeFilterTags.push({
                label: `Date: ${this.activeFilters.dateRange}`,
                key: 'dateRange'
            });
        }

        // Status
        this.activeFilters.status.forEach(status => {
            activeFilterTags.push({
                label: `Status: ${status}`,
                key: 'status',
                value: status
            });
        });

        // Type
        this.activeFilters.type.forEach(type => {
            activeFilterTags.push({
                label: `Type: ${type}`,
                key: 'type',
                value: type
            });
        });

        // Category
        this.activeFilters.category.forEach(category => {
            activeFilterTags.push({
                label: `Category: ${category}`,
                key: 'category',
                value: category
            });
        });

        // Reuse range
        if (this.activeFilters.reuseRange !== null) {
            activeFilterTags.push({
                label: `Reuse: ≥${this.activeFilters.reuseRange}%`,
                key: 'reuseRange'
            });
        }

        if (activeFilterTags.length === 0) {
            container.innerHTML = '';
            return;
        }

        const tagsHTML = activeFilterTags.map(tag => `
            <span class="inline-flex items-center px-3 py-1 text-sm bg-blue-100 text-blue-800 rounded-full">
                ${this.escapeHtml(tag.label)}
                <button
                    class="ml-2 text-blue-600 hover:text-blue-800 focus:outline-none"
                    onclick="window.searchManager.removeFilter('${tag.key}', '${tag.value || ''}')"
                    aria-label="Remove ${tag.label} filter">
                    &times;
                </button>
            </span>
        `).join('');

        container.innerHTML = `
            <div class="flex flex-wrap gap-2 items-center">
                <span class="text-sm text-gray-600">Active filters:</span>
                ${tagsHTML}
            </div>
        `;
    }

    /**
     * Remove specific filter
     *
     * @param {string} key - Filter key
     * @param {string} [value] - Filter value (for array filters)
     */
    removeFilter(key, value = '') {
        if (Array.isArray(this.activeFilters[key])) {
            this.activeFilters[key] = this.activeFilters[key].filter(v => v !== value);
        } else {
            this.activeFilters[key] = null;
        }

        this.syncFiltersToUI();
        this.updateActiveFiltersDisplay();
        this.currentPage = 1;
        this.performSearch();
    }

    /**
     * Update reuse range display
     *
     * @private
     * @param {number} value - Slider value
     */
    updateReuseRangeDisplay(value) {
        const display = document.getElementById('reuse-range-value');
        if (display) {
            display.textContent = `${value}%`;
        }
    }

    /**
     * Check if all filters are empty
     *
     * @private
     * @returns {boolean} - True if no filters active
     */
    isFiltersEmpty() {
        return !this.activeFilters.dateRange &&
               this.activeFilters.status.length === 0 &&
               this.activeFilters.type.length === 0 &&
               this.activeFilters.category.length === 0 &&
               this.activeFilters.reuseRange === null;
    }

    /**
     * Focus search input
     */
    focusSearch() {
        const searchInput = document.getElementById('search-input');
        if (searchInput) {
            searchInput.focus();
            searchInput.select();
        }
    }

    /**
     * Clear search
     */
    clearSearch() {
        const searchInput = document.getElementById('search-input');
        if (searchInput) {
            searchInput.value = '';
        }

        this.currentQuery = '';
        this.updateClearButtonVisibility();
        this.clearResults();
    }

    /**
     * Update clear button visibility
     *
     * @private
     */
    updateClearButtonVisibility() {
        const clearButton = document.getElementById('search-clear');
        if (!clearButton) return;

        if (this.currentQuery) {
            clearButton.classList.remove('hidden');
        } else {
            clearButton.classList.add('hidden');
        }
    }

    /**
     * Update loading state
     *
     * @private
     * @param {boolean} loading - Loading state
     */
    updateLoadingState(loading) {
        const spinner = document.getElementById('search-spinner');
        if (spinner) {
            if (loading) {
                spinner.classList.remove('hidden');
            } else {
                spinner.classList.add('hidden');
            }
        }

        // Disable search input during loading
        const searchInput = document.getElementById('search-input');
        if (searchInput) {
            searchInput.disabled = loading;
        }
    }

    /**
     * Show search history dropdown
     *
     * @private
     */
    showSearchHistory() {
        if (this.searchHistory.length === 0) return;

        const historyContainer = document.getElementById('search-history');
        if (!historyContainer) return;

        const historyHTML = this.searchHistory.map(query => `
            <button
                class="w-full px-4 py-2 text-left text-sm text-gray-700 hover:bg-gray-100 focus:outline-none focus:bg-gray-100"
                data-history-query="${this.escapeHtml(query)}"
                role="option">
                <span class="inline-block mr-2" aria-hidden="true">🕐</span>
                ${this.escapeHtml(query)}
            </button>
        `).join('');

        historyContainer.innerHTML = historyHTML;
        historyContainer.classList.remove('hidden');
    }

    /**
     * Hide search history dropdown
     *
     * @private
     */
    hideSearchHistory() {
        const historyContainer = document.getElementById('search-history');
        if (historyContainer) {
            historyContainer.classList.add('hidden');
        }
    }

    /**
     * Select history item
     *
     * @private
     * @param {string} query - Query from history
     */
    selectHistoryItem(query) {
        const searchInput = document.getElementById('search-input');
        if (searchInput) {
            searchInput.value = query;
        }

        this.currentQuery = query;
        this.updateClearButtonVisibility();
        this.performSearch();
        this.hideSearchHistory();
    }

    /**
     * Add query to search history
     *
     * @private
     * @param {string} query - Search query
     */
    addToSearchHistory(query) {
        if (!query || this.searchHistory.includes(query)) return;

        this.searchHistory.unshift(query);
        this.searchHistory = this.searchHistory.slice(0, 10); // Keep last 10
        this.saveSearchHistory();
    }

    /**
     * Load search history from localStorage
     *
     * @private
     * @returns {Array<string>} - Search history
     */
    loadSearchHistory() {
        try {
            const stored = localStorage.getItem('ai_workspace_search_history');
            return stored ? JSON.parse(stored) : [];
        } catch (error) {
            console.error('[SearchManager] Failed to load search history:', error);
            return [];
        }
    }

    /**
     * Save search history to localStorage
     *
     * @private
     */
    saveSearchHistory() {
        try {
            localStorage.setItem('ai_workspace_search_history', JSON.stringify(this.searchHistory));
        } catch (error) {
            console.error('[SearchManager] Failed to save search history:', error);
        }
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
            if (diffMins < 60) return `${diffMins}m ago`;

            const diffHours = Math.floor(diffMins / 60);
            if (diffHours < 24) return `${diffHours}h ago`;

            const diffDays = Math.floor(diffHours / 24);
            if (diffDays < 7) return `${diffDays}d ago`;

            return date.toLocaleDateString();

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
        if (typeof unsafe !== 'string') return '';

        const div = document.createElement('div');
        div.textContent = unsafe;
        return div.innerHTML;
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
        announcer.setAttribute('aria-atomic', 'true');
        announcer.className = 'sr-only';
        announcer.textContent = message;

        document.body.appendChild(announcer);

        setTimeout(() => announcer.remove(), 1000);
    }
}

// Export singleton (will be initialized in main.js)
export let searchManager = null;

/**
 * Initialize search manager
 *
 * @param {Object} apiClient - API client instance
 * @returns {SearchManager} - Search manager instance
 */
export function initializeSearch(apiClient) {
    searchManager = new SearchManager(apiClient);

    // Expose to global scope for event handlers
    window.searchManager = searchManager;

    return searchManager;
}
