/**
 * ChartManager - Chart.js lifecycle management
 *
 * Handles creation, updates, and destruction of Chart.js visualizations
 * with debounced updates for optimal performance (60fps target).
 *
 * @module ChartManager
 */

export class ChartManager {
    /**
     * Create a ChartManager instance
     */
    constructor() {
        this.charts = {}; // Store chart instances by ID
        this.updateDebounceTimers = {}; // Debounce timers for updates
        this.debounceDelay = 16; // ~60fps (16ms)

        // Default Chart.js options
        this.defaultOptions = {
            responsive: true,
            maintainAspectRatio: true,
            animation: {
                duration: 300,
                easing: 'easeInOutQuart'
            }
        };
    }

    /**
     * Create reuse distribution bar chart
     * From UI Design System: Horizontal bar chart showing reuse ranges
     *
     * @param {string} canvasId - Canvas element ID
     * @param {Object} distribution - Distribution data object
     * @param {number} distribution.0-20 - Count in 0-20% range
     * @param {number} distribution.20-40 - Count in 20-40% range
     * @param {number} distribution.40-60 - Count in 40-60% range
     * @param {number} distribution.60-80 - Count in 60-80% range
     * @param {number} distribution.80-100 - Count in 80-100% range
     */
    createReuseChart(canvasId, distribution) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) {
            console.error(`[ChartManager] Canvas element not found: ${canvasId}`);
            return;
        }

        // Destroy existing chart
        if (this.charts[canvasId]) {
            this.destroyChart(canvasId);
        }

        const data = [
            distribution['0-20'] || 0,
            distribution['20-40'] || 0,
            distribution['40-60'] || 0,
            distribution['60-80'] || 0,
            distribution['80-100'] || 0
        ];

        this.charts[canvasId] = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%'],
                datasets: [{
                    label: 'Number of Checks',
                    data: data,
                    backgroundColor: [
                        'rgba(239, 68, 68, 0.7)',   // Red (low)
                        'rgba(251, 191, 36, 0.7)',  // Amber (medium-low)
                        'rgba(59, 130, 246, 0.7)',  // Blue (medium)
                        'rgba(34, 197, 94, 0.7)',   // Green (medium-high)
                        'rgba(16, 185, 129, 0.7)'   // Emerald (high)
                    ],
                    borderColor: [
                        'rgb(239, 68, 68)',
                        'rgb(251, 191, 36)',
                        'rgb(59, 130, 246)',
                        'rgb(34, 197, 94)',
                        'rgb(16, 185, 129)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                ...this.defaultOptions,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        padding: 12,
                        titleFont: { size: 14, weight: 'bold' },
                        bodyFont: { size: 13 },
                        callbacks: {
                            label: (context) => {
                                const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                const percentage = total > 0
                                    ? ((context.parsed.y / total) * 100).toFixed(1)
                                    : 0;
                                return `${context.parsed.y} checks (${percentage}%)`;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            stepSize: 1,
                            precision: 0
                        },
                        grid: {
                            color: 'rgba(0, 0, 0, 0.05)'
                        }
                    },
                    x: {
                        grid: {
                            display: false
                        }
                    }
                }
            }
        });

        console.log(`[ChartManager] Created reuse chart: ${canvasId}`);
    }

    /**
     * Create phase gates doughnut chart
     * From UI Design System: Doughnut chart showing gate distribution
     *
     * @param {string} canvasId - Canvas element ID
     * @param {Object} phaseCounts - Phase gate counts
     * @param {number} phaseCounts.discovery - Discovery phase count
     * @param {number} phaseCounts.design - Design phase count
     * @param {number} phaseCounts.implementation - Implementation phase count
     * @param {number} phaseCounts.verification - Verification phase count
     * @param {number} phaseCounts.integration - Integration phase count
     */
    createGatesChart(canvasId, phaseCounts) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) {
            console.error(`[ChartManager] Canvas element not found: ${canvasId}`);
            return;
        }

        // Destroy existing chart
        if (this.charts[canvasId]) {
            this.destroyChart(canvasId);
        }

        const data = [
            phaseCounts.discovery || 0,
            phaseCounts.design || 0,
            phaseCounts.implementation || 0,
            phaseCounts.verification || 0,
            phaseCounts.integration || 0
        ];

        this.charts[canvasId] = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Discovery', 'Design', 'Implementation', 'Verification', 'Integration'],
                datasets: [{
                    label: 'Completed Gates',
                    data: data,
                    backgroundColor: [
                        'rgba(239, 68, 68, 0.7)',   // Red
                        'rgba(251, 191, 36, 0.7)',  // Amber
                        'rgba(59, 130, 246, 0.7)',  // Blue
                        'rgba(34, 197, 94, 0.7)',   // Green
                        'rgba(168, 85, 247, 0.7)'   // Purple
                    ],
                    borderColor: '#ffffff',
                    borderWidth: 2
                }]
            },
            options: {
                ...this.defaultOptions,
                cutout: '60%',
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            padding: 15,
                            usePointStyle: true,
                            font: {
                                size: 12
                            }
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        padding: 12,
                        callbacks: {
                            label: (context) => {
                                const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                const percentage = total > 0
                                    ? ((context.parsed / total) * 100).toFixed(1)
                                    : 0;
                                return `${context.label}: ${context.parsed} (${percentage}%)`;
                            }
                        }
                    }
                }
            }
        });

        console.log(`[ChartManager] Created gates chart: ${canvasId}`);
    }

    /**
     * Update chart data with debouncing
     * Prevents excessive re-renders during rapid updates
     *
     * @param {string} chartId - Chart instance ID
     * @param {Array} newData - New dataset values
     * @param {number} [datasetIndex=0] - Dataset to update
     */
    updateChart(chartId, newData, datasetIndex = 0) {
        // Clear existing debounce timer
        if (this.updateDebounceTimers[chartId]) {
            clearTimeout(this.updateDebounceTimers[chartId]);
        }

        // Debounce update
        this.updateDebounceTimers[chartId] = setTimeout(() => {
            const chart = this.charts[chartId];

            if (!chart) {
                console.warn(`[ChartManager] Chart not found: ${chartId}`);
                return;
            }

            if (!chart.data.datasets[datasetIndex]) {
                console.warn(`[ChartManager] Dataset ${datasetIndex} not found in chart: ${chartId}`);
                return;
            }

            // Update data
            chart.data.datasets[datasetIndex].data = newData;

            // Update chart with animation
            chart.update('active');

            console.log(`[ChartManager] Updated chart: ${chartId}`);
        }, this.debounceDelay);
    }

    /**
     * Update reuse distribution chart data
     *
     * @param {string} chartId - Chart instance ID
     * @param {Object} distribution - Distribution data object
     */
    updateReuseData(chartId, distribution) {
        const data = [
            distribution['0-20'] || 0,
            distribution['20-40'] || 0,
            distribution['40-60'] || 0,
            distribution['60-80'] || 0,
            distribution['80-100'] || 0
        ];

        this.updateChart(chartId, data);
    }

    /**
     * Update gates chart data
     *
     * @param {string} chartId - Chart instance ID
     * @param {Object} phaseCounts - Phase gate counts
     */
    updateGatesData(chartId, phaseCounts) {
        const data = [
            phaseCounts.discovery || 0,
            phaseCounts.design || 0,
            phaseCounts.implementation || 0,
            phaseCounts.verification || 0,
            phaseCounts.integration || 0
        ];

        this.updateChart(chartId, data);
    }

    /**
     * Destroy a specific chart
     *
     * @param {string} chartId - Chart instance ID
     */
    destroyChart(chartId) {
        const chart = this.charts[chartId];

        if (chart) {
            chart.destroy();
            delete this.charts[chartId];
            console.log(`[ChartManager] Destroyed chart: ${chartId}`);
        }
    }

    /**
     * Destroy all charts
     */
    destroyAll() {
        Object.keys(this.charts).forEach(chartId => {
            this.destroyChart(chartId);
        });

        console.log('[ChartManager] Destroyed all charts');
    }

    /**
     * Check if chart exists
     *
     * @param {string} chartId - Chart instance ID
     * @returns {boolean} - True if chart exists
     */
    hasChart(chartId) {
        return !!this.charts[chartId];
    }

    /**
     * Get chart instance
     *
     * @param {string} chartId - Chart instance ID
     * @returns {Chart|null} - Chart instance or null
     */
    getChart(chartId) {
        return this.charts[chartId] || null;
    }
}
