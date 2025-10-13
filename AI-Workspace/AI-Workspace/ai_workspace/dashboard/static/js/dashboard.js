// AI Workspace Dashboard JavaScript
// Handles real-time updates and chart visualization

// Global chart instances
let reuseChart = null;
let gatesChart = null;

// WebSocket connection
const socket = io();

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function() {
    loadDashboardData();
    setupWebSocket();
    setupCharts();
});

// Load dashboard data from API
async function loadDashboardData() {
    try {
        // Load summary metrics
        const summaryResponse = await fetch('/api/metrics/summary');
        const summary = await summaryResponse.json();

        updateMetricCards(summary);
        updateQualityMetrics(summary.quality);

        // Load reuse distribution
        const reuseResponse = await fetch('/api/metrics/reuse');
        const reuseData = await reuseResponse.json();

        updateReuseChart(reuseData.distribution);

        // Load gate progress
        const gatesResponse = await fetch('/api/gates/status');
        const gatesData = await gatesResponse.json();

        updateGatesChart(gatesData);

        // Load recent activity
        const trendsResponse = await fetch('/api/metrics/trends');
        const trendsData = await trendsResponse.json();

        updateRecentActivity(trendsData.recent_activity);

    } catch (error) {
        console.error('Error loading dashboard data:', error);
    }
}

// Update metric cards
function updateMetricCards(summary) {
    document.getElementById('avg-reuse').textContent =
        summary.reuse.avg_percentage.toFixed(1) + '%';

    document.getElementById('gates-complete').textContent =
        summary.gates.completed + '/' + summary.gates.total;

    document.getElementById('primary-lang').textContent =
        summary.project.primary_language || 'Unknown';

    document.getElementById('total-checks').textContent =
        summary.reuse.total_checks || 0;
}

// Update quality metrics
function updateQualityMetrics(quality) {
    // Reuse compliance
    document.getElementById('reuse-compliance').textContent =
        quality.reuse_compliance.toFixed(1) + '%';
    document.getElementById('reuse-bar').style.width =
        quality.reuse_compliance + '%';

    // Check before create compliance
    document.getElementById('check-compliance').textContent =
        quality.check_before_create_compliance.toFixed(1) + '%';
    document.getElementById('check-bar').style.width =
        quality.check_before_create_compliance + '%';

    // Gate compliance
    document.getElementById('gate-compliance').textContent =
        quality.gate_compliance.toFixed(1) + '%';
    document.getElementById('gate-bar').style.width =
        quality.gate_compliance + '%';

    // Overall score
    document.getElementById('overall-score').textContent =
        quality.overall_quality_score.toFixed(1) + '%';
}

// Setup charts
function setupCharts() {
    // Reuse distribution chart
    const reuseCtx = document.getElementById('reuseChart').getContext('2d');
    reuseChart = new Chart(reuseCtx, {
        type: 'bar',
        data: {
            labels: ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%'],
            datasets: [{
                label: 'Number of Checks',
                data: [0, 0, 0, 0, 0],
                backgroundColor: [
                    'rgba(239, 68, 68, 0.7)',
                    'rgba(251, 191, 36, 0.7)',
                    'rgba(59, 130, 246, 0.7)',
                    'rgba(34, 197, 94, 0.7)',
                    'rgba(16, 185, 129, 0.7)'
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
            responsive: true,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        stepSize: 1
                    }
                }
            }
        }
    });

    // Phase gate chart
    const gatesCtx = document.getElementById('gatesChart').getContext('2d');
    gatesChart = new Chart(gatesCtx, {
        type: 'doughnut',
        data: {
            labels: ['Discovery', 'Design', 'Implementation', 'Verification', 'Integration'],
            datasets: [{
                label: 'Completed Gates',
                data: [0, 0, 0, 0, 0],
                backgroundColor: [
                    'rgba(239, 68, 68, 0.7)',
                    'rgba(251, 191, 36, 0.7)',
                    'rgba(59, 130, 246, 0.7)',
                    'rgba(34, 197, 94, 0.7)',
                    'rgba(168, 85, 247, 0.7)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'bottom'
                }
            }
        }
    });
}

// Update reuse chart
function updateReuseChart(distribution) {
    if (reuseChart) {
        reuseChart.data.datasets[0].data = [
            distribution['0-20'],
            distribution['20-40'],
            distribution['40-60'],
            distribution['60-80'],
            distribution['80-100']
        ];
        reuseChart.update();
    }
}

// Update gates chart
function updateGatesChart(gatesData) {
    if (gatesChart && gatesData.phase_counts) {
        gatesChart.data.datasets[0].data = [
            gatesData.phase_counts.discovery || 0,
            gatesData.phase_counts.design || 0,
            gatesData.phase_counts.implementation || 0,
            gatesData.phase_counts.verification || 0,
            gatesData.phase_counts.integration || 0
        ];
        gatesChart.update();
    }
}

// Update recent activity
function updateRecentActivity(activities) {
    const container = document.getElementById('recent-activity');

    if (!activities || activities.length === 0) {
        container.innerHTML = '<p class="text-gray-500 text-sm">No recent activity</p>';
        return;
    }

    const activityHTML = activities.map(activity => {
        const timestamp = new Date(activity.timestamp).toLocaleString();
        const icon = activity.type === 'reuse_check' ? '🔍' : '✅';

        return `
            <div class="flex items-start space-x-3 py-2 border-b border-gray-200">
                <span class="text-xl">${icon}</span>
                <div class="flex-1">
                    <p class="text-sm text-gray-800">${activity.description}</p>
                    <p class="text-xs text-gray-500">${timestamp}</p>
                </div>
            </div>
        `;
    }).join('');

    container.innerHTML = activityHTML;
}

// Setup WebSocket for real-time updates
function setupWebSocket() {
    socket.on('connect', function() {
        console.log('WebSocket connected');
        updateConnectionStatus(true);
    });

    socket.on('disconnect', function() {
        console.log('WebSocket disconnected');
        updateConnectionStatus(false);
    });

    socket.on('workspace_update', function(data) {
        console.log('Workspace update:', data);
        // Reload dashboard data when files change
        loadDashboardData();
    });
}

// Update connection status indicator
function updateConnectionStatus(connected) {
    const statusEl = document.getElementById('connection-status');

    if (connected) {
        statusEl.innerHTML = `
            <span class="inline-block w-2 h-2 bg-green-400 rounded-full mr-1"></span>
            Connected
        `;
    } else {
        statusEl.innerHTML = `
            <span class="inline-block w-2 h-2 bg-red-400 rounded-full mr-1"></span>
            Disconnected
        `;
    }
}
