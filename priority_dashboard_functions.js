// ========== PRIORITY ISSUES DASHBOARD FUNCTIONS ==========

// Global variable to track current active category
let currentPriorityCategory = 'productionRisks';

// Function to switch between priority categories
function switchPriorityCategory(category) {
    console.log('Switching priority category to:', category);
    
    // Update active tab
    document.querySelectorAll('.priority-tab').forEach(tab => {
        tab.classList.remove('active');
        tab.classList.add('text-gray-500');
        tab.classList.remove('text-blue-600', 'border-blue-600');
        tab.classList.add('border-transparent');
    });
    
    // Hide all content sections
    document.querySelectorAll('.priority-content').forEach(content => {
        content.classList.add('hidden');
    });
    
    // Show selected content and activate tab
    const selectedTab = document.getElementById(category + 'Tab');
    const selectedContent = document.getElementById(category + 'Content');
    
    if (selectedTab) {
        selectedTab.classList.add('active');
        selectedTab.classList.remove('text-gray-500');
        selectedTab.classList.add('text-blue-600', 'border-blue-600');
        selectedTab.classList.remove('border-transparent');
    }
    
    if (selectedContent) {
        selectedContent.classList.remove('hidden');
    }
    
    // Update current category
    currentPriorityCategory = category;
    
    // Load data for selected category
    loadCategoryData(category);
}

// Function to load data for a specific category
async function loadCategoryData(category) {
    try {
        switch (category) {
            case 'productionRisks':
                await loadProductionRisks();
                break;
            case 'materialShortages':
                await loadMaterialShortages();
                break;
            case 'capacityBottlenecks':
                await loadCapacityBottlenecks();
                break;
            case 'inventoryAlerts':
                await loadInventoryAlerts();
                break;
        }
    } catch (error) {
        console.error(`Error loading ${category} data:`, error);
    }
}

// Load production risks data
async function loadProductionRisks() {
    try {
        const data = await fetchAPI('po-risk-analysis?limit=20&category=all');
        console.log('Production risks data:', data);
        
        if (data && data.status === 'success') {
            updateCategoryCount('productionRisks', data.summary.critical_orders);
            renderProductionRisks(data);
        }
    } catch (error) {
        console.error('Error loading production risks:', error);
        const tbody = document.getElementById('productionRiskTableBody');
        if (tbody) {
            tbody.innerHTML = 
                '<tr><td colspan="5" class="px-4 py-8 text-center text-red-500">Error loading production risks</td></tr>';
        }
    }
}

// Load material shortages data
async function loadMaterialShortages() {
    try {
        const data = await fetchAPI('yarn-intelligence?analysis=shortage&limit=20&sort=urgency');
        console.log('Material shortages data:', data);
        
        if (data && data.shortage_analysis) {
            updateCategoryCount('materialShortages', data.shortage_analysis.critical_count);
            renderMaterialShortages(data.shortage_analysis);
        }
    } catch (error) {
        console.error('Error loading material shortages:', error);
        const tbody = document.getElementById('materialShortageTableBody');
        if (tbody) {
            tbody.innerHTML = 
                '<tr><td colspan="5" class="px-4 py-8 text-center text-red-500">Error loading material shortages</td></tr>';
        }
    }
}

// Load capacity bottlenecks data  
async function loadCapacityBottlenecks() {
    try {
        const data = await fetchAPI('capacity-bottlenecks?limit=20&category=all');
        console.log('Capacity bottlenecks data:', data);
        
        if (data && data.status === 'success') {
            updateCategoryCount('capacityBottlenecks', data.summary.critical_bottlenecks);
            renderCapacityBottlenecks(data);
        }
    } catch (error) {
        console.error('Error loading capacity bottlenecks:', error);
        const tbody = document.getElementById('capacityBottleneckTableBody');
        if (tbody) {
            tbody.innerHTML = 
                '<tr><td colspan="5" class="px-4 py-8 text-center text-red-500">Error loading capacity issues</td></tr>';
        }
    }
}

// Load inventory alerts data
async function loadInventoryAlerts() {
    try {
        const data = await fetchAPI('inventory-intelligence-enhanced?analysis=alerts&limit=20');
        console.log('Inventory alerts data:', data);
        
        if (data && data.inventory_alerts) {
            updateCategoryCount('inventoryAlerts', data.inventory_alerts.summary.critical_alerts);
            renderInventoryAlerts(data.inventory_alerts);
        }
    } catch (error) {
        console.error('Error loading inventory alerts:', error);
        const tbody = document.getElementById('inventoryAlertTableBody');
        if (tbody) {
            tbody.innerHTML = 
                '<tr><td colspan="5" class="px-4 py-8 text-center text-red-500">Error loading inventory alerts</td></tr>';
        }
    }
}

// Update category count badge
function updateCategoryCount(category, count) {
    const countElement = document.getElementById(category + 'Count');
    if (countElement) {
        countElement.textContent = count || 0;
    }
}

// Render production risks data
function renderProductionRisks(data) {
    const tbody = document.getElementById('productionRiskTableBody');
    if (!tbody || !data.risk_analysis) return;
    
    let html = '';
    data.risk_analysis.forEach((risk, index) => {
        const priorityBadge = getPriorityBadge(risk.priority || (index + 1), risk.risk_level);
        const actionButton = getActionButton('Review Order', 'production');
        
        html += `
            <tr class="hover:bg-gray-50">
                <td class="px-4 py-3 text-center">${priorityBadge}</td>
                <td class="px-4 py-3">
                    <div class="text-sm font-medium text-gray-900">${risk.order_id || risk.order_number || 'N/A'}</div>
                    <div class="text-sm text-gray-500">Style: ${risk.style || 'N/A'} • ${(risk.balance_lbs || 0).toFixed(0)} lbs</div>
                </td>
                <td class="px-4 py-3 text-right">
                    <div class="text-sm font-medium text-gray-900">$${(risk.estimated_value || 0).toLocaleString()}</div>
                    <div class="text-sm text-red-600">-$${(risk.potential_loss || 0).toLocaleString()}</div>
                </td>
                <td class="px-4 py-3 text-center">
                    <span class="text-sm ${risk.days_until_start < 0 ? 'text-red-600 font-medium' : 'text-gray-600'}">
                        ${risk.days_until_start < 0 ? `${Math.abs(risk.days_until_start)} overdue` : `${risk.days_until_start} days`}
                    </span>
                </td>
                <td class="px-4 py-3 text-center">${actionButton}</td>
            </tr>
        `;
    });
    
    tbody.innerHTML = html || '<tr><td colspan="5" class="px-4 py-8 text-center text-gray-500">No production risks found</td></tr>';
}

// Render material shortages data
function renderMaterialShortages(data) {
    const tbody = document.getElementById('materialShortageTableBody');
    if (!tbody || !data.critical_shortages) return;
    
    let html = '';
    data.critical_shortages.forEach((shortage, index) => {
        const priorityBadge = getPriorityBadge(index + 1, shortage.severity || shortage.risk_level);
        const actionButton = getActionButton('Order Yarn', 'material');
        
        html += `
            <tr class="hover:bg-gray-50">
                <td class="px-4 py-3 text-center">${priorityBadge}</td>
                <td class="px-4 py-3">
                    <div class="text-sm font-medium text-gray-900">${shortage.yarn_id || 'Unknown'}</div>
                    <div class="text-sm text-gray-500">Severity: ${shortage.severity || 'N/A'}</div>
                </td>
                <td class="px-4 py-3 text-right">
                    <span class="text-sm font-medium text-red-600">
                        ${Math.abs(shortage.shortage || shortage.shortage_pounds || 0).toFixed(0)} lbs short
                    </span>
                </td>
                <td class="px-4 py-3 text-right">
                    <span class="text-sm font-medium text-gray-900">
                        ${shortage.estimated_cost || '$0'}
                    </span>
                </td>
                <td class="px-4 py-3 text-center">${actionButton}</td>
            </tr>
        `;
    });
    
    tbody.innerHTML = html || '<tr><td colspan="5" class="px-4 py-8 text-center text-gray-500">No material shortages found</td></tr>';
}

// Render capacity bottlenecks data
function renderCapacityBottlenecks(data) {
    const tbody = document.getElementById('capacityBottleneckTableBody');
    if (!tbody || !data.bottlenecks) return;
    
    let html = '';
    data.bottlenecks.forEach((bottleneck, index) => {
        const priorityBadge = getPriorityBadge(index + 1, bottleneck.severity);
        const actionButton = getActionButton(bottleneck.suggested_action || 'Resolve', 'capacity');
        
        html += `
            <tr class="hover:bg-gray-50">
                <td class="px-4 py-3 text-center">${priorityBadge}</td>
                <td class="px-4 py-3">
                    <div class="text-sm font-medium text-gray-900">${bottleneck.title}</div>
                    <div class="text-sm text-gray-500">${bottleneck.description}</div>
                </td>
                <td class="px-4 py-3 text-center">
                    <span class="text-sm text-gray-600">
                        ${bottleneck.machine_id || bottleneck.order_id || '--'}
                    </span>
                </td>
                <td class="px-4 py-3 text-right">
                    <span class="text-sm font-medium text-gray-900">
                        ${bottleneck.estimated_impact || 'Process delay'}
                    </span>
                </td>
                <td class="px-4 py-3 text-center">${actionButton}</td>
            </tr>
        `;
    });
    
    tbody.innerHTML = html || '<tr><td colspan="5" class="px-4 py-8 text-center text-gray-500">No capacity issues found</td></tr>';
}

// Render inventory alerts data
function renderInventoryAlerts(data) {
    const tbody = document.getElementById('inventoryAlertTableBody');
    if (!tbody || !data.alerts) return;
    
    let html = '';
    data.alerts.forEach((alert, index) => {
        const priorityBadge = getPriorityBadge(index + 1, alert.severity);
        const actionButton = getActionButton(alert.suggested_action || 'Review', 'inventory');
        
        html += `
            <tr class="hover:bg-gray-50">
                <td class="px-4 py-3 text-center">${priorityBadge}</td>
                <td class="px-4 py-3">
                    <div class="text-sm font-medium text-gray-900">${alert.yarn_id || 'Unknown'}</div>
                    <div class="text-sm text-gray-500">${alert.category || 'N/A'}</div>
                </td>
                <td class="px-4 py-3 text-right">
                    <span class="text-sm font-medium text-gray-900">
                        ${alert.shortage_lbs ? `${alert.shortage_lbs.toFixed(0)} lbs` : 
                          alert.stock_lbs ? `${alert.stock_lbs.toFixed(0)} lbs` : '--'}
                    </span>
                </td>
                <td class="px-4 py-3 text-center">
                    <span class="text-sm text-gray-600">
                        ${alert.coverage_days ? `${alert.coverage_days.toFixed(0)} days` : '--'}
                    </span>
                </td>
                <td class="px-4 py-3 text-center">${actionButton}</td>
            </tr>
        `;
    });
    
    tbody.innerHTML = html || '<tr><td colspan="5" class="px-4 py-8 text-center text-gray-500">No inventory alerts found</td></tr>';
}

// Helper function to get priority badge HTML
function getPriorityBadge(priority, severity) {
    const colors = {
        'CRITICAL': 'bg-red-100 text-red-800 border-red-200',
        'HIGH': 'bg-orange-100 text-orange-800 border-orange-200',
        'MEDIUM': 'bg-yellow-100 text-yellow-800 border-yellow-200',
        'LOW': 'bg-green-100 text-green-800 border-green-200'
    };
    
    const color = colors[severity] || colors['MEDIUM'];
    return `<span class="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium border ${color}">
                ${priority}
            </span>`;
}

// Helper function to get action button HTML
function getActionButton(action, category) {
    const colors = {
        'production': 'bg-red-600 hover:bg-red-700',
        'material': 'bg-orange-600 hover:bg-orange-700', 
        'capacity': 'bg-yellow-600 hover:bg-yellow-700',
        'inventory': 'bg-blue-600 hover:bg-blue-700'
    };
    
    const color = colors[category] || colors['production'];
    return `<button class="inline-flex items-center px-3 py-1 rounded text-xs font-medium text-white ${color} transition-colors">
                ${action}
            </button>`;
}

// Main function to load all priority dashboard data
async function loadPriorityDashboard() {
    console.log('Loading priority dashboard...');
    
    // Load data for all categories in parallel
    try {
        await Promise.all([
            loadProductionRisks(),
            loadMaterialShortages(),
            loadCapacityBottlenecks(),
            loadInventoryAlerts()
        ]);
        console.log('Priority dashboard loaded successfully');
    } catch (error) {
        console.error('Error loading priority dashboard:', error);
    }
}

// Export functions for global use
window.switchPriorityCategory = switchPriorityCategory;
window.loadPriorityDashboard = loadPriorityDashboard;