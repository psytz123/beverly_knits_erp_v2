// ============================================
// UNIVERSAL PAGINATION SYSTEM FOR ALL TABLES
// ============================================

// Pagination state for all tables
const tablePagination = {
    currentShortages: {
        data: [],
        currentPage: 1,
        pageSize: 25,
        tableBodyId: 'currentShortagesTableBody',
        paginationId: 'shortagesPagination'
    },
    forecastShortages: {
        data: [],
        currentPage: 1,
        pageSize: 25,
        tableBodyId: 'forecastShortagesTableBody',
        paginationId: 'forecastPagination'
    },
    yarnAlternatives: {
        data: [],
        currentPage: 1,
        pageSize: 25,
        tableBodyId: 'yarnAlternativesTableBody',
        paginationId: 'alternativesPagination'
    },
    productForecast: {
        data: [],
        currentPage: 1,
        pageSize: 25,
        tableBodyId: 'productForecastTableBody',
        paginationId: 'productForecastPagination'
    },
    fabricForecast: {
        data: [],
        currentPage: 1,
        pageSize: 25,
        tableBodyId: 'fabricForecastTableBody',
        paginationId: 'fabricPagination'
    },
    supplier: {
        data: [],
        currentPage: 1,
        pageSize: 25,
        tableBodyId: 'supplierTableBody',
        paginationId: 'supplierPagination'
    }
};

// Universal pagination display function
function displayTableWithPagination(tableName, data, renderRowFunction) {
    const config = tablePagination[tableName];
    if (!config) return;
    
    // Store the data
    config.data = data || [];
    
    // Get table body element
    const tbody = document.getElementById(config.tableBodyId);
    if (!tbody) return;
    
    // Calculate pagination
    const pageSize = config.pageSize === 'all' ? config.data.length : parseInt(config.pageSize);
    const startIndex = (config.currentPage - 1) * pageSize;
    const endIndex = Math.min(startIndex + pageSize, config.data.length);
    const pageData = config.data.slice(startIndex, endIndex);
    
    // Render rows
    let html = '';
    pageData.forEach(item => {
        html += renderRowFunction(item);
    });
    
    tbody.innerHTML = html || '<tr><td colspan="10" class="text-center py-4 text-gray-500">No data available</td></tr>';
    
    // Update pagination controls
    updatePaginationControls(tableName);
}

// Update pagination controls
function updatePaginationControls(tableName) {
    const config = tablePagination[tableName];
    if (!config) return;
    
    const paginationDiv = document.getElementById(config.paginationId);
    if (!paginationDiv) return;
    
    const pageSize = config.pageSize === 'all' ? config.data.length : parseInt(config.pageSize);
    const totalPages = Math.ceil(config.data.length / pageSize) || 1;
    const startIndex = (config.currentPage - 1) * pageSize;
    const endIndex = Math.min(startIndex + pageSize, config.data.length);
    
    let html = `
        <div class="flex flex-wrap justify-between items-center mt-4 bg-gray-50 p-4 rounded-lg">
            <div class="flex items-center gap-4 mb-2 sm:mb-0">
                <label class="text-sm text-gray-600">Show:</label>
                <select id="${tableName}PageSize" class="px-3 py-1 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                        onchange="changeTablePageSize('${tableName}', this.value)">
                    <option value="10" ${config.pageSize == 10 ? 'selected' : ''}>10</option>
                    <option value="25" ${config.pageSize == 25 ? 'selected' : ''}>25</option>
                    <option value="50" ${config.pageSize == 50 ? 'selected' : ''}>50</option>
                    <option value="100" ${config.pageSize == 100 ? 'selected' : ''}>100</option>
                    <option value="all" ${config.pageSize === 'all' ? 'selected' : ''}>All</option>
                </select>
                <span class="text-sm text-gray-600">entries</span>
            </div>
            
            <div class="flex items-center gap-2">
                <button onclick="changeTablePage('${tableName}', 'prev')" 
                        class="px-3 py-1 bg-white border rounded-lg hover:bg-gray-100 disabled:opacity-50 disabled:cursor-not-allowed"
                        ${config.currentPage <= 1 ? 'disabled' : ''}>
                    <i class="fas fa-chevron-left"></i>
                </button>
                <div class="flex gap-1">`;
    
    // Add page numbers
    const maxButtons = 5;
    let startPage = Math.max(1, config.currentPage - Math.floor(maxButtons / 2));
    let endPage = Math.min(totalPages, startPage + maxButtons - 1);
    
    if (endPage - startPage < maxButtons - 1) {
        startPage = Math.max(1, endPage - maxButtons + 1);
    }
    
    for (let i = startPage; i <= endPage; i++) {
        const activeClass = i === config.currentPage ? 
            'bg-blue-500 text-white' : 
            'bg-white text-gray-700 hover:bg-gray-100';
        html += `
            <button onclick="changeTablePage('${tableName}', ${i})" 
                    class="px-3 py-1 border rounded-lg ${activeClass}">
                ${i}
            </button>
        `;
    }
    
    html += `
                </div>
                <button onclick="changeTablePage('${tableName}', 'next')" 
                        class="px-3 py-1 bg-white border rounded-lg hover:bg-gray-100 disabled:opacity-50 disabled:cursor-not-allowed"
                        ${config.currentPage >= totalPages ? 'disabled' : ''}>
                    <i class="fas fa-chevron-right"></i>
                </button>
            </div>
            
            <div class="text-sm text-gray-600">
                Showing ${config.data.length > 0 ? startIndex + 1 : 0} to ${endIndex} of ${config.data.length} entries
            </div>
        </div>`;
    
    paginationDiv.innerHTML = html;
}

// Change page
function changeTablePage(tableName, action) {
    const config = tablePagination[tableName];
    if (!config) return;
    
    const pageSize = config.pageSize === 'all' ? config.data.length : parseInt(config.pageSize);
    const totalPages = Math.ceil(config.data.length / pageSize);
    
    if (action === 'prev' && config.currentPage > 1) {
        config.currentPage--;
    } else if (action === 'next' && config.currentPage < totalPages) {
        config.currentPage++;
    } else if (typeof action === 'number') {
        config.currentPage = Math.max(1, Math.min(action, totalPages));
    }
    
    // Re-render the table
    refreshTable(tableName);
}

// Change page size
function changeTablePageSize(tableName, newSize) {
    const config = tablePagination[tableName];
    if (!config) return;
    
    config.pageSize = newSize === 'all' ? 'all' : parseInt(newSize);
    config.currentPage = 1; // Reset to first page
    
    // Re-render the table
    refreshTable(tableName);
}

// Refresh table display
function refreshTable(tableName) {
    const renderFunctions = {
        currentShortages: renderCurrentShortageRow,
        forecastShortages: renderForecastShortageRow,
        yarnAlternatives: renderYarnAlternativeRow,
        productForecast: renderProductForecastRow,
        fabricForecast: renderFabricForecastRow,
        supplier: renderSupplierRow
    };
    
    const config = tablePagination[tableName];
    const renderFunction = renderFunctions[tableName];
    
    if (config && renderFunction && config.data) {
        displayTableWithPagination(tableName, config.data, renderFunction);
    }
}

// Row render functions
function renderCurrentShortageRow(item) {
    const urgencyColor = item.urgency === 'Critical' ? 'red' : 
                        item.urgency === 'High' ? 'orange' : 'yellow';
    
    return `
        <tr class="hover:bg-gray-50">
            <td class="px-4 py-3 text-sm font-medium text-blue-600">${item.yarn_id || ''}</td>
            <td class="px-4 py-3 text-sm">${item.description || ''}</td>
            <td class="px-4 py-3 text-sm text-right">${(item.required || 0).toLocaleString()} lbs</td>
            <td class="px-4 py-3 text-sm text-right">${(item.available || 0).toLocaleString()} lbs</td>
            <td class="px-4 py-3 text-sm text-right font-bold text-red-600">
                ${(item.shortage || 0).toLocaleString()} lbs
            </td>
            <td class="px-4 py-3 text-sm">${item.affected_orders || 'N/A'}</td>
            <td class="px-4 py-3 text-sm">
                <span class="px-2 py-1 text-xs font-semibold rounded-full bg-${urgencyColor}-100 text-${urgencyColor}-800">
                    ${item.urgency || 'Medium'}
                </span>
            </td>
        </tr>
    `;
}

function renderForecastShortageRow(item) {
    const urgencyColor = item.urgency === 'Critical' ? 'red' : 
                        item.urgency === 'High' ? 'orange' : 'yellow';
    
    return `
        <tr class="hover:bg-gray-50">
            <td class="px-4 py-3 text-sm font-medium text-purple-600">${item.yarn_id || ''}</td>
            <td class="px-4 py-3 text-sm">${item.description || ''}</td>
            <td class="px-4 py-3 text-sm text-right">${(item.forecast_demand || 0).toLocaleString()} lbs</td>
            <td class="px-4 py-3 text-sm text-right">${(item.current_inventory || 0).toLocaleString()} lbs</td>
            <td class="px-4 py-3 text-sm text-right">${(item.on_order || 0).toLocaleString()} lbs</td>
            <td class="px-4 py-3 text-sm text-right font-bold text-orange-600">
                ${(item.shortage || 0).toLocaleString()} lbs
            </td>
            <td class="px-4 py-3 text-sm">${item.shortage_date || 'TBD'}</td>
            <td class="px-4 py-3 text-sm">
                <span class="px-2 py-1 text-xs font-semibold rounded-full bg-${urgencyColor}-100 text-${urgencyColor}-800">
                    ${item.urgency || 'Medium'}
                </span>
            </td>
        </tr>
    `;
}

function renderYarnAlternativeRow(item) {
    const confidenceWidth = Math.min(100, Math.max(0, item.confidence || 0));
    
    return `
        <tr class="hover:bg-gray-50">
            <td class="px-4 py-3 text-sm font-medium text-red-600">${item.original_yarn || ''}</td>
            <td class="px-4 py-3 text-sm font-medium text-green-600">${item.alternative_yarn || ''}</td>
            <td class="px-4 py-3 text-sm text-right">${(item.available_qty || 0).toLocaleString()} lbs</td>
            <td class="px-4 py-3 text-sm">${item.compatibility || 'Good'}</td>
            <td class="px-4 py-3 text-sm text-center">
                <div class="flex items-center justify-center">
                    <div class="w-full bg-gray-200 rounded-full h-2 max-w-[100px]">
                        <div class="bg-blue-600 h-2 rounded-full" style="width: ${confidenceWidth}%"></div>
                    </div>
                    <span class="ml-2 text-xs font-medium">${item.confidence || 0}%</span>
                </div>
            </td>
            <td class="px-4 py-3 text-sm text-center">${item.success_rate || 'N/A'}</td>
            <td class="px-4 py-3 text-sm">${item.supplier || 'Various'}</td>
            <td class="px-4 py-3 text-sm">
                <button class="px-3 py-1 text-xs bg-green-500 text-white rounded hover:bg-green-600">
                    Use Alternative
                </button>
            </td>
        </tr>
    `;
}

function renderProductForecastRow(item) {
    const trendIcon = item.trend === 'Up' ? '↑' : item.trend === 'Down' ? '↓' : '→';
    const trendColor = item.trend === 'Up' ? 'text-green-600' : item.trend === 'Down' ? 'text-red-600' : 'text-gray-600';
    
    return `
        <tr class="hover:bg-gray-50">
            <td class="px-4 py-3 text-sm font-medium">${item.style || ''}</td>
            <td class="px-4 py-3 text-sm">${item.description || ''}</td>
            <td class="px-4 py-3 text-sm text-right">${(item.forecast_30 || 0).toLocaleString()}</td>
            <td class="px-4 py-3 text-sm text-right">${(item.forecast_60 || 0).toLocaleString()}</td>
            <td class="px-4 py-3 text-sm text-right">${(item.forecast_90 || 0).toLocaleString()}</td>
            <td class="px-4 py-3 text-sm text-center">
                <div class="flex items-center justify-center">
                    <div class="w-full bg-gray-200 rounded-full h-2 max-w-[60px]">
                        <div class="bg-green-500 h-2 rounded-full" style="width: ${item.confidence || 0}%"></div>
                    </div>
                    <span class="ml-2 text-xs">${item.confidence || '0'}%</span>
                </div>
            </td>
            <td class="px-4 py-3 text-sm ${trendColor}">
                <span class="font-bold">${trendIcon}</span> ${item.trend || 'Stable'}
            </td>
            <td class="px-4 py-3 text-sm">${item.seasonality || 'None'}</td>
        </tr>
    `;
}

function renderFabricForecastRow(item) {
    const statusColor = item.status === 'Critical' ? 'red' : 
                       item.status === 'Warning' ? 'yellow' : 'green';
    const netPositionColor = (item.net_position || 0) < 0 ? 'text-red-600' : 'text-green-600';
    
    return `
        <tr class="hover:bg-gray-50">
            <td class="px-4 py-3 text-sm font-medium">${item.style || ''}</td>
            <td class="px-4 py-3 text-sm">${item.fabric_type || ''}</td>
            <td class="px-4 py-3 text-sm">${item.description || ''}</td>
            <td class="px-4 py-3 text-sm text-right">${(item.forecasted_qty || 0).toLocaleString()}</td>
            <td class="px-4 py-3 text-sm text-right">${(item.current_inventory || 0).toLocaleString()}</td>
            <td class="px-4 py-3 text-sm text-right">${(item.on_order || 0).toLocaleString()}</td>
            <td class="px-4 py-3 text-sm text-right font-bold ${netPositionColor}">
                ${(item.net_position || 0).toLocaleString()}
            </td>
            <td class="px-4 py-3 text-sm">${item.lead_time || 'N/A'}</td>
            <td class="px-4 py-3 text-sm">${item.target_date || 'TBD'}</td>
            <td class="px-4 py-3 text-sm">
                <span class="px-2 py-1 text-xs font-semibold rounded-full bg-${statusColor}-100 text-${statusColor}-800">
                    ${item.status || 'OK'}
                </span>
            </td>
        </tr>
    `;
}

function renderSupplierRow(item) {
    const statusColor = item.status === 'Active' ? 'green' : 
                       item.status === 'Warning' ? 'yellow' : 'gray';
    const reliabilityColor = item.reliability >= 90 ? 'text-green-600' : 
                            item.reliability >= 70 ? 'text-yellow-600' : 'text-red-600';
    
    return `
        <tr class="hover:bg-gray-50">
            <td class="px-4 py-3 text-sm font-medium">${item.supplier_name || ''}</td>
            <td class="px-4 py-3 text-sm">${item.yarn_types || ''}</td>
            <td class="px-4 py-3 text-sm text-right">${(item.current_orders || 0).toLocaleString()}</td>
            <td class="px-4 py-3 text-sm text-right">${(item.pending_deliveries || 0).toLocaleString()} lbs</td>
            <td class="px-4 py-3 text-sm">${item.lead_time || 'N/A'}</td>
            <td class="px-4 py-3 text-sm text-center ${reliabilityColor}">
                <span class="font-bold">${item.reliability || '0'}%</span>
            </td>
            <td class="px-4 py-3 text-sm">
                <span class="px-2 py-1 text-xs font-semibold rounded-full bg-${statusColor}-100 text-${statusColor}-800">
                    ${item.status || 'Unknown'}
                </span>
            </td>
            <td class="px-4 py-3 text-sm">
                <button class="text-blue-600 hover:text-blue-800">
                    <i class="fas fa-eye"></i>
                </button>
            </td>
        </tr>
    `;
}

// Export functions for global use
window.tablePagination = tablePagination;
window.displayTableWithPagination = displayTableWithPagination;
window.changeTablePage = changeTablePage;
window.changeTablePageSize = changeTablePageSize;
window.renderCurrentShortageRow = renderCurrentShortageRow;
window.renderForecastShortageRow = renderForecastShortageRow;
window.renderYarnAlternativeRow = renderYarnAlternativeRow;
window.renderProductForecastRow = renderProductForecastRow;
window.renderFabricForecastRow = renderFabricForecastRow;
window.renderSupplierRow = renderSupplierRow;