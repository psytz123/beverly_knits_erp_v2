// Fix for Schedule Board to use real production data
console.log('Loading Schedule Board Fix...');

// Override the loadScheduleBoardData function to use real production data
window.loadScheduleBoardData = async function() {
    const container = document.getElementById('scheduleGrid');
    if (!container) {
        console.error('Schedule grid container not found');
        return;
    }
    
    // Show loading state
    container.innerHTML = '<div class="text-gray-400 text-center py-8"><i class="fas fa-spinner fa-spin text-2xl mb-2"></i><br>Loading real production schedule...</div>';
    
    try {
        console.log('Fetching production planning data for schedule board...');
        
        // Fetch production planning data which has actual schedules with dates
        const [planningResponse, factoryResponse] = await Promise.all([
            fetch(API_BASE_URL + '/api/production-planning'),
            fetch(API_BASE_URL + '/api/factory-floor-ai-dashboard')
        ]);
        
        const planningData = await planningResponse.json();
        const factoryData = await factoryResponse.json();
        
        console.log('Production planning data loaded:', planningData);
        console.log('Factory data loaded:', factoryData);
        
        if (planningData && planningData.production_schedule) {
            window.renderScheduleBoardWithRealData(planningData, factoryData);
        } else {
            console.error('No production schedule found in data');
            container.innerHTML = '<div class="text-red-400 text-center py-8">No production schedule data available</div>';
        }
    } catch (error) {
        console.error('Error loading schedule board:', error);
        container.innerHTML = '<div class="text-red-400 text-center py-8">Error loading schedule data: ' + error.message + '</div>';
    }
};

// New function to render schedule board with real production data
window.renderScheduleBoardWithRealData = function(planningData, factoryData) {
    const container = document.getElementById('scheduleGrid');
    if (!container) return;
    
    container.innerHTML = '';
    
    // Get production schedule from planning data
    const schedule = planningData.production_schedule || [];
    console.log('Processing', schedule.length, 'production orders for schedule board');
    
    // Group orders by machine
    const machineOrders = {};
    schedule.forEach(order => {
        const machine = order.machine || 'Unassigned';
        if (!machineOrders[machine]) {
            machineOrders[machine] = [];
        }
        machineOrders[machine].push(order);
    });
    
    // Get work center mappings from factory data
    const machineToWC = {};
    if (factoryData && factoryData.work_center_groups) {
        factoryData.work_center_groups.forEach(wc => {
            (wc.machines || []).forEach(m => {
                machineToWC[m.machine_id] = wc.work_center_id;
            });
        });
    }
    
    // Sort machines: Unassigned first, then by machine ID
    const sortedMachines = Object.keys(machineOrders).sort((a, b) => {
        if (a === 'Unassigned') return -1;
        if (b === 'Unassigned') return 1;
        return parseFloat(a) - parseFloat(b);
    });
    
    // Render rows for each machine (limit to 15 for display)
    let rowCount = 0;
    sortedMachines.forEach(machineId => {
        if (rowCount >= 15) return;
        
        const orders = machineOrders[machineId];
        const workCenter = machineToWC[machineId] || '';
        const row = createRealScheduleRow(machineId, orders, workCenter, rowCount);
        container.appendChild(row);
        rowCount++;
    });
    
    // Show message if no data
    if (rowCount === 0) {
        container.innerHTML = '<div class="text-gray-400 text-center py-8">No scheduled production orders</div>';
    } else {
        console.log('Successfully rendered', rowCount, 'machine rows with real data');
    }
};

// Create schedule row with real order data
window.createRealScheduleRow = function(machineId, orders, workCenter, index) {
    const row = document.createElement('div');
    row.className = 'flex items-center hover:bg-gray-800 rounded p-1 transition-colors min-w-max mb-2';
    
    // Machine label
    const label = document.createElement('div');
    label.className = 'w-48 text-sm text-gray-300 pr-4';
    
    const totalQty = orders.reduce((sum, o) => sum + (o.quantity_lbs || 0), 0);
    
    if (machineId === 'Unassigned') {
        label.innerHTML = `
            <div class="flex items-center justify-between">
                <span class="font-medium text-yellow-500">⚠️ Unassigned</span>
                <span class="text-xs text-gray-500">${orders.length} orders</span>
            </div>
        `;
    } else {
        label.innerHTML = `
            <div class="flex items-center justify-between">
                <div>
                    <span class="font-medium">Machine ${machineId}</span>
                    ${workCenter ? `<span class="text-xs text-gray-500 ml-1">WC ${workCenter}</span>` : ''}
                </div>
                <span class="text-xs text-gray-500">${totalQty.toLocaleString()} lbs</span>
            </div>
        `;
    }
    row.appendChild(label);
    
    // Timeline container with grid
    const timeline = document.createElement('div');
    timeline.className = 'flex-1 relative h-10 bg-gray-800 rounded';
    timeline.style.minWidth = '800px';
    
    // Add grid lines for 14 days
    const gridLines = document.createElement('div');
    gridLines.className = 'absolute inset-0 flex';
    for (let i = 0; i < 14; i++) {
        const line = document.createElement('div');
        line.className = 'flex-1 border-r border-gray-700';
        gridLines.appendChild(line);
    }
    timeline.appendChild(gridLines);
    
    // Add order bars
    orders.forEach(order => {
        const bar = createRealOrderBar(order, machineId === 'Unassigned');
        timeline.appendChild(bar);
    });
    
    row.appendChild(timeline);
    return row;
};

// Create order bar with real data
window.createRealOrderBar = function(order, isUnassigned) {
    const bar = document.createElement('div');
    
    // Calculate position based on dates
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    
    // Parse the start date from the order
    let startDate;
    if (order.start_date) {
        // Handle different date formats
        if (order.start_date.includes('GMT')) {
            startDate = new Date(order.start_date);
        } else {
            startDate = new Date(order.start_date.replace(/-/g, '/'));
        }
    } else {
        startDate = new Date(); // Default to today if no start date
    }
    
    const endDate = order.end_date ? new Date(order.end_date.replace(/-/g, '/')) : new Date(startDate.getTime() + 86400000);
    
    const daysDiff = Math.floor((startDate - today) / 86400000);
    const duration = Math.ceil((endDate - startDate) / 86400000) || 1;
    
    // Position and width (14 days = 100%)
    const left = Math.max(0, Math.min(95, (daysDiff / 14) * 100));
    const width = Math.max(3, Math.min(100 - left, (duration / 14) * 100));
    
    // Determine color based on priority and status
    let bgColor = 'bg-blue-500';
    let borderColor = 'border-blue-600';
    
    if (isUnassigned) {
        bgColor = 'bg-gray-600';
        borderColor = 'border-gray-700';
    } else if (daysDiff < 0) {
        bgColor = 'bg-red-500'; // Overdue
        borderColor = 'border-red-600';
    } else if (order.priority === 'High') {
        bgColor = 'bg-orange-500';
        borderColor = 'border-orange-600';
    } else if (order.priority === 'Normal') {
        bgColor = 'bg-yellow-500';
        borderColor = 'border-yellow-600';
    } else if (order.status === 'Scheduled') {
        bgColor = 'bg-green-500';
        borderColor = 'border-green-600';
    }
    
    bar.className = `absolute ${bgColor} border ${borderColor} rounded-sm cursor-pointer hover:z-10 hover:shadow-lg transform hover:scale-105 transition-all`;
    bar.style.left = `${left}%`;
    bar.style.width = `${width}%`;
    bar.style.top = '50%';
    bar.style.transform = 'translateY(-50%)';
    bar.style.height = '28px';
    bar.style.minWidth = '60px';
    
    // Add order info
    bar.innerHTML = `
        <div class="flex items-center justify-center h-full px-1">
            <span class="text-white text-xs font-bold truncate">
                ${order.style || order.order_id || 'Order'}
            </span>
        </div>
    `;
    
    // Add detailed tooltip
    const tooltipText = `
Order: ${order.order_id || 'N/A'}
Style: ${order.style || 'N/A'}
Machine: ${order.machine || 'Unassigned'}
Quantity: ${(order.quantity_lbs || 0).toLocaleString()} lbs
Priority: ${order.priority || 'Normal'}
Status: ${order.status || 'Planned'}
Start: ${startDate.toLocaleDateString()}
End: ${endDate.toLocaleDateString()}
Customer: ${order.customer || 'Unknown'}
${daysDiff < 0 ? '⚠️ OVERDUE by ' + Math.abs(daysDiff) + ' days' : ''}
    `.trim();
    
    bar.title = tooltipText;
    
    // Click handler
    bar.onclick = () => {
        console.log('Order clicked:', order);
        alert(tooltipText);
    };
    
    return bar;
};

// Update timeline header with real dates
window.updateScheduleTimeline = function() {
    const header = document.getElementById('dateHeader');
    if (!header) return;
    
    header.innerHTML = '';
    const today = new Date();
    
    for (let i = 0; i < 14; i++) {
        const date = new Date(today);
        date.setDate(today.getDate() + i);
        
        const day = date.getDate();
        const month = date.toLocaleDateString('en-US', { month: 'short' });
        const weekday = date.toLocaleDateString('en-US', { weekday: 'short' });
        
        const div = document.createElement('div');
        div.className = i === 0 ? 'border-r border-gray-700 bg-blue-900 bg-opacity-30 px-1' : 'border-r border-gray-700 px-1';
        div.innerHTML = `
            <div class="text-xs font-medium">${weekday}</div>
            <div class="text-xs">${month} ${day}</div>
        `;
        header.appendChild(div);
    }
};

// Auto-reload schedule data when switching to schedule view
const originalSwitchView = window.switchPlanningView;
window.switchPlanningView = function(view) {
    if (originalSwitchView) {
        originalSwitchView(view);
    }
    
    if (view === 'schedule') {
        console.log('Switching to schedule view - loading real data...');
        updateScheduleTimeline();
        loadScheduleBoardData();
    }
};

console.log('Schedule Board Fix loaded successfully! Use switchPlanningView("schedule") to view.');