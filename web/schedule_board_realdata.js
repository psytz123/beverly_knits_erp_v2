// Real Data Schedule Board Fix
console.log('Loading Real Data Schedule Board...');

// Override the entire schedule rendering system
window.loadScheduleBoardData = async function() {
    const container = document.getElementById('scheduleGrid');
    if (!container) {
        console.error('Schedule grid container not found');
        return;
    }
    
    // Show loading state
    container.innerHTML = '<div class="text-gray-400 text-center py-8"><i class="fas fa-spinner fa-spin text-2xl mb-2"></i><br>Loading real production data...</div>';
    
    try {
        console.log('Fetching real production data...');
        
        // Get the base URL properly
        const baseUrl = window.API_BASE_URL || 'http://localhost:5006';
        
        // Fetch real production planning data
        const response = await fetch(baseUrl + '/api/production-planning');
        const data = await response.json();
        
        console.log('Production data received:', data);
        
        if (data && data.production_schedule) {
            renderRealProductionSchedule(data.production_schedule);
        } else {
            container.innerHTML = '<div class="text-red-400 text-center py-8">No production schedule found</div>';
        }
    } catch (error) {
        console.error('Error loading production data:', error);
        container.innerHTML = '<div class="text-red-400 text-center py-8">Error: ' + error.message + '</div>';
    }
};

function renderRealProductionSchedule(schedule) {
    const container = document.getElementById('scheduleGrid');
    if (!container) return;
    
    container.innerHTML = '';
    
    // Group orders by machine
    const machineGroups = {};
    
    // First add unassigned orders
    machineGroups['Unassigned'] = [];
    
    schedule.forEach(order => {
        const machineId = order.machine || 'Unassigned';
        if (!machineGroups[machineId]) {
            machineGroups[machineId] = [];
        }
        machineGroups[machineId].push(order);
    });
    
    console.log('Grouped orders by machine:', machineGroups);
    
    // Sort machines: Unassigned first, then numeric order
    const sortedMachines = Object.keys(machineGroups).sort((a, b) => {
        if (a === 'Unassigned') return -1;
        if (b === 'Unassigned') return 1;
        // Convert to numbers for proper sorting
        const numA = parseFloat(a);
        const numB = parseFloat(b);
        return numA - numB;
    });
    
    // Create rows for each machine
    let rowIndex = 0;
    sortedMachines.forEach(machineId => {
        const orders = machineGroups[machineId];
        if (!orders || orders.length === 0) return;
        
        if (rowIndex >= 15) return; // Limit display
        
        const row = createMachineScheduleRow(machineId, orders, rowIndex);
        container.appendChild(row);
        rowIndex++;
    });
    
    if (rowIndex === 0) {
        container.innerHTML = '<div class="text-gray-400 text-center py-8">No production orders found</div>';
    }
}

function createMachineScheduleRow(machineId, orders, index) {
    const row = document.createElement('div');
    row.className = 'flex items-center hover:bg-gray-800 rounded p-1 transition-colors mb-2';
    
    // Calculate total quantity
    const totalQty = orders.reduce((sum, o) => sum + (parseFloat(o.quantity_lbs) || 0), 0);
    
    // Machine label (left side)
    const label = document.createElement('div');
    label.className = 'w-40 text-sm text-gray-300 pr-4 flex items-center justify-between';
    
    if (machineId === 'Unassigned') {
        label.innerHTML = `
            <span class="font-medium text-yellow-400">Unassigned</span>
            <span class="text-xs text-gray-500">${orders.length} orders</span>
        `;
    } else {
        // Use actual machine ID
        label.innerHTML = `
            <span class="font-medium">Machine ${machineId}</span>
            <span class="text-xs text-gray-500">${machineId}</span>
        `;
    }
    
    row.appendChild(label);
    
    // Timeline container
    const timeline = document.createElement('div');
    timeline.className = 'flex-1 relative h-10 bg-gray-800 bg-opacity-50 rounded';
    timeline.style.minWidth = '900px';
    
    // Add vertical grid lines for days
    const gridContainer = document.createElement('div');
    gridContainer.className = 'absolute inset-0 flex pointer-events-none';
    for (let i = 0; i < 14; i++) {
        const gridLine = document.createElement('div');
        gridLine.className = 'flex-1 border-r border-gray-700 border-opacity-30';
        gridContainer.appendChild(gridLine);
    }
    timeline.appendChild(gridContainer);
    
    // Add order bars
    orders.forEach(order => {
        const bar = createRealOrderBar(order, machineId === 'Unassigned');
        if (bar) timeline.appendChild(bar);
    });
    
    row.appendChild(timeline);
    return row;
}

function createRealOrderBar(order, isUnassigned) {
    const bar = document.createElement('div');
    
    // Get dates
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    
    let startDate = today; // Default
    if (order.start_date) {
        // Parse various date formats
        if (typeof order.start_date === 'string') {
            if (order.start_date.includes('GMT')) {
                startDate = new Date(order.start_date);
            } else if (order.start_date.includes('/')) {
                startDate = new Date(order.start_date);
            } else {
                startDate = new Date(order.start_date.replace(/-/g, '/'));
            }
        }
    }
    
    // Calculate position (days from today)
    const daysDiff = Math.floor((startDate - today) / (1000 * 60 * 60 * 24));
    
    // Skip if too far in the future
    if (daysDiff > 14) return null;
    
    // Position on timeline (14 days = 100%)
    const leftPos = Math.max(0, (daysDiff / 14) * 100);
    const width = Math.max(5, (1 / 14) * 100); // Minimum width for visibility
    
    // Determine color based on status
    let bgColor = 'bg-green-500'; // Default green for "Start"
    let textColor = 'text-white';
    
    if (isUnassigned) {
        bgColor = 'bg-gray-500';
    } else if (daysDiff < 0) {
        bgColor = 'bg-red-500'; // Overdue
    } else if (order.priority === 'High') {
        bgColor = 'bg-orange-500'; // Due
    } else if (order.priority === 'Normal') {
        bgColor = 'bg-green-500'; // Start
    }
    
    // Get the actual order code or style
    const orderCode = order.order_id || order.style || 'Order';
    
    bar.className = `absolute ${bgColor} rounded cursor-pointer hover:z-20 hover:shadow-lg transition-all duration-200 hover:scale-105`;
    bar.style.left = `${leftPos}%`;
    bar.style.width = `${width * 3}%`; // Make bars wider for visibility
    bar.style.top = '20%';
    bar.style.height = '60%';
    bar.style.minWidth = '80px';
    
    // Display actual order information
    bar.innerHTML = `
        <div class="h-full flex items-center justify-center px-2">
            <span class="${textColor} text-xs font-bold whitespace-nowrap">
                ${orderCode}
            </span>
        </div>
    `;
    
    // Tooltip with real data
    bar.title = `
Order: ${order.order_id || 'N/A'}
Style: ${order.style || 'N/A'}
Machine: ${order.machine || 'Unassigned'}
Quantity: ${parseFloat(order.quantity_lbs || 0).toLocaleString()} lbs
Priority: ${order.priority || 'Normal'}
Status: ${order.status || 'Scheduled'}
Start: ${startDate.toLocaleDateString()}
Customer: ${order.customer || 'Unknown'}
${daysDiff < 0 ? '⚠️ OVERDUE' : ''}
    `.trim();
    
    // Click handler
    bar.onclick = () => {
        console.log('Order details:', order);
        showOrderDetailsModal(order);
    };
    
    return bar;
}

function showOrderDetailsModal(order) {
    // You can replace this with a proper modal
    const details = `
Production Order Details
========================
Order ID: ${order.order_id || 'N/A'}
Style: ${order.style || 'N/A'}
Machine: ${order.machine || 'Unassigned'}
Quantity: ${parseFloat(order.quantity_lbs || 0).toLocaleString()} lbs
Priority: ${order.priority || 'Normal'}
Status: ${order.status || 'Scheduled'}
Customer: ${order.customer || 'Unknown'}

Click OK to close.
    `;
    alert(details);
}

// Auto-trigger when switching to schedule view
const originalSwitch = window.switchPlanningView;
window.switchPlanningView = function(view) {
    if (originalSwitch) originalSwitch(view);
    
    if (view === 'schedule') {
        console.log('Loading real production data for schedule view...');
        setTimeout(() => {
            loadScheduleBoardData();
        }, 100);
    }
};

// Update the date header
function updateDateHeaders() {
    const container = document.querySelector('#scheduleView .grid.grid-cols-14');
    if (!container) return;
    
    container.innerHTML = '';
    const today = new Date();
    
    for (let i = 0; i < 14; i++) {
        const date = new Date(today);
        date.setDate(today.getDate() + i);
        
        const div = document.createElement('div');
        div.className = i === 0 ? 'text-center text-xs text-blue-400' : 'text-center text-xs';
        div.innerHTML = `
            ${date.toLocaleDateString('en-US', { weekday: 'short' })}<br>
            ${date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
        `;
        container.appendChild(div);
    }
}

console.log('Real Data Schedule Board loaded! Switch to Schedule Board view to see actual production data.');