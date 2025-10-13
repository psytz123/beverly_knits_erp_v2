/**
 * NavigationManager - Navigation state and route management
 *
 * Manages navigation state, active routes, and history for multi-page dashboard.
 * Provides centralized navigation control with accessibility support.
 *
 * @module NavigationManager
 */

export class NavigationManager {
    /**
     * Create NavigationManager instance
     */
    constructor() {
        /** @type {string} Current active route */
        this.currentRoute = this.getCurrentRoute();

        /** @type {Array<string>} Navigation history */
        this.history = [this.currentRoute];

        /** @type {number} Maximum history length */
        this.maxHistoryLength = 50;

        /** @type {Object<string, Function>} Route change listeners */
        this.listeners = {};

        /** @type {Array} Available navigation items */
        this.navItems = [
            { path: '/', label: 'Home', icon: '🏠', ariaLabel: 'Navigate to Dashboard Home' },
            { path: '/metrics', label: 'Metrics', icon: '📊', ariaLabel: 'Navigate to Quality Metrics' },
            { path: '/gates', label: 'Gates', icon: '🚪', ariaLabel: 'Navigate to Phase Gates' },
            { path: '/reuse', label: 'Reuse Analysis', icon: '🔄', ariaLabel: 'Navigate to Reuse Analysis' },
            { path: '/agents', label: 'Agents', icon: '🤖', ariaLabel: 'Navigate to AI Agents' }
        ];

        this.initializeNavigation();
    }

    /**
     * Initialize navigation system
     * Sets up event listeners and active states
     *
     * @private
     */
    initializeNavigation() {
        // Update active link highlighting
        this.updateActiveLinks();

        // Listen to browser navigation (back/forward)
        window.addEventListener('popstate', () => {
            this.currentRoute = this.getCurrentRoute();
            this.updateActiveLinks();
            this.notifyRouteChange(this.currentRoute);
        });

        console.log('[NavigationManager] Initialized with route:', this.currentRoute);
    }

    /**
     * Get current route from window location
     *
     * @private
     * @returns {string} - Current route path
     */
    getCurrentRoute() {
        return window.location.pathname || '/';
    }

    /**
     * Navigate to a new route
     * Updates history and triggers route change
     *
     * @param {string} route - Target route path
     * @param {Object} [options] - Navigation options
     * @param {boolean} [options.replace=false] - Replace current history entry
     * @param {Object} [options.state={}] - State object to store with history
     */
    navigateTo(route, options = {}) {
        const { replace = false, state = {} } = options;

        // Prevent unnecessary navigation
        if (route === this.currentRoute && !replace) {
            console.log('[NavigationManager] Already on route:', route);
            return;
        }

        // Update browser history
        if (replace) {
            window.history.replaceState(state, '', route);
        } else {
            window.history.pushState(state, '', route);
            this.addToHistory(route);
        }

        // Update current route
        this.currentRoute = route;

        // Update UI
        this.updateActiveLinks();

        // Notify listeners
        this.notifyRouteChange(route);

        // Announce to screen readers
        this.announceNavigation(route);

        console.log('[NavigationManager] Navigated to:', route);
    }

    /**
     * Add route to navigation history
     *
     * @private
     * @param {string} route - Route to add to history
     */
    addToHistory(route) {
        this.history.push(route);

        // Trim history if it exceeds max length
        if (this.history.length > this.maxHistoryLength) {
            this.history = this.history.slice(-this.maxHistoryLength);
        }
    }

    /**
     * Go back to previous route
     */
    goBack() {
        window.history.back();
    }

    /**
     * Go forward to next route
     */
    goForward() {
        window.history.forward();
    }

    /**
     * Update active link highlighting
     * Adds 'active' class to current route link
     *
     * @private
     */
    updateActiveLinks() {
        const navLinks = document.querySelectorAll('[data-nav-link]');

        navLinks.forEach(link => {
            const linkPath = link.getAttribute('href') || link.getAttribute('data-route');
            const isActive = linkPath === this.currentRoute;

            // Update active class
            if (isActive) {
                link.classList.add('active');
                link.setAttribute('aria-current', 'page');
            } else {
                link.classList.remove('active');
                link.removeAttribute('aria-current');
            }
        });

        console.log('[NavigationManager] Updated active links for:', this.currentRoute);
    }

    /**
     * Subscribe to route changes
     *
     * @param {string} listenerId - Unique listener ID
     * @param {Function} callback - Callback function (route) => void
     */
    onRouteChange(listenerId, callback) {
        if (typeof callback !== 'function') {
            throw new TypeError('Callback must be a function');
        }

        this.listeners[listenerId] = callback;
        console.log(`[NavigationManager] Listener registered: ${listenerId}`);
    }

    /**
     * Unsubscribe from route changes
     *
     * @param {string} listenerId - Listener ID to remove
     */
    offRouteChange(listenerId) {
        delete this.listeners[listenerId];
        console.log(`[NavigationManager] Listener removed: ${listenerId}`);
    }

    /**
     * Notify all listeners of route change
     *
     * @private
     * @param {string} route - New route
     */
    notifyRouteChange(route) {
        Object.entries(this.listeners).forEach(([id, callback]) => {
            try {
                callback(route);
            } catch (error) {
                console.error(`[NavigationManager] Error in listener ${id}:`, error);
            }
        });
    }

    /**
     * Get navigation items
     *
     * @returns {Array} - Navigation items configuration
     */
    getNavItems() {
        return this.navItems;
    }

    /**
     * Get current route information
     *
     * @returns {Object} - Current route details
     */
    getCurrentRouteInfo() {
        const navItem = this.navItems.find(item => item.path === this.currentRoute);

        return {
            path: this.currentRoute,
            label: navItem?.label || 'Unknown',
            icon: navItem?.icon || '📄',
            ariaLabel: navItem?.ariaLabel || 'Current page'
        };
    }

    /**
     * Get navigation history
     *
     * @param {number} [limit=10] - Maximum number of history items
     * @returns {Array<string>} - Recent navigation history
     */
    getHistory(limit = 10) {
        return this.history.slice(-limit).reverse();
    }

    /**
     * Check if on specific route
     *
     * @param {string} route - Route to check
     * @returns {boolean} - True if on specified route
     */
    isOnRoute(route) {
        return this.currentRoute === route;
    }

    /**
     * Check if route is valid
     *
     * @param {string} route - Route to validate
     * @returns {boolean} - True if route exists in nav items
     */
    isValidRoute(route) {
        return this.navItems.some(item => item.path === route);
    }

    /**
     * Announce navigation to screen readers
     *
     * @private
     * @param {string} route - Route being navigated to
     */
    announceNavigation(route) {
        const routeInfo = this.navItems.find(item => item.path === route);
        const message = routeInfo
            ? `Navigated to ${routeInfo.label}`
            : `Navigated to ${route}`;

        const announcer = document.createElement('div');
        announcer.setAttribute('role', 'status');
        announcer.setAttribute('aria-live', 'polite');
        announcer.setAttribute('aria-atomic', 'true');
        announcer.className = 'sr-only';
        announcer.textContent = message;

        document.body.appendChild(announcer);

        // Remove after announcement
        setTimeout(() => announcer.remove(), 1000);
    }

    /**
     * Setup keyboard navigation
     * Enables arrow key navigation between pages
     */
    setupKeyboardNavigation() {
        document.addEventListener('keydown', (event) => {
            // Only handle if no input is focused
            if (document.activeElement?.tagName === 'INPUT' ||
                document.activeElement?.tagName === 'TEXTAREA') {
                return;
            }

            // Alt + Arrow Left: Go back
            if (event.altKey && event.key === 'ArrowLeft') {
                event.preventDefault();
                this.goBack();
            }

            // Alt + Arrow Right: Go forward
            if (event.altKey && event.key === 'ArrowRight') {
                event.preventDefault();
                this.goForward();
            }
        });

        console.log('[NavigationManager] Keyboard navigation enabled');
    }

    /**
     * Clear navigation history
     */
    clearHistory() {
        this.history = [this.currentRoute];
        console.log('[NavigationManager] History cleared');
    }

    /**
     * Get breadcrumb trail based on current route
     *
     * @returns {Array<Object>} - Breadcrumb items
     */
    getBreadcrumbs() {
        const breadcrumbs = [
            { path: '/', label: 'Home', isActive: false }
        ];

        if (this.currentRoute !== '/') {
            const currentRouteInfo = this.getCurrentRouteInfo();
            breadcrumbs.push({
                path: this.currentRoute,
                label: currentRouteInfo.label,
                isActive: true
            });
        } else {
            breadcrumbs[0].isActive = true;
        }

        return breadcrumbs;
    }
}

// Export singleton instance
export const navigation = new NavigationManager();
