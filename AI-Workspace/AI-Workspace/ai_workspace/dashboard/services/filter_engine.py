"""Filter engine service for AI Workspace Dashboard.

Provides multi-dimensional filtering capabilities for dashboard data including
reuse checks, phase gates, and agent activities with support for filter presets.
"""

import glob
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class FilterEngine:
    """Multi-dimensional filter engine for dashboard data."""

    def __init__(self, workspace_root: Optional[Path] = None) -> None:
        """Initialize filter engine.

        Args:
            workspace_root: Root directory of workspace (defaults to .agent-workspace)
        """
        if workspace_root is None:
            workspace_root = Path.cwd() / '.agent-workspace'

        self.workspace_root = Path(workspace_root)
        self.cache_dir = self.workspace_root / 'cache'
        self.handoffs_dir = self.workspace_root / 'handoffs'

    def get_available_dimensions(self) -> List[Dict[str, Any]]:
        """Get all available filter dimensions.

        Returns:
            List of filter dimension configurations
        """
        return [
            {
                'name': 'date_range',
                'type': 'select',
                'options': [
                    {'value': 'last_hour', 'label': 'Last Hour'},
                    {'value': 'last_day', 'label': 'Last 24 Hours'},
                    {'value': 'last_week', 'label': 'Last 7 Days'},
                ],
                'description': 'Filter by time period'
            },
            {
                'name': 'status',
                'type': 'multiselect',
                'options': ['passed', 'failed', 'in_progress', 'pending', 'blocked'],
                'description': 'Filter by status'
            },
            {
                'name': 'type',
                'type': 'multiselect',
                'options': ['reuse_check', 'phase_gate', 'agent_activity'],
                'description': 'Filter by activity type'
            },
        ]

    def get_presets(self) -> List[Dict[str, Any]]:
        """Get all predefined filter presets.

        Returns:
            List of filter preset configurations
        """
        return [
            {
                'id': 'recent_failures',
                'name': 'Recent Failures',
                'description': 'Items that failed in the last 24 hours',
                'filters': {'status': ['failed'], 'date_range': 'last_day'},
                'operator': 'AND'
            },
            {
                'id': 'high_reuse',
                'name': 'High Reuse Opportunities',
                'description': 'Reuse checks with >85% reuse potential',
                'filters': {'type': ['reuse_check']},
                'operator': 'AND'
            },
            {
                'id': 'active_gates',
                'name': 'Active Phase Gates',
                'description': 'Phase gates currently in progress',
                'filters': {'type': ['phase_gate'], 'status': ['in_progress']},
                'operator': 'AND'
            },
        ]

    def apply_filters(
        self,
        filters: Dict[str, Any],
        operator: str = 'AND',
        data_sources: Optional[List[str]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Apply filters to dashboard data.

        Args:
            filters: Dictionary of filter criteria
            operator: Combination operator ('AND' or 'OR')
            data_sources: List of data sources to filter (defaults to all)

        Returns:
            Dictionary with filtered results by data source
        """
        if data_sources is None:
            data_sources = ['reuse_checks', 'phase_gates']

        results = {}

        if 'reuse_checks' in data_sources:
            results['reuse_checks'] = self._filter_reuse_checks(filters, operator)

        if 'phase_gates' in data_sources:
            results['phase_gates'] = self._filter_phase_gates(filters, operator)

        return results

    def _filter_reuse_checks(
        self,
        filters: Dict[str, Any],
        operator: str
    ) -> List[Dict[str, Any]]:
        """Filter reuse check data.

        Args:
            filters: Filter criteria
            operator: Combination operator

        Returns:
            Filtered reuse checks
        """
        reuse_file = self.cache_dir / 'reuse_checks.json'

        if not reuse_file.exists():
            return []

        try:
            with open(reuse_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            checks = data if isinstance(data, list) else data.get('checks', [])
            return self._apply_filter_logic(checks, filters, operator)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading reuse checks: {e}")
            return []

    def _filter_phase_gates(
        self,
        filters: Dict[str, Any],
        operator: str
    ) -> List[Dict[str, Any]]:
        """Filter phase gate data.

        Args:
            filters: Filter criteria
            operator: Combination operator

        Returns:
            Filtered phase gates
        """
        gates = []
        gate_pattern = str(self.handoffs_dir / 'gate-*.json')

        for gate_file in glob.glob(gate_pattern):
            try:
                with open(gate_file, 'r', encoding='utf-8') as f:
                    gate_data = json.load(f)
                    gates.append(gate_data)
            except (json.JSONDecodeError, IOError):
                continue

        return self._apply_filter_logic(gates, filters, operator)

    def _apply_filter_logic(
        self,
        items: List[Dict[str, Any]],
        filters: Dict[str, Any],
        operator: str
    ) -> List[Dict[str, Any]]:
        """Apply filter logic to items.

        Args:
            items: Items to filter
            filters: Filter criteria
            operator: 'AND' or 'OR'

        Returns:
            Filtered items
        """
        if not filters:
            return items

        filtered = []

        for item in items:
            matches = []

            # Apply each filter criterion
            if 'date_range' in filters:
                matches.append(self._match_date_range(item, filters['date_range']))

            if 'status' in filters:
                matches.append(self._match_status(item, filters['status']))

            if 'type' in filters:
                item_type = self._infer_type(item)
                matches.append(item_type in filters['type'])

            # Apply operator logic
            if not matches:
                filtered.append(item)
            elif operator == 'AND':
                if all(matches):
                    filtered.append(item)
            else:  # OR
                if any(matches):
                    filtered.append(item)

        return filtered

    def _match_date_range(self, item: Dict[str, Any], date_range: str) -> bool:
        """Check if item matches date range filter."""
        timestamp_str = item.get('timestamp') or item.get('created_at') or item.get('updated_at')

        if not timestamp_str:
            return True

        try:
            item_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            return True

        now = datetime.now(item_time.tzinfo) if item_time.tzinfo else datetime.now()

        if date_range == 'last_hour':
            return item_time >= now - timedelta(hours=1)
        elif date_range == 'last_day':
            return item_time >= now - timedelta(days=1)
        elif date_range == 'last_week':
            return item_time >= now - timedelta(days=7)

        return True

    def _match_status(self, item: Dict[str, Any], statuses: List[str]) -> bool:
        """Check if item matches status filter."""
        item_status = item.get('status', '').lower()
        return item_status in [s.lower() for s in statuses]

    def _infer_type(self, item: Dict[str, Any]) -> str:
        """Infer item type from its structure."""
        if 'reuse_percentage' in item or 'files_checked' in item:
            return 'reuse_check'
        elif 'phase' in item or 'gate' in str(item.get('task', '')).lower():
            return 'phase_gate'
        else:
            return 'agent_activity'
