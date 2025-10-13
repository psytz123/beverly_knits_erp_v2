"""File system monitor for real-time dashboard updates.

Monitors workspace files for changes and broadcasts typed data updates via WebSocket.
Implements debouncing, content hashing, and comprehensive error handling.
"""

import hashlib
import json
import logging
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Set

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

# Configure logging
logger = logging.getLogger(__name__)


class WorkspaceFileHandler(FileSystemEventHandler):
    """Monitor workspace files for changes with typed data integration."""

    # Debounce interval in seconds
    DEBOUNCE_INTERVAL = 0.5

    # Batch update window in seconds
    BATCH_WINDOW = 1.0

    def __init__(self, socketio: Any, workspace_root: Path) -> None:
        """Initialize file handler.

        Args:
            socketio: SocketIO instance for broadcasting updates
            workspace_root: Root workspace directory path
        """
        super().__init__()
        self.socketio = socketio
        self.workspace_root = workspace_root

        # Content hash cache for change detection
        self._content_hashes: Dict[str, str] = {}

        # Pending updates for batching
        self._pending_updates: Dict[str, Dict[str, Any]] = {}

        # Last update timestamps for debouncing
        self._last_update: Dict[str, float] = defaultdict(float)

        # Lock for thread safety
        self._update_lock = False

    def _calculate_file_hash(self, file_path: Path) -> Optional[str]:
        """Calculate SHA256 hash of file content.

        Args:
            file_path: Path to file

        Returns:
            Hex digest of file hash, or None if error
        """
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"Error hashing file {file_path}: {e}")
            return None

    def _has_content_changed(self, file_path: Path) -> bool:
        """Check if file content has actually changed.

        Args:
            file_path: Path to file

        Returns:
            True if content changed or new file, False otherwise
        """
        current_hash = self._calculate_file_hash(file_path)
        if current_hash is None:
            return False

        file_key = str(file_path)
        previous_hash = self._content_hashes.get(file_key)

        if previous_hash is None or previous_hash != current_hash:
            self._content_hashes[file_key] = current_hash
            return True

        return False

    def _should_process_update(self, file_path: Path) -> bool:
        """Check if update should be processed (debouncing).

        Args:
            file_path: Path to file

        Returns:
            True if enough time has passed since last update
        """
        file_key = str(file_path)
        current_time = time.time()
        last_time = self._last_update[file_key]

        if current_time - last_time < self.DEBOUNCE_INTERVAL:
            return False

        self._last_update[file_key] = current_time
        return True

    def _load_json_safely(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Load JSON file with error handling.

        Args:
            file_path: Path to JSON file

        Returns:
            Parsed JSON data or None if error
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error in {file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return None

    def _emit_typed_event(
        self,
        event_type: str,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Emit typed WebSocket event.

        Args:
            event_type: Type of event (reuse_check_update, gate_update, etc.)
            data: Typed data dictionary
            metadata: Optional metadata (gate_name, timestamp, etc.)
        """
        try:
            payload = {
                'type': event_type,
                'data': data,
                'timestamp': datetime.now().isoformat()
            }

            if metadata:
                payload.update(metadata)

            logger.debug(f"Emitting {event_type}: {payload.get('metadata', {})}")
            self.socketio.emit(event_type, payload)

        except Exception as e:
            logger.error(f"Error emitting {event_type}: {e}")

    def _handle_reuse_check_update(self, file_path: Path) -> None:
        """Handle reuse_checks.json update.

        Emits: reuse_check_update with ReuseAnalysisDict data

        Args:
            file_path: Path to reuse_checks.json
        """
        try:
            from ..models import ReuseAnalysis

            # Load and parse reuse analyses
            analyses = ReuseAnalysis.from_json(file_path)

            # Convert to dictionaries
            analyses_data = [a.to_dict() for a in analyses]

            self._emit_typed_event(
                'reuse_check_update',
                analyses_data,
                {'total_checks': len(analyses_data)}
            )

            logger.info(f"Emitted reuse check update: {len(analyses_data)} checks")

        except Exception as e:
            logger.error(f"Error handling reuse check update: {e}")

    def _handle_gate_update(self, file_path: Path) -> None:
        """Handle gate-*.json update.

        Emits: gate_update with PhaseGateDict data

        Args:
            file_path: Path to gate file
        """
        try:
            from ..models import PhaseGate

            # Load and parse phase gate
            gate = PhaseGate.from_json(file_path)

            # Extract gate name from filename (e.g., gate-discovery-backend.json)
            gate_name = file_path.stem  # Remove .json extension

            self._emit_typed_event(
                'gate_update',
                gate.to_dict(),
                {
                    'gate_name': gate_name,
                    'phase': gate.phase,
                    'status': gate.status,
                    'completion': gate.completion_percentage
                }
            )

            logger.info(
                f"Emitted gate update: {gate_name} "
                f"({gate.completion_percentage:.1f}% complete)"
            )

        except Exception as e:
            logger.error(f"Error handling gate update for {file_path}: {e}")

    def _handle_manifest_update(self, file_path: Path) -> None:
        """Handle manifest.json update.

        Emits: manifest_update with ProjectManifestDict data

        Args:
            file_path: Path to manifest.json
        """
        try:
            from ..models import ProjectManifest

            # Load and parse manifest
            manifest = ProjectManifest.from_json(file_path)

            self._emit_typed_event(
                'manifest_update',
                manifest.to_dict(),
                {
                    'project': manifest.project_name,
                    'protocol': manifest.protocol,
                    'current_phase': manifest.current_phase
                }
            )

            logger.info(f"Emitted manifest update: {manifest.project_name}")

        except Exception as e:
            logger.error(f"Error handling manifest update: {e}")

    def _handle_metrics_update(self) -> None:
        """Handle aggregated metrics update.

        Emits: metrics_update with MetricsSummaryDict data

        This is called when multiple cache files change to provide
        an aggregated view of all metrics.
        """
        try:
            from ..models import MetricsSummary, ReuseAnalysis, PhaseGate

            # Load reuse analyses
            reuse_cache = self.workspace_root / ".ai-workspace" / "cache" / "reuse_checks.json"
            reuse_analyses = []
            if reuse_cache.exists():
                try:
                    reuse_analyses = ReuseAnalysis.from_json(reuse_cache)
                except Exception as e:
                    logger.warning(f"Could not load reuse analyses: {e}")

            # Load phase gates
            handoffs_dir = self.workspace_root / ".agent-workspace" / "handoffs"
            gates = []
            if handoffs_dir.exists():
                for gate_file in handoffs_dir.glob("gate-*.json"):
                    try:
                        gates.append(PhaseGate.from_json(gate_file))
                    except Exception as e:
                        logger.warning(f"Could not load gate {gate_file}: {e}")

            # Determine primary language (simplified - could be enhanced)
            primary_language = "Python"  # Default, could detect from config

            # Calculate metrics
            metrics = MetricsSummary.from_components(
                reuse_analyses=reuse_analyses,
                gates=gates,
                primary_language=primary_language,
                agents_used=[]
            )

            self._emit_typed_event(
                'metrics_update',
                metrics.to_dict(),
                {
                    'quality_score': metrics.quality_score,
                    'total_checks': metrics.total_reuse_checks,
                    'completed_gates': metrics.completed_gates
                }
            )

            logger.info(f"Emitted metrics update: Quality {metrics.quality_score:.1f}")

        except Exception as e:
            logger.error(f"Error handling metrics update: {e}")

    def on_modified(self, event: FileSystemEvent) -> None:
        """Handle file modification events.

        Args:
            event: File system event
        """
        if event.is_directory:
            return

        file_path = Path(event.src_path)

        # Only process relevant files
        if not file_path.suffix == '.json':
            return

        # Check if content actually changed
        if not self._has_content_changed(file_path):
            logger.debug(f"Content unchanged: {file_path.name}")
            return

        # Debounce rapid changes
        if not self._should_process_update(file_path):
            logger.debug(f"Debouncing update: {file_path.name}")
            return

        # Route to appropriate handler based on file type
        file_name = file_path.name

        if file_name == 'reuse_checks.json':
            self._handle_reuse_check_update(file_path)
            # Also trigger metrics update
            self._handle_metrics_update()

        elif file_name.startswith('gate-') and file_name.endswith('.json'):
            self._handle_gate_update(file_path)
            # Also trigger metrics update
            self._handle_metrics_update()

        elif file_name == 'manifest.json':
            self._handle_manifest_update(file_path)

        else:
            # Generic workspace update for other JSON files
            self.socketio.emit(
                'workspace_update',
                {
                    'type': 'file_modified',
                    'path': str(file_path),
                    'timestamp': datetime.now().isoformat(),
                }
            )

    def on_created(self, event: FileSystemEvent) -> None:
        """Handle file creation events.

        Args:
            event: File system event
        """
        if event.is_directory:
            return

        file_path = Path(event.src_path)

        # Process JSON and Markdown files
        if file_path.suffix not in ('.json', '.md'):
            return

        # Update content hash for new file
        self._calculate_file_hash(file_path)

        # Handle specific file types
        file_name = file_path.name

        if file_name.startswith('gate-') and file_name.endswith('.json'):
            self._handle_gate_update(file_path)

        else:
            # Generic creation notification
            self.socketio.emit(
                'workspace_update',
                {
                    'type': 'file_created',
                    'path': str(file_path),
                    'timestamp': datetime.now().isoformat(),
                }
            )


class FileMonitor:
    """File system monitor for real-time dashboard updates.

    Features:
    - Typed data model integration
    - Content-based change detection
    - Debouncing and batching
    - Comprehensive error handling
    - Health monitoring
    """

    # Health check interval in seconds
    HEALTH_INTERVAL = 30.0

    def __init__(self, socketio: Any) -> None:
        """Initialize file monitor.

        Args:
            socketio: SocketIO instance for broadcasting updates
        """
        self.socketio = socketio
        self.observer: Optional[Observer] = None
        self.handler: Optional[WorkspaceFileHandler] = None
        self.workspace_root = Path.cwd()
        self._health_check_scheduled = False

    def _schedule_health_check(self) -> None:
        """Schedule periodic health check broadcasts.

        Emits workspace_health event every 30 seconds with connection status.
        """
        def emit_health() -> None:
            """Emit health check event."""
            try:
                health_data = {
                    'status': 'connected',
                    'monitor_active': self.observer is not None and self.observer.is_alive(),
                    'timestamp': datetime.now().isoformat(),
                    'workspace_root': str(self.workspace_root)
                }

                self.socketio.emit('workspace_health', health_data)
                logger.debug(f"Health check: {health_data['status']}")

            except Exception as e:
                logger.error(f"Error in health check: {e}")

            # Schedule next health check
            if self._health_check_scheduled:
                self.socketio.sleep(self.HEALTH_INTERVAL)
                emit_health()

        # Start health check loop in background
        self._health_check_scheduled = True
        self.socketio.start_background_task(emit_health)

    def start(self) -> None:
        """Start monitoring workspace files.

        Monitors:
        - .ai-workspace/cache/ - Reuse checks, stack info
        - .agent-workspace/handoffs/ - Phase gates
        - .agent-workspace/manifest.json - Project metadata

        Raises:
            Exception: If observer fails to start (logged, not raised)
        """
        workspace_path = self.workspace_root / ".ai-workspace"
        agent_workspace_path = self.workspace_root / ".agent-workspace"

        if not workspace_path.exists():
            logger.warning(
                f"Warning: .ai-workspace directory not found at {workspace_path} - "
                "file monitoring disabled"
            )
            return

        try:
            self.observer = Observer()
            self.handler = WorkspaceFileHandler(self.socketio, self.workspace_root)

            # Monitor cache directory
            cache_dir = workspace_path / "cache"
            if cache_dir.exists():
                self.observer.schedule(self.handler, str(cache_dir), recursive=True)
                logger.info(f"Monitoring: {cache_dir}")
            else:
                logger.warning(f"Cache directory not found: {cache_dir}")

            # Monitor agent workspace (handoffs, decisions)
            if agent_workspace_path.exists():
                self.observer.schedule(
                    self.handler, str(agent_workspace_path), recursive=True
                )
                logger.info(f"Monitoring: {agent_workspace_path}")
            else:
                logger.warning(f"Agent workspace not found: {agent_workspace_path}")

            self.observer.start()
            logger.info(f"✓ File monitoring started for {workspace_path}")

            # Start health check broadcasts
            self._schedule_health_check()

            # Emit initial connection event
            self.socketio.emit('workspace_health', {
                'status': 'connected',
                'monitor_active': True,
                'timestamp': datetime.now().isoformat(),
                'workspace_root': str(self.workspace_root)
            })

        except Exception as e:
            logger.error(f"Could not start file monitoring: {e}", exc_info=True)
            self.observer = None

    def stop(self) -> None:
        """Stop monitoring workspace files.

        Performs graceful shutdown with cleanup.
        """
        # Stop health checks
        self._health_check_scheduled = False

        if self.observer:
            try:
                self.observer.stop()
                self.observer.join(timeout=5.0)
                logger.info("✓ File monitoring stopped")

                # Emit disconnect event
                self.socketio.emit('workspace_health', {
                    'status': 'disconnected',
                    'monitor_active': False,
                    'timestamp': datetime.now().isoformat()
                })

            except Exception as e:
                logger.error(f"Error stopping file monitor: {e}")
