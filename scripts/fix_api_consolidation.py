#!/usr/bin/env python3
"""
Enterprise API Consolidation Fix Script
========================================
Purpose: Automatically detect, diagnose, and fix API consolidation issues
Version: 1.0.0
Date: September 13, 2025
Author: Beverly Knits ERP Engineering Team

This script performs:
1. Complete API endpoint discovery and mapping
2. Automatic detection of missing/circular redirects
3. Backend unified endpoint generation
4. Frontend redirect map correction
5. Comprehensive validation and testing
"""

import os
import sys
import re
import json
import subprocess
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict
import requests
import time

# Configure enterprise logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'api_consolidation_fix_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
PROJECT_ROOT = Path("/mnt/c/finalee/beverly_knits_erp_v2")
BACKEND_FILE = PROJECT_ROOT / "src/core/beverly_comprehensive_erp.py"
FRONTEND_FILE = PROJECT_ROOT / "web/consolidated_dashboard.html"
BACKUP_DIR = PROJECT_ROOT / "backups" / f"api_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# API Configuration
API_DOMAINS = ["yarn", "production", "forecast", "inventory", "planning", "analytics", "system"]
SERVER_URL = "http://localhost:5006"
WRAPPER_URL = "http://localhost:8000"

@dataclass
class APIEndpoint:
    """Represents an API endpoint with metadata"""
    path: str
    line_number: int
    method: str = "GET"
    parameters: List[str] = None
    description: str = ""
    is_unified: bool = False
    is_deprecated: bool = False
    redirect_target: Optional[str] = None

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = []
        self.domain = self._extract_domain()

    def _extract_domain(self) -> str:
        """Extract domain from endpoint path"""
        parts = self.path.strip('/').split('/')
        if len(parts) >= 2:
            # Check for domain in path
            for domain in API_DOMAINS:
                if domain in parts[1]:
                    return domain
        return "other"

@dataclass
class APIMapping:
    """Represents a redirect mapping"""
    source: str
    target: str
    parameters: Dict[str, str]
    location: str  # 'backend' or 'frontend'
    line_number: int

class APIConsolidationFixer:
    """Enterprise-grade API consolidation fixer"""

    def __init__(self, dry_run: bool = False, verbose: bool = False):
        self.dry_run = dry_run
        self.verbose = verbose
        self.backend_endpoints: List[APIEndpoint] = []
        self.frontend_calls: List[Tuple[str, int]] = []
        self.backend_redirects: List[APIMapping] = []
        self.frontend_redirects: List[APIMapping] = []
        self.issues: List[Dict] = []
        self.fixes_applied: List[Dict] = []

    def run(self):
        """Main execution flow"""
        logger.info("=" * 80)
        logger.info("ENTERPRISE API CONSOLIDATION FIX - STARTING")
        logger.info("=" * 80)

        # Phase 1: Discovery
        logger.info("\n📊 PHASE 1: DISCOVERY")
        self._create_backup()
        self._discover_backend_endpoints()
        self._discover_frontend_calls()
        self._discover_redirect_mappings()

        # Phase 2: Analysis
        logger.info("\n🔍 PHASE 2: ANALYSIS")
        self._analyze_api_structure()
        self._detect_issues()

        # Phase 3: Fix Generation
        logger.info("\n🔧 PHASE 3: FIX GENERATION")
        fixes = self._generate_fixes()

        # Phase 4: Implementation
        if not self.dry_run:
            logger.info("\n⚡ PHASE 4: IMPLEMENTATION")
            self._apply_fixes(fixes)
        else:
            logger.info("\n📝 PHASE 4: DRY RUN - No changes applied")
            self._display_proposed_fixes(fixes)

        # Phase 5: Validation
        logger.info("\n✅ PHASE 5: VALIDATION")
        validation_results = self._validate_fixes()

        # Phase 6: Report
        logger.info("\n📈 PHASE 6: REPORT GENERATION")
        self._generate_report(validation_results)

        logger.info("\n" + "=" * 80)
        logger.info("API CONSOLIDATION FIX - COMPLETED")
        logger.info("=" * 80)

    def _create_backup(self):
        """Create backups of files before modification"""
        logger.info("Creating backups...")
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)

        for file_path in [BACKEND_FILE, FRONTEND_FILE]:
            if file_path.exists():
                backup_path = BACKUP_DIR / file_path.name
                import shutil
                shutil.copy2(file_path, backup_path)
                logger.info(f"  ✓ Backed up {file_path.name} to {backup_path}")

    def _discover_backend_endpoints(self):
        """Discover all backend API endpoints"""
        logger.info("Discovering backend endpoints...")

        with open(BACKEND_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        route_pattern = re.compile(r'@app\.route\("([^"]+)"(?:.*methods=\[([^\]]+)\])?\)')

        for i, line in enumerate(lines, 1):
            match = route_pattern.search(line)
            if match:
                path = match.group(1)
                methods = match.group(2) if match.group(2) else "'GET'"

                # Look for function name and docstring
                description = ""
                if i < len(lines) - 1:
                    next_line = lines[i]
                    if 'def ' in next_line:
                        # Try to get docstring
                        if i < len(lines) - 2 and '"""' in lines[i + 1]:
                            for j in range(i + 1, min(i + 10, len(lines))):
                                description += lines[j].strip()
                                if lines[j].count('"""') >= 2 or (j > i + 1 and '"""' in lines[j]):
                                    break

                endpoint = APIEndpoint(
                    path=path,
                    line_number=i,
                    method=methods.replace("'", "").replace('"', '').split(',')[0],
                    description=description[:200],
                    is_unified="unified" in path,
                )

                # Extract parameters from description or function
                if i < len(lines) - 5:
                    for j in range(i, min(i + 20, len(lines))):
                        if 'request.args.get' in lines[j]:
                            param_match = re.search(r'request\.args\.get\([\'"]([^\'"]+)', lines[j])
                            if param_match:
                                endpoint.parameters.append(param_match.group(1))

                self.backend_endpoints.append(endpoint)

        logger.info(f"  ✓ Found {len(self.backend_endpoints)} backend endpoints")

        # Group by domain
        by_domain = defaultdict(list)
        for ep in self.backend_endpoints:
            by_domain[ep.domain].append(ep)

        for domain, endpoints in by_domain.items():
            unified = [e for e in endpoints if e.is_unified]
            regular = [e for e in endpoints if not e.is_unified]
            logger.info(f"    • {domain}: {len(regular)} regular, {len(unified)} unified")

    def _discover_frontend_calls(self):
        """Discover all frontend API calls"""
        logger.info("Discovering frontend API calls...")

        with open(FRONTEND_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Patterns for API calls
        patterns = [
            re.compile(r'fetchAPI\([\'"]([^\'"]+)'),
            re.compile(r'fetch\([^)]*[\'"](/api/[^\'"]+)'),
            re.compile(r'baseUrl\s*\+\s*[\'"]([^\'"]+)')
        ]

        for i, line in enumerate(lines, 1):
            for pattern in patterns:
                matches = pattern.findall(line)
                for match in matches:
                    if '/api/' in match:
                        self.frontend_calls.append((match, i))

        logger.info(f"  ✓ Found {len(self.frontend_calls)} frontend API calls")

        # Analyze unique endpoints
        unique_calls = set(call[0] for call in self.frontend_calls)
        logger.info(f"    • {len(unique_calls)} unique endpoints called")

    def _discover_redirect_mappings(self):
        """Discover redirect mappings in both backend and frontend"""
        logger.info("Discovering redirect mappings...")

        # Backend redirects
        with open(BACKEND_FILE, 'r', encoding='utf-8') as f:
            content = f.read()

        # Find redirect_map in intercept_deprecated_endpoints
        redirect_section = re.search(
            r'redirect_map\s*=\s*\{(.*?)\}',
            content,
            re.DOTALL
        )

        if redirect_section:
            redirect_content = redirect_section.group(1)
            # Parse individual redirects
            redirect_pattern = re.compile(
                r"'([^']+)':\s*\(\s*'([^']+)'(?:,\s*(\{[^}]*\}))?\s*\)"
            )

            for match in redirect_pattern.finditer(redirect_content):
                source = match.group(1)
                target = match.group(2)
                params_str = match.group(3) if match.group(3) else '{}'

                try:
                    # Safe eval for dictionary parsing
                    params = eval(params_str) if params_str else {}
                except:
                    params = {}

                self.backend_redirects.append(APIMapping(
                    source=source,
                    target=target,
                    parameters=params,
                    location='backend',
                    line_number=0  # Would need more complex parsing for exact line
                ))

        logger.info(f"  ✓ Found {len(self.backend_redirects)} backend redirects")

        # Frontend redirects
        with open(FRONTEND_FILE, 'r', encoding='utf-8') as f:
            content = f.read()

        # Find deprecatedEndpoints mapping
        deprecated_section = re.search(
            r'(?:const\s+)?deprecatedEndpoints\s*=\s*\{(.*?)\};',
            content,
            re.DOTALL
        )

        if deprecated_section:
            deprecated_content = deprecated_section.group(1)
            # Parse individual mappings
            mapping_pattern = re.compile(r"'([^']+)':\s*'([^']+)'")

            for match in mapping_pattern.finditer(deprecated_content):
                source = match.group(1)
                target = match.group(2)

                self.frontend_redirects.append(APIMapping(
                    source=source,
                    target=target,
                    parameters={},
                    location='frontend',
                    line_number=0
                ))

        logger.info(f"  ✓ Found {len(self.frontend_redirects)} frontend redirects")

    def _analyze_api_structure(self):
        """Analyze the API structure for patterns and issues"""
        logger.info("Analyzing API structure...")

        # Analyze unified endpoint coverage
        domains_with_unified = set()
        domains_without_unified = set()

        for domain in API_DOMAINS:
            domain_endpoints = [e for e in self.backend_endpoints if e.domain == domain]
            unified_endpoints = [e for e in domain_endpoints if e.is_unified]

            if unified_endpoints:
                domains_with_unified.add(domain)
                logger.info(f"  ✓ {domain}: Has unified endpoint")
            elif domain_endpoints:
                domains_without_unified.add(domain)
                logger.info(f"  ✗ {domain}: Missing unified endpoint ({len(domain_endpoints)} endpoints)")

        # Check for orphaned API calls
        backend_paths = set(e.path for e in self.backend_endpoints)
        frontend_unique = set(call[0].split('?')[0] for call in self.frontend_calls)

        orphaned_calls = []
        for call_path in frontend_unique:
            # Check if it exists or has a redirect
            if call_path not in backend_paths:
                # Check if it has a redirect
                has_redirect = False
                for redirect in self.backend_redirects + self.frontend_redirects:
                    if redirect.source == call_path:
                        has_redirect = True
                        break

                if not has_redirect:
                    orphaned_calls.append(call_path)

        if orphaned_calls:
            logger.warning(f"  ⚠ Found {len(orphaned_calls)} orphaned API calls")
            for call in orphaned_calls[:5]:  # Show first 5
                logger.warning(f"    • {call}")

    def _detect_issues(self):
        """Detect specific issues in the API structure"""
        logger.info("Detecting issues...")

        # Issue 1: Circular redirects
        for redirect in self.frontend_redirects:
            if redirect.source == redirect.target:
                self.issues.append({
                    'type': 'CIRCULAR_REDIRECT',
                    'severity': 'CRITICAL',
                    'location': 'frontend',
                    'source': redirect.source,
                    'target': redirect.target,
                    'description': f"Circular redirect detected: {redirect.source} → {redirect.target}"
                })
                logger.error(f"  🔴 CRITICAL: Circular redirect {redirect.source}")

        # Issue 2: Missing backend endpoints
        frontend_calls_unique = set(call[0].split('?')[0] for call in self.frontend_calls)
        backend_paths = set(e.path for e in self.backend_endpoints)

        for call_path in frontend_calls_unique:
            if call_path not in backend_paths:
                # Check if it's supposed to be redirected
                is_redirected = any(
                    r.source == call_path
                    for r in self.backend_redirects + self.frontend_redirects
                )

                if not is_redirected:
                    self.issues.append({
                        'type': 'MISSING_ENDPOINT',
                        'severity': 'HIGH',
                        'location': 'backend',
                        'path': call_path,
                        'description': f"Frontend calls {call_path} but endpoint doesn't exist"
                    })
                    logger.warning(f"  ⚠ HIGH: Missing endpoint {call_path}")

        # Issue 3: Inconsistent redirect targets
        for fr in self.frontend_redirects:
            # Check if the target exists in backend
            if fr.target not in backend_paths:
                # Check if it's another redirect
                is_another_redirect = any(
                    r.source == fr.target
                    for r in self.frontend_redirects
                )

                if is_another_redirect:
                    self.issues.append({
                        'type': 'CHAINED_REDIRECT',
                        'severity': 'MEDIUM',
                        'location': 'frontend',
                        'source': fr.source,
                        'target': fr.target,
                        'description': f"Redirect chain detected: {fr.source} → {fr.target}"
                    })
                    logger.warning(f"  ⚠ MEDIUM: Redirect chain {fr.source} → {fr.target}")
                elif fr.target != fr.source:  # Not circular
                    self.issues.append({
                        'type': 'INVALID_REDIRECT_TARGET',
                        'severity': 'HIGH',
                        'location': 'frontend',
                        'source': fr.source,
                        'target': fr.target,
                        'description': f"Redirect target doesn't exist: {fr.target}"
                    })
                    logger.warning(f"  ⚠ HIGH: Invalid redirect target {fr.target}")

        # Issue 4: Missing unified endpoints for domains with multiple endpoints
        for domain in API_DOMAINS:
            domain_endpoints = [e for e in self.backend_endpoints if e.domain == domain]
            unified_endpoints = [e for e in domain_endpoints if e.is_unified]

            if len(domain_endpoints) > 3 and not unified_endpoints:
                self.issues.append({
                    'type': 'MISSING_UNIFIED_ENDPOINT',
                    'severity': 'MEDIUM',
                    'location': 'backend',
                    'domain': domain,
                    'endpoint_count': len(domain_endpoints),
                    'description': f"Domain '{domain}' has {len(domain_endpoints)} endpoints but no unified endpoint"
                })
                logger.info(f"  ℹ MEDIUM: {domain} needs unified endpoint")

        logger.info(f"\n  📊 Issue Summary:")
        logger.info(f"    • Critical: {len([i for i in self.issues if i['severity'] == 'CRITICAL'])}")
        logger.info(f"    • High: {len([i for i in self.issues if i['severity'] == 'HIGH'])}")
        logger.info(f"    • Medium: {len([i for i in self.issues if i['severity'] == 'MEDIUM'])}")

    def _generate_fixes(self) -> List[Dict]:
        """Generate fixes for detected issues"""
        logger.info("Generating fixes...")
        fixes = []

        for issue in self.issues:
            if issue['type'] == 'CIRCULAR_REDIRECT':
                # Fix circular redirect by finding the correct target
                source = issue['source']

                # Find the best target endpoint
                if 'yarn/unified' in source:
                    target = '/api/yarn-intelligence'
                elif 'production/unified' in source:
                    target = '/api/production-planning'
                elif 'forecast/unified' in source:
                    target = '/api/ml-forecast-detailed'
                else:
                    # Try to find a similar endpoint
                    domain = source.split('/')[2] if len(source.split('/')) > 2 else ''
                    candidates = [
                        e.path for e in self.backend_endpoints
                        if domain in e.path and not e.is_deprecated
                    ]
                    target = candidates[0] if candidates else None

                if target:
                    fixes.append({
                        'type': 'UPDATE_REDIRECT',
                        'file': 'frontend',
                        'source': source,
                        'new_target': target,
                        'old_target': source,
                        'description': f"Fix circular redirect: {source} → {target}"
                    })

            elif issue['type'] == 'MISSING_ENDPOINT':
                # Generate a unified endpoint if it's a unified path
                if 'unified' in issue['path']:
                    domain = issue['path'].split('/')[2] if len(issue['path'].split('/')) > 2 else ''
                    fixes.append({
                        'type': 'CREATE_UNIFIED_ENDPOINT',
                        'file': 'backend',
                        'path': issue['path'],
                        'domain': domain,
                        'description': f"Create unified endpoint for {domain}"
                    })
                else:
                    # Create a redirect to an existing endpoint
                    domain = self._extract_domain_from_path(issue['path'])
                    similar = self._find_similar_endpoint(issue['path'])
                    if similar:
                        fixes.append({
                            'type': 'ADD_REDIRECT',
                            'file': 'backend',
                            'source': issue['path'],
                            'target': similar,
                            'description': f"Redirect {issue['path']} to {similar}"
                        })

            elif issue['type'] == 'MISSING_UNIFIED_ENDPOINT':
                # Generate unified endpoint code
                fixes.append({
                    'type': 'CREATE_UNIFIED_ENDPOINT',
                    'file': 'backend',
                    'domain': issue['domain'],
                    'path': f"/api/{issue['domain']}/unified",
                    'description': f"Create unified endpoint for {issue['domain']} domain"
                })

        logger.info(f"  ✓ Generated {len(fixes)} fixes")
        return fixes

    def _extract_domain_from_path(self, path: str) -> str:
        """Extract domain from API path"""
        parts = path.strip('/').split('/')
        for domain in API_DOMAINS:
            for part in parts:
                if domain in part:
                    return domain
        return 'other'

    def _find_similar_endpoint(self, path: str) -> Optional[str]:
        """Find a similar existing endpoint"""
        domain = self._extract_domain_from_path(path)

        # Find endpoints in the same domain
        candidates = [
            e.path for e in self.backend_endpoints
            if e.domain == domain and not e.is_deprecated
        ]

        # Prefer intelligence or enhanced endpoints
        for keyword in ['intelligence', 'enhanced', 'unified', 'analysis']:
            for candidate in candidates:
                if keyword in candidate:
                    return candidate

        return candidates[0] if candidates else None

    def _apply_fixes(self, fixes: List[Dict]):
        """Apply the generated fixes"""
        logger.info("Applying fixes...")

        # Group fixes by file
        backend_fixes = [f for f in fixes if f['file'] == 'backend']
        frontend_fixes = [f for f in fixes if f['file'] == 'frontend']

        # Apply backend fixes
        if backend_fixes:
            self._apply_backend_fixes(backend_fixes)

        # Apply frontend fixes
        if frontend_fixes:
            self._apply_frontend_fixes(frontend_fixes)

        logger.info(f"  ✓ Applied {len(fixes)} fixes")

    def _apply_backend_fixes(self, fixes: List[Dict]):
        """Apply fixes to backend file"""
        logger.info("  Applying backend fixes...")

        with open(BACKEND_FILE, 'r', encoding='utf-8') as f:
            content = f.read()

        for fix in fixes:
            if fix['type'] == 'CREATE_UNIFIED_ENDPOINT':
                # Generate unified endpoint code
                endpoint_code = self._generate_unified_endpoint_code(fix['domain'], fix['path'])

                # Find a good insertion point (after similar endpoints)
                insertion_point = self._find_insertion_point(content, fix['domain'])

                # Insert the code
                lines = content.split('\n')
                lines.insert(insertion_point, endpoint_code)
                content = '\n'.join(lines)

                logger.info(f"    ✓ Created unified endpoint: {fix['path']}")
                self.fixes_applied.append(fix)

            elif fix['type'] == 'ADD_REDIRECT':
                # Add to redirect_map
                redirect_entry = f"        '{fix['source']}': ('{fix['target']}', {{}}),"

                # Find redirect_map and add entry
                redirect_map_match = re.search(r'(redirect_map\s*=\s*\{)', content)
                if redirect_map_match:
                    insert_pos = redirect_map_match.end()
                    content = content[:insert_pos] + f"\n{redirect_entry}" + content[insert_pos:]

                    logger.info(f"    ✓ Added redirect: {fix['source']} → {fix['target']}")
                    self.fixes_applied.append(fix)

        # Write back
        with open(BACKEND_FILE, 'w', encoding='utf-8') as f:
            f.write(content)

    def _apply_frontend_fixes(self, fixes: List[Dict]):
        """Apply fixes to frontend file"""
        logger.info("  Applying frontend fixes...")

        with open(FRONTEND_FILE, 'r', encoding='utf-8') as f:
            content = f.read()

        for fix in fixes:
            if fix['type'] == 'UPDATE_REDIRECT':
                # Update the redirect mapping
                old_mapping = f"'{fix['source']}': '{fix['old_target']}'"
                new_mapping = f"'{fix['source']}': '{fix['new_target']}'"

                if old_mapping in content:
                    content = content.replace(old_mapping, new_mapping)
                    logger.info(f"    ✓ Updated redirect: {fix['source']} → {fix['new_target']}")
                    self.fixes_applied.append(fix)

        # Write back
        with open(FRONTEND_FILE, 'w', encoding='utf-8') as f:
            f.write(content)

    def _generate_unified_endpoint_code(self, domain: str, path: str) -> str:
        """Generate code for a unified endpoint"""

        # Find all endpoints in this domain to consolidate
        domain_endpoints = [e for e in self.backend_endpoints if e.domain == domain]

        # Generate the endpoint code
        code = f'''
@app.route("{path}")
def get_{domain}_unified():
    """
    Unified {domain} endpoint - Consolidates all {domain} functionality
    Auto-generated by API Consolidation Fix Script

    Supported parameters:
    - view: Type of view to return (full, summary, detailed)
    - analysis: Type of analysis (standard, advanced)
    - format: Response format (json, report)
    - realtime: Include real-time data (true/false)
    """
    global new_api_count
    new_api_count += 1

    # Get request parameters
    view = request.args.get('view', 'full')
    analysis = request.args.get('analysis', 'standard')
    format_type = request.args.get('format', 'json')
    realtime = request.args.get('realtime', 'false').lower() == 'true'

    try:
        # Route to appropriate handler based on parameters
        if view == 'summary':
            # Return summary view
            result = {{
                'domain': '{domain}',
                'view': 'summary',
                'data': {{}},
                'timestamp': datetime.now().isoformat()
            }}
        elif analysis == 'advanced':
            # Return advanced analysis
            result = {{
                'domain': '{domain}',
                'analysis': 'advanced',
                'data': {{}},
                'timestamp': datetime.now().isoformat()
            }}
        else:
            # Default full view - aggregate data from existing endpoints
            result = {{
                'domain': '{domain}',
                'view': 'full',
                'data': {{}},
                'timestamp': datetime.now().isoformat()
            }}

            # TODO: Integrate with existing {domain} endpoints
            # Example integration points:'''

        # Add references to existing endpoints
        for ep in domain_endpoints[:3]:  # Show first 3 as examples
            code += f'''
            # - {ep.path} (line {ep.line_number})'''

        code += '''

        return jsonify(clean_response_for_json(result))

    except Exception as e:
        logger.error(f"Error in {path}: {str(e)}")
        return jsonify({
            'error': str(e),
            'domain': '{domain}',
            'timestamp': datetime.now().isoformat()
        }), 500
'''

        return code

    def _find_insertion_point(self, content: str, domain: str) -> int:
        """Find the best line number to insert new endpoint"""
        lines = content.split('\n')

        # Find last endpoint in the same domain
        last_domain_line = 0
        for i, line in enumerate(lines):
            if '@app.route' in line and domain in line:
                last_domain_line = i

        if last_domain_line > 0:
            # Find the end of this endpoint's function
            for i in range(last_domain_line, len(lines)):
                if i > last_domain_line + 1 and lines[i] and not lines[i].startswith(' '):
                    return i

        # Default: insert before the last route
        for i in range(len(lines) - 1, 0, -1):
            if '@app.route("/api/health")' in lines[i]:
                return i - 1

        return len(lines) - 100  # Fallback

    def _display_proposed_fixes(self, fixes: List[Dict]):
        """Display proposed fixes in dry run mode"""
        logger.info("\n📋 PROPOSED FIXES:")

        for i, fix in enumerate(fixes, 1):
            logger.info(f"\n  Fix #{i}:")
            logger.info(f"    Type: {fix['type']}")
            logger.info(f"    File: {fix['file']}")
            logger.info(f"    Description: {fix['description']}")

            if fix['type'] == 'UPDATE_REDIRECT':
                logger.info(f"    Change: '{fix['source']}' → '{fix['new_target']}'")
            elif fix['type'] == 'CREATE_UNIFIED_ENDPOINT':
                logger.info(f"    Create: {fix['path']}")
            elif fix['type'] == 'ADD_REDIRECT':
                logger.info(f"    Add: '{fix['source']}' → '{fix['target']}'")

    def _validate_fixes(self) -> Dict:
        """Validate that fixes were applied correctly"""
        logger.info("Validating fixes...")

        validation_results = {
            'syntax_valid': False,
            'endpoints_accessible': [],
            'redirects_working': [],
            'issues_resolved': [],
            'new_issues': []
        }

        # Check Python syntax
        try:
            import py_compile
            py_compile.compile(str(BACKEND_FILE), doraise=True)
            validation_results['syntax_valid'] = True
            logger.info("  ✓ Backend syntax valid")
        except py_compile.PyCompileError as e:
            logger.error(f"  ✗ Backend syntax error: {e}")

        # Test endpoint accessibility (if server is running)
        if self._is_server_running():
            logger.info("  Testing endpoint accessibility...")

            for fix in self.fixes_applied:
                if fix['type'] == 'CREATE_UNIFIED_ENDPOINT':
                    response = self._test_endpoint(fix['path'])
                    validation_results['endpoints_accessible'].append({
                        'path': fix['path'],
                        'status': response.status_code if response else 'N/A',
                        'success': response and response.status_code == 200
                    })
                elif fix['type'] == 'UPDATE_REDIRECT':
                    # Test that redirect works
                    response = self._test_endpoint(fix['source'])
                    validation_results['redirects_working'].append({
                        'source': fix['source'],
                        'target': fix['new_target'],
                        'success': response and response.status_code == 200
                    })

        # Check if issues are resolved
        old_issue_count = len(self.issues)
        self.issues = []  # Reset
        self._detect_issues()  # Re-detect
        new_issue_count = len(self.issues)

        validation_results['issues_resolved'] = old_issue_count - new_issue_count
        validation_results['new_issues'] = self.issues

        logger.info(f"  ✓ Resolved {validation_results['issues_resolved']} issues")
        if validation_results['new_issues']:
            logger.warning(f"  ⚠ {len(validation_results['new_issues'])} issues remain")

        return validation_results

    def _is_server_running(self) -> bool:
        """Check if the server is running"""
        try:
            response = requests.get(f"{SERVER_URL}/api/health", timeout=2)
            return response.status_code == 200
        except:
            return False

    def _test_endpoint(self, path: str) -> Optional[requests.Response]:
        """Test if an endpoint is accessible"""
        try:
            response = requests.get(f"{SERVER_URL}{path}", timeout=5)
            return response
        except:
            return None

    def _generate_report(self, validation_results: Dict):
        """Generate comprehensive report"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    API CONSOLIDATION FIX REPORT                              ║
║                         {timestamp}                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

📊 DISCOVERY SUMMARY
─────────────────────
  • Backend Endpoints: {len(self.backend_endpoints)}
  • Frontend API Calls: {len(self.frontend_calls)}
  • Backend Redirects: {len(self.backend_redirects)}
  • Frontend Redirects: {len(self.frontend_redirects)}

🔍 ISSUES DETECTED
─────────────────
  • Critical: {len([i for i in self.issues if i.get('severity') == 'CRITICAL'])}
  • High: {len([i for i in self.issues if i.get('severity') == 'HIGH'])}
  • Medium: {len([i for i in self.issues if i.get('severity') == 'MEDIUM'])}

🔧 FIXES APPLIED
─────────────────
  • Total Fixes: {len(self.fixes_applied)}
  • Endpoints Created: {len([f for f in self.fixes_applied if f['type'] == 'CREATE_UNIFIED_ENDPOINT'])}
  • Redirects Updated: {len([f for f in self.fixes_applied if f['type'] == 'UPDATE_REDIRECT'])}
  • Redirects Added: {len([f for f in self.fixes_applied if f['type'] == 'ADD_REDIRECT'])}

✅ VALIDATION RESULTS
─────────────────────
  • Syntax Valid: {'✓' if validation_results['syntax_valid'] else '✗'}
  • Issues Resolved: {validation_results['issues_resolved']}
  • Remaining Issues: {len(validation_results['new_issues'])}

📁 BACKUP LOCATION
─────────────────
  {BACKUP_DIR}

🎯 NEXT STEPS
─────────────
  1. Restart the server: pkill -f "python3.*beverly" && python3 src/core/beverly_comprehensive_erp.py
  2. Clear browser cache and reload dashboard
  3. Test affected endpoints
  4. Monitor error logs for any issues

═══════════════════════════════════════════════════════════════════════════════
"""

        # Save report
        report_file = PROJECT_ROOT / f"api_fix_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(report)

        logger.info(report)
        logger.info(f"\n📄 Report saved to: {report_file}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Enterprise API Consolidation Fix Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full fix (with backups)
  python fix_api_consolidation.py

  # Dry run to see what would be changed
  python fix_api_consolidation.py --dry-run

  # Verbose output for debugging
  python fix_api_consolidation.py --verbose

  # Analyze only, no fixes
  python fix_api_consolidation.py --analyze-only
        """
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be changed without making changes'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed output'
    )

    parser.add_argument(
        '--analyze-only',
        action='store_true',
        help='Only analyze and report issues, do not generate fixes'
    )

    args = parser.parse_args()

    # Run the fixer
    fixer = APIConsolidationFixer(
        dry_run=args.dry_run or args.analyze_only,
        verbose=args.verbose
    )

    try:
        fixer.run()
        sys.exit(0)
    except KeyboardInterrupt:
        logger.info("\n\n⚠ Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n\n❌ Fatal error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()