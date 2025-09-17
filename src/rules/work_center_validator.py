"""Work Center Pattern Validator.

Validates work center codes following the pattern:
TYPE.DIAMETER.MANUFACTURER.CUT

Where:
- TYPE: Machine type (1-9)
- DIAMETER: Machine diameter (e.g., 30, 38)
- MANUFACTURER: F=Monarch, M=Mayer, etc.
- CUT: Needles per inch on the diameter
"""

from __future__ import annotations

from typing import Optional, Dict, List, Tuple
import re
import logging

import pandas as pd

logger = logging.getLogger(__name__)


class WorkCenterValidator:
    """Validator for work center pattern compliance.

    Ensures all work center codes follow the standard format
    and validates machine assignments against this pattern.
    """

    # Master pattern: TYPE.DIAMETER.MANUFACTURER.CUT
    PATTERN = re.compile(r'^(\d)\.(\d{2})\.(\d{2})\.([A-Z]+)$')

    # Manufacturer codes
    MANUFACTURER_CODES = {
        'F': 'Monarch',
        'M': 'Mayer',
        'C': 'Circular',
        'V': 'Vanguard',
        'J': 'Jacquard',
        'R': 'Rib',
        'T': 'Terry',
        'S': 'Special',
        'FOW': 'Monarch Open Width',  # Special case from data
    }

    # Machine type descriptions
    MACHINE_TYPES = {
        '1': 'Single Jersey',
        '2': 'Double Jersey',
        '3': 'Rib',
        '4': 'Interlock',
        '5': 'Fleece',
        '6': 'Terry',
        '7': 'Jacquard',
        '8': 'Special Knit',
        '9': 'Warp Knit'
    }

    def __init__(self) -> None:
        """Initialize validator."""
        self.validation_cache: Dict[str, Dict] = {}
        logger.info("WorkCenterValidator initialized")

    def validate_work_center(self, work_center: str) -> Tuple[bool, Optional[Dict]]:
        """Validate work center pattern and extract components.

        Args:
            work_center: Work center code to validate

        Returns:
            Tuple of (is_valid, parsed_components)
            Components dict contains: type, diameter, cut, manufacturer

        Examples:
            >>> validator = WorkCenterValidator()
            >>> valid, parts = validator.validate_work_center("1.30.20.F")
            >>> assert valid is True
            >>> assert parts['manufacturer'] == 'Monarch'
        """
        # Check cache first
        if work_center in self.validation_cache:
            cached = self.validation_cache[work_center]
            return cached['valid'], cached['components']

        # Clean input
        work_center = work_center.strip().upper()

        # Match pattern
        match = self.PATTERN.match(work_center)

        if not match:
            logger.warning(f"Invalid work center format: {work_center}")
            self.validation_cache[work_center] = {'valid': False, 'components': None}
            return False, None

        # Extract components
        machine_type, diameter, cut, manufacturer_code = match.groups()

        # Validate manufacturer code
        if manufacturer_code not in self.MANUFACTURER_CODES:
            # Check for special cases like FOW
            if manufacturer_code not in ['FOW']:
                logger.warning(f"Unknown manufacturer code: {manufacturer_code}")

        # Build components dictionary
        components = {
            'type': machine_type,
            'type_desc': self.MACHINE_TYPES.get(machine_type, 'Unknown'),
            'diameter': int(diameter),
            'cut': int(cut),
            'manufacturer_code': manufacturer_code,
            'manufacturer': self.MANUFACTURER_CODES.get(
                manufacturer_code, manufacturer_code
            ),
            'full_code': work_center
        }

        # Validate ranges
        if not self._validate_ranges(components):
            self.validation_cache[work_center] = {'valid': False, 'components': None}
            return False, None

        # Cache result
        self.validation_cache[work_center] = {
            'valid': True,
            'components': components
        }

        return True, components

    def _validate_ranges(self, components: Dict) -> bool:
        """Validate component ranges.

        Args:
            components: Parsed work center components

        Returns:
            True if all components are within valid ranges
        """
        # Type must be 1-9
        if not 1 <= int(components['type']) <= 9:
            logger.warning(f"Invalid machine type: {components['type']}")
            return False

        # Diameter typically 10-99
        if not 10 <= components['diameter'] <= 99:
            logger.warning(f"Unusual diameter: {components['diameter']}")
            # Don't fail, just warn

        # Cut (needles per inch) typically 10-40
        if not 5 <= components['cut'] <= 50:
            logger.warning(f"Unusual cut: {components['cut']}")
            # Don't fail, just warn

        return True

    def validate_machine_mapping(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate all work centers in a machine mapping DataFrame.

        Args:
            df: DataFrame with WC (work center) and MACH (machine) columns

        Returns:
            DataFrame with validation results added
        """
        if 'WC' not in df.columns:
            raise ValueError("DataFrame must have 'WC' column")

        # Validate each work center
        results = []
        for wc in df['WC'].unique():
            valid, components = self.validate_work_center(wc)
            results.append({
                'WC': wc,
                'Valid': valid,
                'Type': components['type'] if components else None,
                'Diameter': components['diameter'] if components else None,
                'Cut': components['cut'] if components else None,
                'Manufacturer': components['manufacturer'] if components else None
            })

        # Create results DataFrame
        result_df = pd.DataFrame(results)

        # Merge with original
        return df.merge(result_df, on='WC', how='left')

    def get_machines_for_pattern(
        self,
        df: pd.DataFrame,
        machine_type: Optional[str] = None,
        diameter: Optional[int] = None,
        manufacturer: Optional[str] = None
    ) -> List[str]:
        """Get machines matching specified pattern criteria.

        Args:
            df: Machine mapping DataFrame
            machine_type: Filter by machine type (1-9)
            diameter: Filter by diameter
            manufacturer: Filter by manufacturer code

        Returns:
            List of machine IDs matching criteria
        """
        machines = []

        for _, row in df.iterrows():
            wc = row['WC']
            valid, components = self.validate_work_center(wc)

            if not valid:
                continue

            # Apply filters
            if machine_type and components['type'] != machine_type:
                continue
            if diameter and components['diameter'] != diameter:
                continue
            if manufacturer and components['manufacturer_code'] != manufacturer:
                continue

            machines.append(str(row['MACH']))

        return machines

    def format_work_center(
        self,
        machine_type: str,
        diameter: int,
        cut: int,
        manufacturer: str
    ) -> str:
        """Format components into standard work center code.

        Args:
            machine_type: Type code (1-9)
            diameter: Machine diameter
            cut: Needles per inch
            manufacturer: Manufacturer code (F, M, etc.)

        Returns:
            Formatted work center code

        Examples:
            >>> validator = WorkCenterValidator()
            >>> wc = validator.format_work_center("1", 30, 20, "F")
            >>> assert wc == "1.30.20.F"
        """
        return f"{machine_type}.{diameter:02d}.{cut:02d}.{manufacturer}"

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validated work centers.

        Returns:
            Summary statistics of validations
        """
        total = len(self.validation_cache)
        valid = sum(1 for v in self.validation_cache.values() if v['valid'])
        invalid = total - valid

        # Count by manufacturer
        manufacturer_counts = {}
        for cache_entry in self.validation_cache.values():
            if cache_entry['valid'] and cache_entry['components']:
                mfr = cache_entry['components']['manufacturer']
                manufacturer_counts[mfr] = manufacturer_counts.get(mfr, 0) + 1

        return {
            'total_validated': total,
            'valid': valid,
            'invalid': invalid,
            'validity_rate': (valid / total * 100) if total > 0 else 0,
            'manufacturers': manufacturer_counts
        }


if __name__ == "__main__":
    """Validation with real work center patterns."""

    validator = WorkCenterValidator()

    # Test Case 1: Standard Monarch machine
    valid1, comp1 = validator.validate_work_center("1.30.20.F")
    assert valid1 is True
    assert comp1['manufacturer'] == 'Monarch'
    assert comp1['diameter'] == 30
    logger.info(f"Test 1 passed: {comp1}")

    # Test Case 2: Mayer machine
    valid2, comp2 = validator.validate_work_center("1.30.20.M")
    assert valid2 is True
    assert comp2['manufacturer'] == 'Mayer'
    logger.info(f"Test 2 passed: {comp2}")

    # Test Case 3: Special FOW case
    valid3, comp3 = validator.validate_work_center("1.30.20.FOW")
    assert valid3 is True
    assert comp3['manufacturer_code'] == 'FOW'
    logger.info(f"Test 3 passed: FOW handling")

    # Test Case 4: Invalid format
    valid4, comp4 = validator.validate_work_center("INVALID")
    assert valid4 is False
    assert comp4 is None
    logger.info("Test 4 passed: Invalid format rejected")

    # Test Case 5: Format work center
    formatted = validator.format_work_center("9", 38, 20, "F")
    assert formatted == "9.38.20.F"
    logger.info(f"Test 5 passed: Formatted as {formatted}")

    print("All validations passed!")