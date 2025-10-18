"""
Style Mapping Module for Beverly Knits ERP
Maps between fStyle# (sales) and Style# (BOM) using Turso database
"""

from typing import Dict, List, Optional, Set
import logging

logger = logging.getLogger(__name__)


class StyleMapper:
    """Maps between different style naming conventions across the ERP system"""

    def __init__(self, turso_client=None):
        """
        Initialize the style mapper with Turso database client

        Args:
            turso_client: TursoClient instance (will be auto-loaded if None)
        """
        self.fstyle_to_gbase = {}
        self.gbase_to_bom_styles = {}
        self.direct_mappings = {}
        self.turso_client = turso_client

        # Load mappings from Turso database
        self._load_from_turso()

    def _load_from_turso(self) -> bool:
        """Load style mappings from Turso database"""
        try:
            # Lazy load turso_client if not provided
            if self.turso_client is None:
                from src.database.turso_client import get_turso_client
                self.turso_client = get_turso_client()

            # Query all style mappings
            sql = "SELECT fstyle, gbase, style FROM style_mappings"
            rows = self.turso_client.execute(sql)

            if not rows:
                logger.warning("No style mappings found in Turso database")
                logger.warning("Run: python scripts/import_style_mappings_to_turso.py")
                return False

            # Build internal mappings
            for row in rows:
                fstyle = row.get('fstyle', '')
                gbase = row.get('gbase', '')
                style = row.get('style', '')

                if fstyle and gbase:
                    self.fstyle_to_gbase[fstyle] = gbase

                    if style:
                        self.direct_mappings[fstyle] = style

            logger.info(f"✓ Loaded {len(self.fstyle_to_gbase)} style mappings from Turso")
            return True

        except Exception as e:
            logger.error(f"Error loading style mappings from Turso: {e}")
            logger.error("Make sure to run: python scripts/import_style_mappings_to_turso.py")
            return False

    def reload_mappings(self) -> bool:
        """Reload mappings from Turso database"""
        self.fstyle_to_gbase.clear()
        self.gbase_to_bom_styles.clear()
        self.direct_mappings.clear()
        return self._load_from_turso()

    def _build_mappings(self):
        """Build internal mapping dictionaries (deprecated - now loads from Turso)"""
        # This method is kept for backward compatibility but does nothing
        # Mappings are now loaded directly from Turso in __init__
        pass

    def set_bom_styles(self, bom_styles: Set[str]):
        """Set the available BOM styles for matching"""
        self.gbase_to_bom_styles.clear()

        for bom_style in bom_styles:
            if isinstance(bom_style, str):
                # Extract base from BOM style (e.g., "C1B4014/1A" -> "C1B4014")
                base = bom_style.split("/")[0] if "/" in bom_style else bom_style
                base = base.split("-")[0] if "-" in base and "/" not in base else base

                if base not in self.gbase_to_bom_styles:
                    self.gbase_to_bom_styles[base] = []
                self.gbase_to_bom_styles[base].append(bom_style)

    def map_sales_to_bom(self, sales_style: str) -> List[str]:
        """
        Map a sales style (fStyle#) to potential BOM styles
        Returns a list of matching BOM styles
        """
        if not sales_style:
            return []

        sales_style = str(sales_style).strip()

        # First, get the gBase from the mapping
        gbase = self.fstyle_to_gbase.get(sales_style)

        if not gbase:
            # Try case-insensitive match
            for fstyle, base in self.fstyle_to_gbase.items():
                if fstyle.lower() == sales_style.lower():
                    gbase = base
                    break

        if not gbase:
            return []

        # Now find all BOM styles that match this gBase
        matching_bom_styles = []

        # Direct match
        if gbase in self.gbase_to_bom_styles:
            matching_bom_styles.extend(self.gbase_to_bom_styles[gbase])

        # Case-insensitive match
        gbase_lower = gbase.lower()
        for base, styles in self.gbase_to_bom_styles.items():
            if base.lower() == gbase_lower and base != gbase:
                matching_bom_styles.extend(styles)

        return list(set(matching_bom_styles))  # Remove duplicates

    def map_bom_to_sales(self, bom_style: str) -> List[str]:
        """
        Map a BOM style to potential sales styles (fStyle#)
        Returns a list of matching sales styles
        """
        if not bom_style:
            return []

        bom_style = str(bom_style).strip()

        # Extract base from BOM style
        base = bom_style.split("/")[0] if "/" in bom_style else bom_style

        # Find all fStyles that map to this base
        matching_sales_styles = []
        for fstyle, gbase in self.fstyle_to_gbase.items():
            if gbase == base or gbase.lower() == base.lower():
                matching_sales_styles.append(fstyle)

        return matching_sales_styles

    def get_all_mappings(self) -> Dict[str, List[str]]:
        """Get all sales to BOM mappings"""
        mappings = {}
        for fstyle in self.fstyle_to_gbase.keys():
            bom_styles = self.map_sales_to_bom(fstyle)
            if bom_styles:
                mappings[fstyle] = bom_styles
        return mappings

    def get_mapping_stats(self) -> Dict:
        """Get statistics about the loaded mappings"""
        return {
            "total_fstyles": len(self.fstyle_to_gbase),
            "unique_gbases": len(set(self.fstyle_to_gbase.values())),
            "bom_bases_mapped": len(self.gbase_to_bom_styles),
            "mappings_loaded": len(self.fstyle_to_gbase) > 0,
            "source": "turso_database"
        }


# Global instance
_style_mapper = None


def get_style_mapper(turso_client=None) -> StyleMapper:
    """
    Get or create the global style mapper instance

    Args:
        turso_client: Optional TursoClient instance (auto-loaded if None)

    Returns:
        StyleMapper singleton instance
    """
    global _style_mapper

    if _style_mapper is None:
        _style_mapper = StyleMapper(turso_client)

    return _style_mapper
