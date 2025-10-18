"""
Fabric Inquiry Blueprint - Search and view fabric inventory across production stages

This blueprint provides endpoints for:
- Searching fabric inventory by ID
- Viewing inventory distribution across stages (G00, G02, I01, F01)
- Tracking fabric movement history
- Getting inventory summaries

Integrates with:
- finished_fabric_specs table (F IDs)
- greige_fabric_specs table (G IDs)
- fabric_inventory table (stage quantities)
- fabric_movements table (movement history)
"""

from flask import Blueprint, jsonify, request
import httpx
import os
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Create the blueprint
fabric_inquiry_bp = Blueprint("fabric_inquiry", __name__, url_prefix="/api/fabric-inquiry")

# Turso database connection
database_url = os.getenv("TURSO_DATABASE_URL", "").replace("libsql://", "https://")
auth_token = os.getenv("TURSO_AUTH_TOKEN", "")

# Create persistent HTTP client
turso_client = httpx.Client(
    headers={
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json"
    },
    timeout=30.0
)


def execute_sql(sql: str, params: List = None) -> dict:
    """
    Execute SQL on Turso database

    Args:
        sql: SQL query string
        params: Optional list of parameters for parameterized query

    Returns:
        Dict containing query results
    """
    try:
        payload = {"statements": [{"q": sql, "params": params or []}]}
        response = turso_client.post(database_url, json=payload)
        response.raise_for_status()
        result = response.json()

        # Handle list or dict response
        if isinstance(result, list) and len(result) > 0:
            result = result[0]

        return result
    except Exception as e:
        logger.error(f"SQL execution error: {e}")
        raise


@fabric_inquiry_bp.route("/search", methods=["POST"])
def search_fabric():
    """
    Search fabric inventory by ID

    Request body:
    {
        "fabric_id": "1234",
        "include_specs": true
    }

    Response:
    {
        "fabric_id": "1234",
        "fabric_type": "finished",
        "specs": {...},
        "inventory_by_stage": {
            "G00": {"yards": 100, "lbs": 50, "rolls": 2},
            ...
        },
        "total": {"yards": 500, "lbs": 250, "rolls": 10}
    }
    """
    try:
        data = request.json or {}
        fabric_id = data.get("fabric_id")
        include_specs = data.get("include_specs", True)

        if not fabric_id:
            return jsonify({"error": "fabric_id is required"}), 400

        # Find fabric type and specs
        fabric_type = None
        specs = {}

        # Try finished fabric first
        sql = """
            SELECT f_id, name, gsm, overall_width, yds_per_lb, construction, composition
            FROM finished_fabric_specs
            WHERE f_id = ?
        """
        result = execute_sql(sql, [str(fabric_id)])

        if result.get("results", {}).get("rows"):
            rows = result["results"].get("rows", [])
            if rows:
                row = rows[0]
                fabric_type = "finished"
                specs = {
                    "id": row[0],
                    "name": row[1],
                    "gsm": row[2],
                    "width": row[3],
                    "yds_per_lb": row[4],
                    "construction": row[5],
                    "composition": row[6]
                }

        # If not found, try greige
        if not fabric_type:
            sql = """
                SELECT g_id, g_base, construction, customer, knit_price
                FROM greige_fabric_specs
                WHERE g_id = ?
            """
            result = execute_sql(sql, [str(fabric_id)])

            if result.get("results", {}).get("rows"):
                rows = result["results"].get("rows", [])
                if rows:
                    row = rows[0]
                    fabric_type = "greige"
                    specs = {
                        "id": row[0],
                        "g_base": row[1],
                        "construction": row[2],
                        "customer": row[3],
                        "knit_price": row[4]
                    }

        if not fabric_type:
            return jsonify({"error": "Fabric not found in specs database"}), 404

        # Get inventory by stage
        sql = """
            SELECT stage,
                   SUM(quantity_yards) as total_yards,
                   SUM(quantity_lbs) as total_lbs,
                   SUM(rolls) as total_rolls,
                   COUNT(DISTINCT lot_number) as lot_count
            FROM fabric_inventory
            WHERE fabric_id = ?
            GROUP BY stage
        """
        inv_result = execute_sql(sql, [str(fabric_id)])

        inventory_by_stage = {}
        total_yards = 0
        total_lbs = 0
        total_rolls = 0

        if inv_result.get("results", {}).get("rows"):
            for row in inv_result["results"]["rows"]:
                stage = row[0]
                yards = row[1] or 0
                lbs = row[2] or 0
                rolls = row[3] or 0
                lot_count = row[4] or 0

                inventory_by_stage[stage] = {
                    "yards": yards,
                    "lbs": lbs,
                    "rolls": rolls,
                    "lots": lot_count
                }

                total_yards += yards
                total_lbs += lbs
                total_rolls += rolls

        # Build response
        response = {
            "fabric_id": fabric_id,
            "fabric_type": fabric_type,
            "inventory_by_stage": inventory_by_stage,
            "total": {
                "yards": total_yards,
                "lbs": total_lbs,
                "rolls": total_rolls
            }
        }

        if include_specs:
            response["specs"] = specs

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in search_fabric: {e}")
        return jsonify({"error": str(e)}), 500


@fabric_inquiry_bp.route("/by-style-range", methods=["GET"])
def get_by_style_range():
    """
    Get inventory for a range of fabric styles

    Query params:
    - start_id: Starting fabric ID
    - end_id: Ending fabric ID
    - fabric_type: 'finished' or 'greige' (default: 'finished')

    Response:
    {
        "fabrics": [
            {
                "fabric_id": "1234",
                "inventory": {
                    "G00": {"yards": 100, ...},
                    ...
                }
            }
        ],
        "count": 10
    }
    """
    try:
        start_id = request.args.get("start_id")
        end_id = request.args.get("end_id")
        fabric_type = request.args.get("fabric_type", "finished")

        if not start_id or not end_id:
            return jsonify({"error": "start_id and end_id are required"}), 400

        # Get fabric IDs in range
        if fabric_type == "finished":
            table = "finished_fabric_specs"
            id_col = "f_id"
        else:
            table = "greige_fabric_specs"
            id_col = "g_id"

        sql = f"""
            SELECT {id_col}
            FROM {table}
            WHERE {id_col} >= ? AND {id_col} <= ?
            ORDER BY {id_col}
            LIMIT 100
        """

        result = execute_sql(sql, [start_id, end_id])

        fabrics = []
        if result.get("results", {}).get("rows"):
            for row in result["results"]["rows"]:
                fabric_id = row[0]

                # Get inventory for this fabric
                inv_sql = """
                    SELECT stage, SUM(quantity_yards), SUM(quantity_lbs), SUM(rolls)
                    FROM fabric_inventory
                    WHERE fabric_id = ?
                    GROUP BY stage
                """
                inv_result = execute_sql(inv_sql, [fabric_id])

                inventory = {}
                if inv_result.get("results", {}).get("rows"):
                    for inv_row in inv_result["results"]["rows"]:
                        inventory[inv_row[0]] = {
                            "yards": inv_row[1],
                            "lbs": inv_row[2],
                            "rolls": inv_row[3]
                        }

                fabrics.append({
                    "fabric_id": fabric_id,
                    "inventory": inventory
                })

        return jsonify({"fabrics": fabrics, "count": len(fabrics)})

    except Exception as e:
        logger.error(f"Error in get_by_style_range: {e}")
        return jsonify({"error": str(e)}), 500


@fabric_inquiry_bp.route("/movement-history/<fabric_id>", methods=["GET"])
def get_movement_history(fabric_id: str):
    """
    Get movement history for a fabric

    Query params:
    - limit: Maximum number of records to return (default: 50)

    Response:
    {
        "fabric_id": "1234",
        "movements": [
            {
                "from_stage": "G00",
                "to_stage": "G02",
                "quantity_yards": 100,
                "quantity_lbs": 50,
                "date": "2025-10-18T...",
                "operator": "John Doe",
                ...
            }
        ],
        "count": 10
    }
    """
    try:
        limit = request.args.get("limit", 50, type=int)

        sql = """
            SELECT from_stage, to_stage, quantity_yards, quantity_lbs,
                   movement_date, operator, reference_doc, notes
            FROM fabric_movements
            WHERE fabric_id = ?
            ORDER BY movement_date DESC
            LIMIT ?
        """

        result = execute_sql(sql, [fabric_id, limit])

        movements = []
        if result.get("results", {}).get("rows"):
            for row in result["results"]["rows"]:
                movements.append({
                    "from_stage": row[0],
                    "to_stage": row[1],
                    "quantity_yards": row[2],
                    "quantity_lbs": row[3],
                    "date": row[4],
                    "operator": row[5],
                    "reference": row[6],
                    "notes": row[7]
                })

        return jsonify({
            "fabric_id": fabric_id,
            "movements": movements,
            "count": len(movements)
        })

    except Exception as e:
        logger.error(f"Error in get_movement_history: {e}")
        return jsonify({"error": str(e)}), 500


@fabric_inquiry_bp.route("/stages", methods=["GET"])
def get_stages():
    """
    Get list of production stages

    Response:
    [
        {"code": "G00", "name": "Raw Greige", "category": "greige"},
        ...
    ]
    """
    stages = [
        {"code": "G00", "name": "Raw Greige", "category": "greige", "order": 1},
        {"code": "G02", "name": "Processed Greige", "category": "greige", "order": 2},
        {"code": "I01", "name": "QC Inspection", "category": "inspection", "order": 3},
        {"code": "F01", "name": "Finished", "category": "finished", "order": 4}
    ]
    return jsonify(stages)


@fabric_inquiry_bp.route("/summary", methods=["GET"])
def get_inventory_summary():
    """
    Get overall inventory summary by stage

    Response:
    [
        {
            "stage": "G00",
            "fabric_type": "finished",
            "fabric_count": 10,
            "total_yards": 1000,
            "total_lbs": 500,
            "total_rolls": 20
        },
        ...
    ]
    """
    try:
        sql = """
            SELECT stage, fabric_type,
                   COUNT(DISTINCT fabric_id) as fabric_count,
                   SUM(quantity_yards) as total_yards,
                   SUM(quantity_lbs) as total_lbs,
                   SUM(rolls) as total_rolls
            FROM fabric_inventory
            GROUP BY stage, fabric_type
            ORDER BY
                CASE stage
                    WHEN 'G00' THEN 1
                    WHEN 'G02' THEN 2
                    WHEN 'I01' THEN 3
                    WHEN 'F01' THEN 4
                    ELSE 5
                END
        """

        result = execute_sql(sql)

        summary = []
        if result.get("results", {}).get("rows"):
            for row in result["results"]["rows"]:
                summary.append({
                    "stage": row[0],
                    "fabric_type": row[1],
                    "fabric_count": row[2],
                    "total_yards": row[3],
                    "total_lbs": row[4],
                    "total_rolls": row[5]
                })

        return jsonify(summary)

    except Exception as e:
        logger.error(f"Error in get_inventory_summary: {e}")
        return jsonify({"error": str(e)}), 500


@fabric_inquiry_bp.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "fabric_inquiry"})
