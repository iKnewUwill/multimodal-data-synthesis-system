#!/usr/bin/env python3
"""
Migration script to import existing JSON result files into SQLite database.

This script:
- Reads all .json files from data/outputs/
- Parses each JSON file as FinancialTaskResult
- Inserts into SQLite database using DatabaseManager
- Renames processed files to .json.migrated
- Provides progress feedback and handles errors gracefully

Usage:
    python scripts/migrate_to_sqlite.py [--dry-run]
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional

# Setup path for project imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.database import DatabaseManager
from src.models import FinancialTaskResult, FinancialQAResult, TaskStatus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class MigrationError(Exception):
    """Custom exception for migration errors"""
    pass


def parse_task_status(status_value: str) -> TaskStatus:
    """Parse status string to TaskStatus enum"""
    status_mapping = {
        "待处理": TaskStatus.PENDING,
        "处理中": TaskStatus.PROCESSING,
        "已完成": TaskStatus.COMPLETED,
        "处理失败": TaskStatus.FAILED,
        "pending": TaskStatus.PENDING,
        "processing": TaskStatus.PROCESSING,
        "completed": TaskStatus.COMPLETED,
        "failed": TaskStatus.FAILED,
    }
    
    # Try direct mapping first
    if status_value in status_mapping:
        return status_mapping[status_value]
    
    # Try case-insensitive matching
    for key, value in status_mapping.items():
        if key.lower() == status_value.lower():
            return value
    
    # Default to FAILED for unknown status
    logger.warning(f"Unknown status '{status_value}', defaulting to FAILED")
    return TaskStatus.FAILED


def parse_financial_task_result(data: dict, file_path: Path) -> Tuple[Optional[FinancialTaskResult], Optional[str]]:
    """
    Parse JSON data into FinancialTaskResult model.
    
    Returns:
        Tuple of (FinancialTaskResult or None, error_message or None)
    """
    try:
        # Handle status field - could be string or enum
        status = data.get("status", "处理失败")
        if isinstance(status, str):
            data["status"] = parse_task_status(status)
        elif not isinstance(status, TaskStatus):
            data["status"] = TaskStatus.FAILED
        
        # Parse qa_pairs if present
        if "qa_pairs" in data and data["qa_pairs"]:
            parsed_qa_pairs = []
            for qa in data["qa_pairs"]:
                try:
                    # Handle datetime parsing
                    if "created_at" in qa and isinstance(qa["created_at"], str):
                        try:
                            qa["created_at"] = datetime.fromisoformat(qa["created_at"])
                        except ValueError:
                            qa["created_at"] = datetime.now()
                    parsed_qa_pairs.append(FinancialQAResult(**qa))
                except Exception as e:
                    logger.warning(f"Failed to parse QA pair in {file_path}: {e}")
                    # Continue with other QA pairs
            data["qa_pairs"] = parsed_qa_pairs
        
        # Parse completed_at if present
        if "completed_at" in data and data["completed_at"]:
            if isinstance(data["completed_at"], str):
                try:
                    data["completed_at"] = datetime.fromisoformat(data["completed_at"])
                except ValueError:
                    data["completed_at"] = None
        
        # Create the result object
        result = FinancialTaskResult(**data)
        return result, None
        
    except Exception as e:
        error_msg = f"Validation error: {str(e)}"
        return None, error_msg


def get_json_files(output_dir: Path) -> List[Path]:
    """Get all .json files from output directory, excluding already migrated files"""
    json_files = []
    
    if not output_dir.exists():
        logger.warning(f"Output directory does not exist: {output_dir}")
        return json_files
    
    for file_path in output_dir.glob("*.json"):
        # Skip already migrated files
        if file_path.suffix == ".migrated":
            continue
        # Skip files that are .json.migrated (double extension)
        if file_path.name.endswith(".json.migrated"):
            continue
        json_files.append(file_path)
    
    # Sort by name for consistent ordering
    json_files.sort()
    return json_files


def check_database_has_data(db_manager: DatabaseManager) -> bool:
    """Check if database already contains data"""
    try:
        results = db_manager.get_all_results()
        return len(results) > 0
    except Exception:
        # If we can't check, assume empty
        return False


def migrate_file(
    file_path: Path,
    db_manager: DatabaseManager,
    dry_run: bool = False
) -> Tuple[bool, str]:
    """
    Migrate a single JSON file to database.
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        # Read and parse JSON
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Parse as FinancialTaskResult
        result, error = parse_financial_task_result(data, file_path)
        
        if error:
            return False, f"Parse error: {error}"
        
        if result is None:
            return False, "Failed to create FinancialTaskResult"
        
        # Save to database (or simulate in dry-run)
        if dry_run:
            logger.info(f"[DRY-RUN] Would save {result.task_id} to database")
        else:
            db_manager.save_result(result)
            logger.info(f"Saved {result.task_id} to database")
        
        # Rename file to .json.migrated
        if not dry_run:
            migrated_path = file_path.with_suffix('.json.migrated')
            file_path.rename(migrated_path)
            logger.info(f"Renamed {file_path.name} -> {migrated_path.name}")
        
        return True, "Success"
        
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON format: {e}"
    except Exception as e:
        return False, f"Error: {str(e)}"


def main():
    """Main migration function"""
    parser = argparse.ArgumentParser(
        description="Migrate JSON result files to SQLite database"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without making modifications"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/outputs",
        help="Directory containing JSON files to migrate (default: data/outputs)"
    )
    
    args = parser.parse_args()
    
    # Resolve output directory
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    
    logger.info("=" * 60)
    logger.info("Starting migration to SQLite database")
    if args.dry_run:
        logger.info("[DRY-RUN MODE - No changes will be made]")
    logger.info("=" * 60)
    
    # Initialize DatabaseManager
    try:
        db_manager = DatabaseManager()
        logger.info("DatabaseManager initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize DatabaseManager: {e}")
        sys.exit(1)
    
    # Check if database already has data
    if check_database_has_data(db_manager):
        logger.error("Database already contains data. Aborting migration to prevent duplicates.")
        logger.error("Please backup or clear the existing database first.")
        sys.exit(1)
    
    # Get list of JSON files
    json_files = get_json_files(output_dir)
    
    if not json_files:
        logger.info("No JSON files found to migrate.")
        sys.exit(0)
    
    logger.info(f"Found {len(json_files)} JSON file(s) to migrate")
    
    # Migrate each file
    successful = 0
    skipped = 0
    errors = []
    
    for idx, file_path in enumerate(json_files, 1):
        logger.info(f"Migrating {idx}/{len(json_files)}: {file_path.name}")
        
        success, message = migrate_file(file_path, db_manager, dry_run=args.dry_run)
        
        if success:
            successful += 1
        else:
            skipped += 1
            errors.append((file_path.name, message))
            logger.warning(f"Skipped {file_path.name}: {message}")
    
    # Print summary
    logger.info("=" * 60)
    logger.info("MIGRATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total files processed: {len(json_files)}")
    logger.info(f"Successfully migrated: {successful}")
    logger.info(f"Skipped (with errors): {skipped}")
    
    if errors:
        logger.info("-" * 60)
        logger.info("Errors:")
        for filename, error in errors:
            logger.info(f"  - {filename}: {error}")
    
    logger.info("=" * 60)
    
    if args.dry_run:
        logger.info("[DRY-RUN] No actual changes were made.")
        logger.info("Run without --dry-run to perform the migration.")
    
    sys.exit(0 if skipped == 0 else 1)


if __name__ == "__main__":
    main()
