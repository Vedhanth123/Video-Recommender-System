"""
Database clearing utility for Emotion-Based Video Recommendation system.
This script clears all data from the database tables without dropping the tables.
"""

import sqlite3
from config import DATABASE_PATH

def clear_all_tables():
    """
    Clear all data from all tables in the database.
    """
    try:
        # Connect to the database
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        print("Found the following tables:")
        for table in tables:
            table_name = table[0]
            print(f"- {table_name}")
        
        # Delete data from each table
        for table in tables:
            table_name = table[0]
            if table_name != 'sqlite_sequence':  # Skip internal SQLite tables
                try:
                    cursor.execute(f"DELETE FROM {table_name}")
                    print(f"Cleared data from table: {table_name}")
                except Exception as e:
                    print(f"Error clearing table {table_name}: {e}")
        
        # Commit the changes
        conn.commit()
        print("\nAll tables have been cleared successfully!")
        
        # Vacuum the database to reclaim free space
        print("Optimizing database size...")
        conn.execute("VACUUM")
        print("Database optimization complete.")
        
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    # Confirm before proceeding
    print("WARNING: This will delete ALL data from ALL tables in the database.")
    print("This action cannot be undone.")
    confirmation = input("Do you want to proceed? (yes/no): ")
    
    if confirmation.lower() == "yes":
        clear_all_tables()
    else:
        print("Operation cancelled.")
