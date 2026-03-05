import sqlite3
import os

def export_to_markdown():
    db_path = 'attendance.db'
    output_dir = 'view_tables'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row['name'] for row in cursor.fetchall()]
    
    for table in tables:
        print(f"Exporting table: {table}...")
        cursor.execute(f"SELECT * FROM {table}")
        rows = cursor.fetchall()
        
        if not rows:
            with open(os.path.join(output_dir, f"{table}.md"), 'w') as f:
                f.write(f"# Table: {table}\n\nNo data found in this table.")
            continue
            
        columns = rows[0].keys()
        
        md_content = f"# Table: {table}\n\n"
        # Headers
        md_content += "| " + " | ".join(columns) + " |\n"
        # Separator
        md_content += "| " + " | ".join(["---"] * len(columns)) + " |\n"
        
        # Rows
        for row in rows:
            values = [str(val).replace('\n', '<br>') if val is not None else "" for val in row]
            md_content += "| " + " | ".join(values) + " |\n"
            
        with open(os.path.join(output_dir, f"{table}.md"), 'w', encoding='utf-8') as f:
            f.write(md_content)
            
    conn.close()
    print(f"\n✅ All tables exported to the '{output_dir}' folder.")

if __name__ == "__main__":
    export_to_markdown()
