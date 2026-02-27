import sqlite3
import time
import json
import threading


class TelemetryLogger:
    def __init__(self, db_name="logistics_telemetry.db", batch_size=10):
        self.db_name = db_name
        self.batch_size = batch_size
        self.buffer = []
        self.lock = threading.Lock()

        self.conn = sqlite3.connect(self.db_name, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")

        self.cursor = self.conn.cursor()
        self._initialize_table()

    def _initialize_table(self):
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS mission_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                agent_id TEXT,
                current_x INTEGER,
                current_y INTEGER,
                traffic_prob REAL,
                system_status TEXT,
                metadata TEXT
            )
        """)

        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON mission_logs(timestamp)
        """)

        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_agent
            ON mission_logs(agent_id)
        """)

        self.conn.commit()

    # ----------------------------------------------------
    # Logging
    # ----------------------------------------------------

    def log_state(self, agent_id, x, y, traffic_prob, status, metadata=None):
        entry = (
            time.time(),
            agent_id,
            x,
            y,
            traffic_prob,
            status,
            json.dumps(metadata) if metadata else None
        )

        with self.lock:
            self.buffer.append(entry)

            if len(self.buffer) >= self.batch_size:
                self._flush_internal()

    def flush(self):
        with self.lock:
            self._flush_internal()

    def _flush_internal(self):
        if not self.buffer:
            return

        self.cursor.executemany("""
            INSERT INTO mission_logs
            (timestamp, agent_id, current_x, current_y,
             traffic_prob, system_status, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, self.buffer)

        self.conn.commit()
        self.buffer.clear()

    # ----------------------------------------------------
    # Retrieval
    # ----------------------------------------------------

    def fetch_recent_logs(self, limit=10):
        self.cursor.execute("""
            SELECT timestamp, agent_id, current_x,
                   current_y, traffic_prob, system_status, metadata
            FROM mission_logs
            ORDER BY id DESC
            LIMIT ?
        """, (limit,))

        rows = self.cursor.fetchall()

        return [
            (
                r[0],
                r[1],
                r[2],
                r[3],
                r[4],
                r[5],
                json.loads(r[6]) if r[6] else None
            )
            for r in rows
        ]

    # ----------------------------------------------------
    # Analytics
    # ----------------------------------------------------

    def get_agent_statistics(self, agent_id):
        self.cursor.execute("""
            SELECT COUNT(*), AVG(traffic_prob)
            FROM mission_logs
            WHERE agent_id = ?
        """, (agent_id,))

        count, avg_risk = self.cursor.fetchone()

        return {
            "total_records": count or 0,
            "avg_delay_risk": float(avg_risk) if avg_risk else 0.0
        }

    def get_system_statistics(self):
        self.cursor.execute("""
            SELECT COUNT(*), AVG(traffic_prob)
            FROM mission_logs
        """)

        count, avg_risk = self.cursor.fetchone()

        return {
            "total_records": count or 0,
            "avg_delay_risk": float(avg_risk) if avg_risk else 0.0
        }

    # ----------------------------------------------------
    # Cleanup
    # ----------------------------------------------------

    def close(self):
        self.flush()
        self.conn.close()