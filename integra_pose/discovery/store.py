from __future__ import annotations

import json
import sqlite3
import uuid
import zlib
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def pack(value):
    return zlib.compress(json.dumps(value, allow_nan=False, ensure_ascii=False).encode('utf-8'))


def unpack(value):
    return json.loads(zlib.decompress(value))


class Workspace:
    """One SQLite file; immutable runs, separate reversible review history."""

    def __init__(self, path):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            version = db.execute('PRAGMA user_version').fetchone()[0]
            if version not in (0, 1):
                raise ValueError(f'Unsupported discovery workspace version: {version}')
            db.executescript('''
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, created TEXT NOT NULL,
                    payload BLOB NOT NULL, archived INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS edits (
                    id INTEGER PRIMARY KEY, run TEXT NOT NULL, payload BLOB NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1);
                CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, payload BLOB NOT NULL);
                PRAGMA user_version=1;
            ''')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=15)
        try:
            with db:
                yield db
        finally:
            db.close()

    def add_run(self, payload, name=None):
        rows = payload.get('rows', [])
        if not rows or len({r['row_id'] for r in rows}) != len(rows):
            raise ValueError('Discovery runs require non-empty, unique observation identities.')
        run_id = payload.get('run_id') or uuid.uuid4().hex
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as db:
            db.execute('INSERT INTO runs(id,name,created,payload) VALUES(?,?,?,?)',
                       (run_id, name or f'Run {len(self.runs()) + 1}', now, pack(payload)))
        self.set_state('active_run', run_id)
        return run_id

    def runs(self, include_archived=True):
        with self.connect() as db:
            return [dict(zip(('id', 'name', 'created', 'bytes', 'archived'), row)) for row in db.execute(
                'SELECT id,name,created,length(payload),archived FROM runs '
                + ('' if include_archived else 'WHERE archived=0 ') + 'ORDER BY created')]

    def load_run(self, run_id):
        with self.connect() as db:
            row = db.execute('SELECT payload FROM runs WHERE id=?', (run_id,)).fetchone()
        if row is None:
            raise ValueError('Selected discovery run is missing.')
        return unpack(row[0])

    def rename(self, run_id, name):
        if not name.strip():
            raise ValueError('A run name cannot be empty.')
        with self.connect() as db:
            db.execute('UPDATE runs SET name=? WHERE id=?', (name.strip(), run_id))

    def archive(self, run_id, value=True):
        with self.connect() as db:
            db.execute('UPDATE runs SET archived=? WHERE id=?', (int(value), run_id))

    def delete_run(self, run_id):
        if run_id == self.state('active_run'):
            raise ValueError('Select another run before deleting this one. Archive it to keep a recoverable copy.')
        with self.connect() as db:
            db.execute('DELETE FROM edits WHERE run=?', (run_id,))
            db.execute('DELETE FROM runs WHERE id=?', (run_id,))
            db.execute('DELETE FROM state WHERE key=?', ('view:'+run_id,))
        with self.connect() as db:
            db.execute('VACUUM')

    def edit(self, run_id, row_ids, label, reviewer, reason='', status='reviewed'):
        if not reviewer.strip() or not label.strip():
            raise ValueError('Enter reviewer initials and a non-empty annotation label.')
        if status not in ('reviewed', 'uncertain', 'artifact', 'excluded'):
            raise ValueError('Unknown review status.')
        ids = sorted(set(row_ids))
        valid = {row['row_id'] for row in self.load_run(run_id)['rows']}
        if not ids or not set(ids).issubset(valid):
            raise ValueError('The selection contains no valid observations from this run.')
        payload = dict(rows=ids, label=label.strip(), reviewer=reviewer.strip(),
                       reason=reason.strip(), status=status,
                       timestamp=datetime.now(timezone.utc).isoformat())
        with self.connect() as db:
            db.execute('UPDATE edits SET active=-1 WHERE run=? AND active=0', (run_id,))
            db.execute('INSERT INTO edits(run,payload) VALUES(?,?)', (run_id, pack(payload)))

    def edits(self, run_id):
        with self.connect() as db:
            return [dict(unpack(blob), edit_id=eid, active=active == 1, superseded=active == -1) for eid, blob, active in
                    db.execute('SELECT id,payload,active FROM edits WHERE run=? ORDER BY id', (run_id,))]

    def undo(self, run_id):
        with self.connect() as db:
            db.execute('UPDATE edits SET active=0 WHERE id=(SELECT max(id) FROM edits WHERE run=? AND active=1)', (run_id,))

    def redo(self, run_id):
        with self.connect() as db:
            db.execute('UPDATE edits SET active=1 WHERE id=(SELECT min(id) FROM edits WHERE run=? AND active=0)', (run_id,))

    def reviewed_rows(self, run_id, payload=None):
        rows = [dict(row) for row in (payload or self.load_run(run_id))['rows']]
        by_id = {r['row_id']: r for r in rows}
        for row in rows:
            row['review_label'] = row['cluster_label']
            row['review_status'] = 'unreviewed'
        for edit in self.edits(run_id):
            if edit['active']:
                for row_id in edit['rows']:
                    by_id[row_id].update(review_label=edit['label'], review_status=edit['status'])
        return rows

    def set_state(self, key, value):
        with self.connect() as db:
            db.execute('INSERT OR REPLACE INTO state VALUES(?,?)', (key, pack(value)))

    def state(self, key, default=None):
        with self.connect() as db:
            row = db.execute('SELECT payload FROM state WHERE key=?', (key,)).fetchone()
        return unpack(row[0]) if row else default

    def clone(self, destination):
        destination = Path(destination).resolve()
        if destination == self.path:
            return self
        if destination.exists():
            raise FileExistsError(f'Discovery destination already exists: {destination}')
        destination.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as source:
            target = sqlite3.connect(destination)
            try:
                source.backup(target)
            finally:
                target.close()
        return Workspace(destination)
