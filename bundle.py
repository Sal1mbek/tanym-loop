import os
import fnmatch

def is_text_file(filepath):
    try:
        with open(filepath, 'rb') as f:
            chunk = f.read(1024)
            if b'\0' in chunk:
                return False
            return True
    except:
        return False

exclude_dirs = {'__pycache__', 'flagged', 'build', 'dist', '.eggs', 'venv', '.vscode', '.idea', '.pytest_cache', '.git'}

def should_exclude_file(filepath):
    file = os.path.basename(filepath)
    # Specific files
    if file in {'db.sqlite3', 'codebase_bundle.md', 'bundle.py'}:
        return True
    # Pattern matches
    if fnmatch.fnmatch(file, '*.py[co]') or fnmatch.fnmatch(file, '*$py.class') or fnmatch.fnmatch(file, '*.so') or fnmatch.fnmatch(file, '*.log') or fnmatch.fnmatch(file, '*.swp') or fnmatch.fnmatch(file, '.DS_Store') or fnmatch.fnmatch(file, '.coverage') or fnmatch.fnmatch(file, '*.bak') or fnmatch.fnmatch(file, '*.csv'):
        return True
    if file.startswith('.env'):
        return True
    # Specific paths
    if 'backend/staticfiles' in filepath or 'backend/media' in filepath:
        return True
    if ('data/cache' in filepath or 'data/output' in filepath or 'data/processed' in filepath) and not filepath.endswith('.gitkeep'):
        return True
    return False

with open('codebase_bundle.md', 'w') as out:
    for root, dirs, files in os.walk('.'):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for file in files:
            filepath = os.path.join(root, file)
            if should_exclude_file(filepath):
                continue
            if is_text_file(filepath):
                out.write(f'# File: {filepath}\n\n')
                try:
                    with open(filepath, 'r') as f:
                        content = f.read()
                        if filepath.endswith('.py'):
                            out.write('```python\n')
                        else:
                            out.write('```\n')
                        out.write(content)
                        out.write('\n```\n\n')
                except UnicodeDecodeError:
                    pass  # skip if decode error
