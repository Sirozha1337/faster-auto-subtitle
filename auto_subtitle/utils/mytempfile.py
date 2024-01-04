import tempfile
import os
import shutil

class MyTempFile:
    def __init__(self, file_path):
        self.file_path = file_path

    def __enter__(self):
        self.tmp_file = tempfile.NamedTemporaryFile('w', dir='.', delete=False)
        self.tmp_file_path = os.path.relpath(self.tmp_file.name, '.')
        shutil.copyfile(self.file_path, self.tmp_file_path)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.tmp_file.close()
        if os.path.isfile(self.tmp_file_path):
            os.remove(self.tmp_file_path)
