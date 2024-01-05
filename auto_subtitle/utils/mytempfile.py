import tempfile
import os
import shutil

class MyTempFile:
    """
    A context manager for creating a temporary file in current directory, copying the content from
    a specified file, and handling cleanup operations upon exiting the context.

    Usage:
    ```python
    with MyTempFile(file_path) as temp_file_manager:
        # Access the temporary file using temp_file_manager.tmp_file
        # ...
    # The temporary file is automatically closed and removed upon exiting the context.
    ```

    Args:
    - file_path (str): The path to the file whose content will be copied to the temporary file.
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.tmp_file = None
        self.tmp_file_path = None

    def __enter__(self):
        self.tmp_file = tempfile.NamedTemporaryFile('w', dir='.', delete=False)
        self.tmp_file_path = os.path.relpath(self.tmp_file.name, '.')
        shutil.copyfile(self.file_path, self.tmp_file_path)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.tmp_file.close()
        if os.path.isfile(self.tmp_file_path):
            os.remove(self.tmp_file_path)
