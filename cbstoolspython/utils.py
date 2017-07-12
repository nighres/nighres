def create_dir(some_directory):
    """
    Create directory recursively if it does not exist
      - uses os.mkdirs
    """
    import os
    if not os.path.exists(some_directory):
        os.makedirs(some_directory)
