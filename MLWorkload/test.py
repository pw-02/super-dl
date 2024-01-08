import sys

# Specify the path to remove
path_to_remove = '/workspaces/super-dl/MLWorklaods/Classification'

# Check if the path is in sys.path before removing
if path_to_remove in sys.path:
    sys.path.remove(path_to_remove)

# Print the updated sys.path
print(sys.path)