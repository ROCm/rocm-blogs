import yaml
from typing import List, Dict, Any
import sys

class ListFlowStyleRepresenter:
    """
    Custom representer to force specific flow style for nested lists
    """
    def __init__(self):
        self.add_representer()

    def represent_list(self, dumper, data):
        # Check if this is a nested list containing strings/patterns
        if data and isinstance(data[0], str):
            return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=False)

    def add_representer(self):
        yaml.add_representer(list, self.represent_list)

def load_yaml(file_path: str) -> Any:
    """
    Load YAML file with error handling.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except yaml.scanner.ScannerError as e:
        print(f"\nYAML Syntax Error in {file_path}:")
        print(f"Line {e.problem_mark.line + 1}, Column {e.problem_mark.column + 1}")
        print(f"Error: {e.problem}")
        return None
    except Exception as e:
        print(f"\nError loading {file_path}:")
        print(str(e))
        return None

def merge_matrix_entries(default_entries: List[Dict], user_entries: List[Dict]) -> List[Dict]:
    """
    Merge matrix entries, combining sources only for entries with matching names.
    - Skip merging if the 'sources' list in the user entries is empty.
    - If the main configuration has an empty 'sources' list, replace it with the user's list.
    """
    result = default_entries.copy()
    default_map = {entry.get('name'): entry for entry in result}

    for user_entry in user_entries:
        user_name = user_entry.get('name')
        if user_name and user_name in default_map:
            if 'sources' in user_entry:
                user_sources = user_entry['sources']

                # Skip merging if user sources are empty
                if not user_sources or all(not source for source in user_sources):
                    continue

                # Check the main configuration's sources
                default_sources = default_map[user_name].get('sources', [])

                # If the main config's sources are empty, replace them
                if not default_sources or all(not source for source in default_sources):
                    default_map[user_name]['sources'] = user_sources
                else:
                    # Otherwise, merge the lists
                    default_map[user_name]['sources'].extend(user_sources)

    return result

def merge_configs(default: Any, user: Any) -> Any:
    """
    Recursively merge two configurations.
    """
    if user is None:
        return default

    if isinstance(default, dict) and isinstance(user, dict):
        result = default.copy()
        for key, value in user.items():
            if key in result:
                result[key] = merge_configs(result[key], value)
            else:
                result[key] = value
        return result

    if isinstance(default, list) and isinstance(user, list):
        if (len(default) > 0 and isinstance(default[0], dict) and
            'name' in default[0] and 'sources' in default[0]):
            return merge_matrix_entries(default, user)
        return default + user

    return user

def save_yaml(data: Dict, file_path: str) -> bool:
    """
    Save YAML file with custom formatting.
    """
    try:
        # Initialize custom representer
        ListFlowStyleRepresenter()

        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        return True
    except Exception as e:
        print(f"\nError saving {file_path}:")
        print(str(e))
        return False

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Merge spellcheck YAML configurations')
    parser.add_argument('template', help='Path to template YAML configuration')
    parser.add_argument('local', help='Path to local YAML configuration')
    parser.add_argument('output', help='Path to save merged configuration')

    args = parser.parse_args()

    # Load configurations
    template_config = load_yaml(args.template)
    if template_config is None:
        sys.exit(1)

    local_config = load_yaml(args.local)
    if local_config is None:
        sys.exit(1)

    # Merge configurations
    merged_config = merge_configs(template_config, local_config)

    # Save merged configuration
    if not save_yaml(merged_config, args.output):
        sys.exit(1)

    print(f"\nSuccessfully merged configurations into {args.output}")
