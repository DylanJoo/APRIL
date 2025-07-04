import argparse
import yaml
import os
from typing import Any, Dict
from pprint import pprint
import importlib.resources as pkg_resources

class ConfigManager:
    def __init__(
        self, 
        default_config_path = pkg_resources.files("reranking.configs").joinpath("default_config.yaml")
    ):
        self.config = self.load_yaml(default_config_path)
        self.parse_and_override()

    def load_yaml(self, path: str) -> Dict[str, Any]:
        path = os.path.abspath(path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def parse_and_override(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")

        # Capture all --key=value args
        args, unknown = parser.parse_known_args()

        if args.config:
            self.config.update(self.load_yaml(args.config))

        overrides = self.parse_unknown_args(unknown)
        self.apply_overrides(self.config, overrides)

    def parse_unknown_args(self, args: list) -> Dict[str, Any]:
        result = {}
        for arg in args:
            if not arg.startswith("--") or "=" not in arg:
                continue
            key, value = arg[2:].split("=", 1)
            self.insert_nested_key(result, key, self._infer_type(value))
        return result

    def insert_nested_key(self, d: Dict[str, Any], key: str, value: Any):
        """Insert value into nested dictionary using dot notation."""
        keys = key.split(".")
        current = d
        for k in keys[:-1]:
            if k not in current or not isinstance(current[k], dict):
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value

    def apply_overrides(self, base: Dict[str, Any], overrides: Dict[str, Any]):
        """Recursively apply overrides to the base config."""
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                self.apply_overrides(base[key], value)
            else: 
                base[key] = value

    def _infer_type(self, value: str) -> Any:
        if value.lower() in {"true", "false"}:
            print(f"Converting '{value}' to boolean")
            return value.lower() == "true"
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            return value

    def get_config(self, return_dict=False) -> Dict[str, Any]:
        if return_dict:
            return self.config

        def dict_to_namespace(d):
            namespace = argparse.Namespace()
            for key, value in d.items():
                if isinstance(value, dict):
                    setattr(namespace, key, dict_to_namespace(value))
                else:
                    setattr(namespace, key, value)
            return namespace
        return dict_to_namespace(self.config)
