from typing import List, Dict, Any, Union, Optional, Tuple
import json
import yaml
from dataclasses import dataclass, field
from enum import Enum
import re


class LayerType(Enum):
    """Enumeration of available layer types."""
    SELF_SDPA = "self_sdpa"
    SELF_FLEX = "self_flex"
    DIFF_ATTN = "diff_attn"
    TOKEN_PARAM = "token_param"
    SPECTRAL = "spectral"


class MLPType(Enum):
    """Enumeration of available MLP types."""
    SWIGLU = "swiglu"
    RELU2 = "relu2"


@dataclass
class LayerSpec:
    """Specification for a single layer."""
    attention_type: str
    mlp_type: str
    layer_idx: int
    custom_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Validate attention and MLP types
        if self.attention_type not in [e.value for e in LayerType]:
            raise ValueError(f"Invalid attention type: {self.attention_type}")
        if self.mlp_type not in [e.value for e in MLPType]:
            raise ValueError(f"Invalid MLP type: {self.mlp_type}")


@dataclass
class LayerPattern:
    """Defines a pattern for layer configuration."""
    name: str
    description: str
    base_attention: str = "self_sdpa"
    base_mlp: str = "swiglu"
    overrides: List[Dict[str, Any]] = field(default_factory=list)
    
    def generate_layers(self, n_layers: int) -> List[LayerSpec]:
        """Generate layer specifications based on the pattern."""
        layers = []
        
        for i in range(n_layers):
            # Start with base configuration
            attention_type = self.base_attention
            mlp_type = self.base_mlp
            custom_params = {}
            
            # Apply overrides
            for override in self.overrides:
                if self._matches_condition(i, n_layers, override.get("condition", {})):
                    attention_type = override.get("attention_type", attention_type)
                    mlp_type = override.get("mlp_type", mlp_type)
                    custom_params.update(override.get("custom_params", {}))
            
            layers.append(LayerSpec(
                attention_type=attention_type,
                mlp_type=mlp_type,
                layer_idx=i,
                custom_params=custom_params
            ))
        
        return layers
    
    def _matches_condition(self, layer_idx: int, n_layers: int, condition: Dict[str, Any]) -> bool:
        """Check if a layer matches the given condition."""
        if not condition:
            return False
        
        # Layer index conditions
        if "layer_idx" in condition:
            if isinstance(condition["layer_idx"], list):
                if layer_idx not in condition["layer_idx"]:
                    return False
            elif layer_idx != condition["layer_idx"]:
                return False
        
        # Range conditions
        if "range" in condition:
            start, end = condition["range"]
            if not (start <= layer_idx < end):
                return False
        
        # Modulo conditions (for tiling patterns)
        if "every" in condition:
            every = condition["every"]
            offset = condition.get("offset", 0)
            if (layer_idx - offset) % every != 0:
                return False
        
        # Position-based conditions
        if "position" in condition:
            pos = condition["position"]
            if pos == "first" and layer_idx != 0:
                return False
            elif pos == "last" and layer_idx != n_layers - 1:
                return False
            elif pos == "middle" and layer_idx != n_layers // 2:
                return False
        
        # Fraction-based conditions
        if "fraction" in condition:
            frac_start, frac_end = condition["fraction"]
            start_idx = int(frac_start * n_layers)
            end_idx = int(frac_end * n_layers)
            if not (start_idx <= layer_idx < end_idx):
                return False
        
        return True


class LayerConfigBuilder:
    """Builder class for creating layer configurations."""
    
    def __init__(self):
        self.patterns = self._load_default_patterns()
    
    def _load_default_patterns(self) -> Dict[str, LayerPattern]:
        """Load default layer patterns."""
        patterns = {}
        
        # Standard uniform pattern
        patterns["uniform"] = LayerPattern(
            name="uniform",
            description="All layers use the same attention and MLP types",
            base_attention="self_sdpa",
            base_mlp="swiglu"
        )
        
        # Alternating pattern
        patterns["alternating"] = LayerPattern(
            name="alternating",
            description="Alternates between two attention types",
            base_attention="self_sdpa",
            base_mlp="swiglu",
            overrides=[
                {
                    "condition": {"every": 2, "offset": 1},
                    "attention_type": "token_param"
                }
            ]
        )
        
        # Sandwich pattern (special layers at beginning and end)
        patterns["sandwich"] = LayerPattern(
            name="sandwich",
            description="Special attention at first and last layers",
            base_attention="self_sdpa",
            base_mlp="swiglu",
            overrides=[
                {
                    "condition": {"position": "first"},
                    "attention_type": "spectral"
                },
                {
                    "condition": {"position": "last"},
                    "attention_type": "spectral"
                }
            ]
        )
        
        # Pyramid pattern (different attention in different sections)
        patterns["pyramid"] = LayerPattern(
            name="pyramid",
            description="Different attention types in different sections",
            base_attention="self_sdpa",
            base_mlp="swiglu",
            overrides=[
                {
                    "condition": {"fraction": [0.0, 0.3]},
                    "attention_type": "spectral"
                },
                {
                    "condition": {"fraction": [0.7, 1.0]},
                    "attention_type": "diff_attn"
                }
            ]
        )
        
        # Sparse special layers
        patterns["sparse_special"] = LayerPattern(
            name="sparse_special",
            description="Special attention every 4th layer",
            base_attention="self_sdpa",
            base_mlp="swiglu",
            overrides=[
                {
                    "condition": {"every": 4},
                    "attention_type": "token_param"
                }
            ]
        )
        
        return patterns
    
    def from_pattern(self, pattern_name: str, n_layers: int, **kwargs) -> List[LayerSpec]:
        """Generate layers from a predefined pattern."""
        if pattern_name not in self.patterns:
            raise ValueError(f"Unknown pattern: {pattern_name}")
        
        pattern = self.patterns[pattern_name]
        
        # Allow overriding base types
        if "base_attention" in kwargs:
            pattern.base_attention = kwargs["base_attention"]
        if "base_mlp" in kwargs:
            pattern.base_mlp = kwargs["base_mlp"]
        
        return pattern.generate_layers(n_layers)
    
    def from_lists(self, attention_types: List[str], mlp_types: List[str]) -> List[LayerSpec]:
        """Generate layers from explicit lists (current approach)."""
        if len(attention_types) != len(mlp_types):
            raise ValueError("attention_types and mlp_types must have the same length")
        
        return [
            LayerSpec(
                attention_type=att,
                mlp_type=mlp,
                layer_idx=i
            )
            for i, (att, mlp) in enumerate(zip(attention_types, mlp_types))
        ]
    
    def from_string_pattern(self, pattern: str, n_layers: int) -> List[LayerSpec]:
        """Generate layers from a string pattern like 'SDPA*3,TOKEN_PARAM,SDPA*2'."""
        layers = []
        layer_idx = 0
        
        # Parse pattern
        parts = [p.strip() for p in pattern.split(',')]
        
        for part in parts:
            if '*' in part:
                # Handle repetition like 'SDPA*3'
                layer_type, count = part.split('*')
                count = int(count)
            else:
                layer_type = part
                count = 1
            
            # Map string to actual types
            attention_type = self._map_string_to_attention(layer_type)
            mlp_type = "swiglu"  # Default MLP type
            
            for _ in range(count):
                if layer_idx >= n_layers:
                    break
                layers.append(LayerSpec(
                    attention_type=attention_type,
                    mlp_type=mlp_type,
                    layer_idx=layer_idx
                ))
                layer_idx += 1
        
        # Fill remaining layers with default if needed
        while layer_idx < n_layers:
            layers.append(LayerSpec(
                attention_type="self_sdpa",
                mlp_type="swiglu",
                layer_idx=layer_idx
            ))
            layer_idx += 1
        
        return layers[:n_layers]
    
    def _map_string_to_attention(self, layer_str: str) -> str:
        """Map string representation to attention type."""
        mapping = {
            "SDPA": "self_sdpa",
            "SELF_SDPA": "self_sdpa",
            "DIFF": "diff_attn",
            "DIFF_ATTN": "diff_attn",
            "TOKEN_PARAM": "token_param",
            "SPECTRAL": "spectral",
            "FLEX": "self_flex"
        }
        return mapping.get(layer_str.upper(), "self_sdpa")
    
    def from_json(self, json_path: str) -> List[LayerSpec]:
        """Load layer configuration from JSON file."""
        with open(json_path, 'r') as f:
            config = json.load(f)
        return self._parse_config_dict(config)
    
    def from_yaml(self, yaml_path: str) -> List[LayerSpec]:
        """Load layer configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        return self._parse_config_dict(config)
    
    def _parse_config_dict(self, config: Dict[str, Any]) -> List[LayerSpec]:
        """Parse configuration dictionary."""
        if "pattern" in config:
            # Pattern-based configuration
            pattern_name = config["pattern"]["name"]
            n_layers = config["pattern"]["n_layers"]
            kwargs = config["pattern"].get("params", {})
            return self.from_pattern(pattern_name, n_layers, **kwargs)
        
        elif "layers" in config:
            # Explicit layer specification
            layers = []
            for i, layer_config in enumerate(config["layers"]):
                layers.append(LayerSpec(
                    attention_type=layer_config["attention_type"],
                    mlp_type=layer_config.get("mlp_type", "swiglu"),
                    layer_idx=i,
                    custom_params=layer_config.get("custom_params", {})
                ))
            return layers
        
        elif "string_pattern" in config:
            # String pattern configuration
            pattern = config["string_pattern"]["pattern"]
            n_layers = config["string_pattern"]["n_layers"]
            return self.from_string_pattern(pattern, n_layers)
        
        else:
            raise ValueError("Invalid configuration format")
    
    def add_custom_pattern(self, pattern: LayerPattern):
        """Add a custom pattern to the builder."""
        self.patterns[pattern.name] = pattern
    
    def list_patterns(self) -> List[str]:
        """List available patterns."""
        return list(self.patterns.keys())
    
    def get_pattern_info(self, pattern_name: str) -> str:
        """Get information about a pattern."""
        if pattern_name not in self.patterns:
            return f"Pattern '{pattern_name}' not found"
        
        pattern = self.patterns[pattern_name]
        return f"{pattern.name}: {pattern.description}"


def create_example_configs():
    """Create example configuration files."""
    
    # JSON example
    json_config = {
        "pattern": {
            "name": "alternating",
            "n_layers": 12,
            "params": {
                "base_attention": "self_sdpa",
                "base_mlp": "swiglu"
            }
        }
    }
    
    with open("example_config.json", "w") as f:
        json.dump(json_config, f, indent=2)
    
    # YAML example
    yaml_config = {
        "layers": [
            {"attention_type": "spectral", "mlp_type": "swiglu"},
            {"attention_type": "self_sdpa", "mlp_type": "swiglu"},
            {"attention_type": "self_sdpa", "mlp_type": "swiglu"},
            {"attention_type": "token_param", "mlp_type": "swiglu"},
            {"attention_type": "self_sdpa", "mlp_type": "swiglu"},
            {"attention_type": "self_sdpa", "mlp_type": "swiglu"},
            {"attention_type": "diff_attn", "mlp_type": "swiglu"},
            {"attention_type": "self_sdpa", "mlp_type": "swiglu"},
        ]
    }
    
    with open("example_config.yaml", "w") as f:
        yaml.dump(yaml_config, f, default_flow_style=False)
    
    # String pattern example
    string_config = {
        "string_pattern": {
            "pattern": "SPECTRAL,SDPA*3,TOKEN_PARAM,SDPA*2,DIFF_ATTN,SDPA*4",
            "n_layers": 12
        }
    }
    
    with open("example_string_config.json", "w") as f:
        json.dump(string_config, f, indent=2)


if __name__ == "__main__":
    # Example usage
    builder = LayerConfigBuilder()
    
    print("Available patterns:")
    for pattern in builder.list_patterns():
        print(f"  - {builder.get_pattern_info(pattern)}")
    
    print("\nExample: Alternating pattern with 8 layers")
    layers = builder.from_pattern("alternating", 8)
    for layer in layers:
        print(f"Layer {layer.layer_idx}: {layer.attention_type} + {layer.mlp_type}")
    
    print("\nExample: String pattern")
    layers = builder.from_string_pattern("SPECTRAL,SDPA*2,TOKEN_PARAM,SDPA*3", 7)
    for layer in layers:
        print(f"Layer {layer.layer_idx}: {layer.attention_type} + {layer.mlp_type}")
    
    # Create example config files
    create_example_configs()
    print("\nExample configuration files created!") 