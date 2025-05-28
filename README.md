# Flexible Transformer Layer Configuration

This system provides multiple ways to configure transformer layers with different attention and MLP types, supporting complex patterns and tiling configurations.

## Features

- **Predefined Patterns**: Common layer arrangements like alternating, sandwich, pyramid
- **String Patterns**: Compact notation like `"SPECTRAL,SDPA*3,TOKEN_PARAM"`
- **Configuration Files**: JSON/YAML files for complex configurations
- **Custom Patterns**: Define your own reusable patterns with conditions
- **GUI Interface**: Streamlit app for visual configuration
- **Backward Compatibility**: Works with existing code

## Quick Start

### 1. Using Predefined Patterns

```python
from models.transformer.modeling_transformer import TransformerConfig

# Alternating pattern: SDPA and token_param every other layer
config = TransformerConfig.from_pattern(
    pattern_name="alternating",
    n_layers=12,
    hidden_size=512,
    n_heads=8
)

# Available patterns: uniform, alternating, sandwich, pyramid, sparse_special
```

### 2. Using String Patterns

```python
# Compact string notation
config = TransformerConfig.from_string_pattern(
    pattern="SPECTRAL,SDPA*3,TOKEN_PARAM,SDPA*2,DIFF_ATTN,SDPA*4",
    n_layers=12,
    hidden_size=512,
    n_heads=8
)
```

### 3. Using Configuration Files

**JSON Example:**
```json
{
  "pattern": {
    "name": "alternating",
    "n_layers": 12,
    "params": {
      "base_attention": "self_sdpa",
      "base_mlp": "swiglu"
    }
  }
}
```

**YAML Example:**
```yaml
layers:
  - attention_type: spectral
    mlp_type: swiglu
  - attention_type: self_sdpa
    mlp_type: swiglu
  - attention_type: token_param
    mlp_type: swiglu
    custom_params:
      attention:
        param_sharing: true
```

```python
config = TransformerConfig.from_config_file(
    config_file="my_config.yaml",
    hidden_size=512,
    n_heads=8
)
```

## Available Layer Types

### Attention Types
- `self_sdpa`: Standard scaled dot-product attention
- `diff_attn`: Differential attention
- `token_param`: Token-parameterized attention
- `spectral`: Spectral attention
- `self_flex`: Flexible attention (if implemented)

### MLP Types
- `swiglu`: SwiGLU activation
- `relu2`: ReLU² activation

## Pattern System

### Predefined Patterns

1. **uniform**: All layers use the same type
2. **alternating**: Alternates between two attention types
3. **sandwich**: Special attention at first and last layers
4. **pyramid**: Different attention in different sections
5. **sparse_special**: Special attention every N layers

### Custom Patterns

Create your own patterns with flexible conditions:

```python
from models.transformer.layer_config import LayerPattern

custom_pattern = LayerPattern(
    name="my_pattern",
    description="Custom mixed pattern",
    base_attention="self_sdpa",
    base_mlp="swiglu",
    overrides=[
        {
            "condition": {"every": 3},  # Every 3rd layer
            "attention_type": "spectral"
        },
        {
            "condition": {"fraction": [0.4, 0.6]},  # Middle 20%
            "attention_type": "token_param"
        },
        {
            "condition": {"position": "last"},  # Last layer
            "attention_type": "diff_attn"
        }
    ]
)
```

### Condition Types

- **every**: `{"every": 3, "offset": 1}` - Every 3rd layer with offset
- **position**: `{"position": "first|last|middle"}` - Specific positions
- **range**: `{"range": [start, end]}` - Layer index range
- **fraction**: `{"fraction": [0.2, 0.8]}` - Fraction of total layers
- **layer_idx**: `{"layer_idx": [0, 5, 10]}` - Specific layer indices

## GUI Interface

Launch the Streamlit GUI for visual configuration:

```bash
streamlit run models/transformer/layer_config_gui.py
```

Features:
- Visual layer pattern design
- Real-time preview
- Export to JSON/YAML/Python
- Pattern library
- Custom pattern creation

## Advanced Usage

### Custom Parameters per Layer

```python
config_dict = {
    "layers": [
        {
            "attention_type": "spectral",
            "mlp_type": "swiglu",
            "custom_params": {
                "attention": {"use_fft": True},
                "mlp": {"expansion_ratio": 4.0}
            }
        },
        # ... more layers
    ]
}

config = TransformerConfig(
    hidden_size=512,
    n_heads=8,
    layer_config=config_dict
)
```

### Programmatic Configuration

```python
from models.transformer.layer_config import LayerConfigBuilder, LayerSpec

builder = LayerConfigBuilder()

# From lists (original method)
layers = builder.from_lists(
    attention_types=["spectral", "self_sdpa", "token_param"],
    mlp_types=["swiglu", "swiglu", "swiglu"]
)

# From string pattern
layers = builder.from_string_pattern("SPECTRAL,SDPA*2", n_layers=3)

# From predefined pattern
layers = builder.from_pattern("alternating", n_layers=8)
```

## Examples

### Example 1: Research Configuration
```python
# For research: spectral attention every 4th layer, diff attention in middle
config = TransformerConfig.from_string_pattern(
    pattern="SPECTRAL,SDPA*3,SPECTRAL,SDPA*2,DIFF_ATTN*2,SDPA*2,SPECTRAL,SDPA*3",
    n_layers=16,
    hidden_size=768,
    n_heads=12
)
```

### Example 2: Efficient Configuration
```python
# For efficiency: mostly SDPA with occasional token_param
config = TransformerConfig.from_pattern(
    pattern_name="sparse_special",
    n_layers=24,
    hidden_size=1024,
    n_heads=16,
    base_attention="self_sdpa",
    # Override pattern to use token_param every 6th layer
)
```

### Example 3: Hybrid Architecture
```python
# Complex hybrid: different sections for different tasks
custom_pattern = LayerPattern(
    name="hybrid",
    description="Hybrid architecture for multi-task learning",
    base_attention="self_sdpa",
    base_mlp="swiglu",
    overrides=[
        {"condition": {"range": [0, 4]}, "attention_type": "spectral"},      # Early layers: spectral
        {"condition": {"range": [4, 8]}, "attention_type": "self_sdpa"},     # Middle: standard
        {"condition": {"range": [8, 10]}, "attention_type": "token_param"},  # Late: token param
        {"condition": {"range": [10, 12]}, "attention_type": "diff_attn"},   # Final: differential
    ]
)
```

## Migration from Old System

The new system is fully backward compatible:

```python
# Old way (still works)
config = TransformerConfig(
    attention_types=["self_sdpa", "token_param", "self_sdpa"],
    mlp_types=["swiglu", "swiglu", "swiglu"],
    # ... other params
)

# New way (equivalent)
config = TransformerConfig.from_string_pattern(
    pattern="SDPA,TOKEN_PARAM,SDPA",
    n_layers=3,
    # ... other params
)
```

## Performance Considerations

- **Pattern Resolution**: Patterns are resolved once during config creation
- **Memory**: No additional memory overhead during training
- **Flexibility**: Custom parameters allow fine-tuning per layer
- **Validation**: All configurations are validated at creation time

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure `layer_config.py` is in the same directory
2. **Pattern Not Found**: Check available patterns with `builder.list_patterns()`
3. **Invalid Layer Type**: Verify attention/MLP types are supported
4. **File Not Found**: Check file paths for configuration files

### Debug Information

```python
# Get detailed layer information
print(config.get_layer_info())

# Inspect layer specifications
for i, spec in enumerate(config.layer_specs):
    print(f"Layer {i}: {spec.attention_type} + {spec.mlp_type}")
    if spec.custom_params:
        print(f"  Custom params: {spec.custom_params}")
```

## Contributing

To add new attention or MLP types:

1. Add the type to `LayerType` or `MLPType` enum in `layer_config.py`
2. Update the layer creation logic in `TransformerBlock.__init__`
3. Add validation in `LayerSpec.__post_init__`
4. Update documentation and examples

## API Reference

### TransformerConfig Methods

- `from_pattern(pattern_name, n_layers, **kwargs)`: Create from predefined pattern
- `from_string_pattern(pattern, n_layers, **kwargs)`: Create from string pattern
- `from_config_file(config_file, **kwargs)`: Create from JSON/YAML file
- `get_layer_info()`: Get configuration summary

### LayerConfigBuilder Methods

- `from_pattern(pattern_name, n_layers, **kwargs)`: Generate from pattern
- `from_string_pattern(pattern, n_layers)`: Generate from string
- `from_lists(attention_types, mlp_types)`: Generate from lists
- `from_json(json_path)`: Load from JSON file
- `from_yaml(yaml_path)`: Load from YAML file
- `add_custom_pattern(pattern)`: Add custom pattern
- `list_patterns()`: List available patterns

### LayerPattern Class

- `generate_layers(n_layers)`: Generate layer specifications
- Supports complex condition matching for flexible patterns 