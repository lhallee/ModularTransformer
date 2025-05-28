"""
Examples demonstrating the flexible layer configuration system for transformers.

This module shows various ways to configure transformer layers using:
1. Predefined patterns
2. String patterns
3. Configuration files
4. Custom patterns
5. Manual configuration
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.transformer.modeling_transformer import TransformerConfig, TransformerForLM
from models.transformer.layer_config import LayerConfigBuilder, LayerPattern
from transformers import AutoTokenizer
import json
import yaml


def example_1_predefined_patterns():
    """Example 1: Using predefined patterns."""
    print("="*60)
    print("Example 1: Predefined Patterns")
    print("="*60)
    
    # Available patterns: uniform, alternating, sandwich, pyramid, sparse_special
    
    # Alternating pattern: SDPA and token_param every other layer
    config1 = TransformerConfig.from_pattern(
        pattern_name="alternating",
        n_layers=8,
        hidden_size=256,
        n_heads=4,
        vocab_size=1000,
        base_attention="self_sdpa",  # Override base attention
        base_mlp="swiglu"
    )
    
    print("Alternating Pattern:")
    print(config1.get_layer_info())
    
    # Sandwich pattern: special attention at first and last layers
    config2 = TransformerConfig.from_pattern(
        pattern_name="sandwich",
        n_layers=6,
        hidden_size=256,
        n_heads=4,
        vocab_size=1000
    )
    
    print("\nSandwich Pattern:")
    print(config2.get_layer_info())
    
    # Pyramid pattern: different sections use different attention
    config3 = TransformerConfig.from_pattern(
        pattern_name="pyramid",
        n_layers=12,
        hidden_size=256,
        n_heads=4,
        vocab_size=1000
    )
    
    print("\nPyramid Pattern:")
    print(config3.get_layer_info())


def example_2_string_patterns():
    """Example 2: Using string patterns."""
    print("\n" + "="*60)
    print("Example 2: String Patterns")
    print("="*60)
    
    # Simple string pattern
    config1 = TransformerConfig.from_string_pattern(
        pattern="SPECTRAL,SDPA*3,TOKEN_PARAM,SDPA*2,DIFF_ATTN",
        n_layers=8,
        hidden_size=256,
        n_heads=4,
        vocab_size=1000
    )
    
    print("String Pattern 1: 'SPECTRAL,SDPA*3,TOKEN_PARAM,SDPA*2,DIFF_ATTN'")
    print(config1.get_layer_info())
    
    # More complex pattern
    config2 = TransformerConfig.from_string_pattern(
        pattern="SPECTRAL*2,SDPA*4,TOKEN_PARAM*2,SDPA*4,SPECTRAL*2",
        n_layers=14,
        hidden_size=256,
        n_heads=4,
        vocab_size=1000
    )
    
    print("\nString Pattern 2: 'SPECTRAL*2,SDPA*4,TOKEN_PARAM*2,SDPA*4,SPECTRAL*2'")
    print(config2.get_layer_info())


def example_3_config_files():
    """Example 3: Using configuration files."""
    print("\n" + "="*60)
    print("Example 3: Configuration Files")
    print("="*60)
    
    # Create example JSON config
    json_config = {
        "pattern": {
            "name": "alternating",
            "n_layers": 10,
            "params": {
                "base_attention": "self_sdpa",
                "base_mlp": "swiglu"
            }
        }
    }
    
    with open("example_pattern_config.json", "w") as f:
        json.dump(json_config, f, indent=2)
    
    # Create example YAML config with explicit layers
    yaml_config = {
        "layers": [
            {"attention_type": "spectral", "mlp_type": "swiglu"},
            {"attention_type": "self_sdpa", "mlp_type": "swiglu"},
            {"attention_type": "self_sdpa", "mlp_type": "swiglu"},
            {"attention_type": "token_param", "mlp_type": "swiglu"},
            {"attention_type": "self_sdpa", "mlp_type": "swiglu"},
            {"attention_type": "diff_attn", "mlp_type": "swiglu"},
            {"attention_type": "self_sdpa", "mlp_type": "swiglu"},
            {"attention_type": "spectral", "mlp_type": "swiglu"},
        ]
    }
    
    with open("example_explicit_config.yaml", "w") as f:
        yaml.dump(yaml_config, f, default_flow_style=False)
    
    # Load from JSON
    config1 = TransformerConfig.from_config_file(
        config_file="example_pattern_config.json",
        hidden_size=256,
        n_heads=4,
        vocab_size=1000
    )
    
    print("JSON Config (Pattern-based):")
    print(config1.get_layer_info())
    
    # Load from YAML
    config2 = TransformerConfig.from_config_file(
        config_file="example_explicit_config.yaml",
        hidden_size=256,
        n_heads=4,
        vocab_size=1000
    )
    
    print("\nYAML Config (Explicit layers):")
    print(config2.get_layer_info())
    
    # Clean up
    os.remove("example_pattern_config.json")
    os.remove("example_explicit_config.yaml")


def example_4_custom_patterns():
    """Example 4: Creating custom patterns."""
    print("\n" + "="*60)
    print("Example 4: Custom Patterns")
    print("="*60)
    
    # Create a custom pattern: spectral attention every 3rd layer, token_param in the middle
    custom_pattern = LayerPattern(
        name="custom_mixed",
        description="Spectral every 3rd layer, token_param in middle section",
        base_attention="self_sdpa",
        base_mlp="swiglu",
        overrides=[
            {
                "condition": {"every": 3},
                "attention_type": "spectral"
            },
            {
                "condition": {"fraction": [0.4, 0.6]},
                "attention_type": "token_param"
            }
        ]
    )
    
    builder = LayerConfigBuilder()
    builder.add_custom_pattern(custom_pattern)
    
    config = TransformerConfig.from_pattern(
        pattern_name="custom_mixed",
        n_layers=12,
        hidden_size=256,
        n_heads=4,
        vocab_size=1000
    )
    
    print("Custom Pattern:")
    print(config.get_layer_info())
    
    # Another custom pattern: different attention types in different ranges
    range_pattern = LayerPattern(
        name="range_based",
        description="Different attention types in different ranges",
        base_attention="self_sdpa",
        base_mlp="swiglu",
        overrides=[
            {
                "condition": {"range": [0, 3]},
                "attention_type": "spectral"
            },
            {
                "condition": {"range": [3, 6]},
                "attention_type": "token_param"
            },
            {
                "condition": {"range": [6, 9]},
                "attention_type": "diff_attn"
            },
            {
                "condition": {"range": [9, 12]},
                "attention_type": "spectral"
            }
        ]
    )
    
    builder.add_custom_pattern(range_pattern)
    
    config2 = TransformerConfig.from_pattern(
        pattern_name="range_based",
        n_layers=12,
        hidden_size=256,
        n_heads=4,
        vocab_size=1000
    )
    
    print("\nRange-based Pattern:")
    print(config2.get_layer_info())


def example_5_manual_configuration():
    """Example 5: Manual layer-by-layer configuration."""
    print("\n" + "="*60)
    print("Example 5: Manual Configuration")
    print("="*60)
    
    # Manual configuration using the original approach (still supported)
    attention_types = [
        "spectral", "self_sdpa", "self_sdpa", "token_param",
        "self_sdpa", "diff_attn", "self_sdpa", "spectral"
    ]
    mlp_types = ["swiglu"] * 8
    
    config = TransformerConfig(
        hidden_size=256,
        n_heads=4,
        n_layers=8,
        attention_types=attention_types,
        mlp_types=mlp_types,
        vocab_size=1000
    )
    
    print("Manual Configuration:")
    print(config.get_layer_info())


def example_6_with_custom_parameters():
    """Example 6: Using custom parameters for specific layers."""
    print("\n" + "="*60)
    print("Example 6: Custom Parameters")
    print("="*60)
    
    # Create config with custom parameters for specific layers
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
            {"attention_type": "self_sdpa", "mlp_type": "swiglu"},
            {"attention_type": "self_sdpa", "mlp_type": "swiglu"},
            {
                "attention_type": "token_param", 
                "mlp_type": "swiglu",
                "custom_params": {
                    "attention": {"param_sharing": True}
                }
            },
            {"attention_type": "self_sdpa", "mlp_type": "swiglu"},
            {"attention_type": "spectral", "mlp_type": "swiglu"},
        ]
    }
    
    config = TransformerConfig(
        hidden_size=256,
        n_heads=4,
        vocab_size=1000,
        layer_config=config_dict
    )
    
    print("Configuration with Custom Parameters:")
    print(config.get_layer_info())
    
    # Show layer specs with custom params
    for i, spec in enumerate(config.layer_specs):
        if spec.custom_params:
            print(f"Layer {i} custom params: {spec.custom_params}")


def example_7_real_model_usage():
    """Example 7: Creating and using a real model."""
    print("\n" + "="*60)
    print("Example 7: Real Model Usage")
    print("="*60)
    
    try:
        # Create a tokenizer (you might need to install transformers)
        tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
        
        # Create config with interesting pattern
        config = TransformerConfig.from_pattern(
            pattern_name="sandwich",
            n_layers=6,
            hidden_size=128,
            n_heads=4,
            vocab_size=tokenizer.vocab_size,
            max_length=512
        )
        
        print("Model Configuration:")
        print(config.get_layer_info())
        
        # Create model
        model = TransformerForLM(config, tokenizer)
        
        print(f"\nModel created successfully!")
        print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test with dummy input
        import torch
        dummy_input = torch.randint(0, tokenizer.vocab_size, (2, 10))
        
        with torch.no_grad():
            outputs = model(dummy_input)
            print(f"Output shape: {outputs.logits.shape}")
            print("Model forward pass successful!")
            
    except Exception as e:
        print(f"Could not create real model: {e}")
        print("This might be due to missing dependencies or attention modules.")


def main():
    """Run all examples."""
    print("Flexible Transformer Layer Configuration Examples")
    print("=" * 60)
    
    example_1_predefined_patterns()
    example_2_string_patterns()
    example_3_config_files()
    example_4_custom_patterns()
    example_5_manual_configuration()
    example_6_with_custom_parameters()
    example_7_real_model_usage()
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)


if __name__ == "__main__":
    main() 