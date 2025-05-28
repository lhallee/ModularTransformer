import streamlit as st
import json
import yaml
from typing import List, Dict, Any
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from layer_config import LayerConfigBuilder, LayerPattern, LayerSpec, LayerType, MLPType


def main():
    st.set_page_config(
        page_title="Transformer Layer Configuration",
        page_icon="🔧",
        layout="wide"
    )
    
    st.title("🔧 Transformer Layer Configuration Tool")
    st.markdown("Design your custom transformer architecture with flexible layer patterns!")
    
    # Initialize session state
    if 'builder' not in st.session_state:
        st.session_state.builder = LayerConfigBuilder()
    if 'layers' not in st.session_state:
        st.session_state.layers = []
    
    # Sidebar for configuration method selection
    st.sidebar.title("Configuration Method")
    config_method = st.sidebar.selectbox(
        "Choose how to configure layers:",
        ["Pattern-based", "Manual Design", "String Pattern", "File Upload", "Custom Pattern"]
    )
    
    if config_method == "Pattern-based":
        pattern_based_config()
    elif config_method == "Manual Design":
        manual_design_config()
    elif config_method == "String Pattern":
        string_pattern_config()
    elif config_method == "File Upload":
        file_upload_config()
    elif config_method == "Custom Pattern":
        custom_pattern_config()
    
    # Display results
    if st.session_state.layers:
        display_results()


def pattern_based_config():
    st.header("Pattern-based Configuration")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Pattern Selection")
        
        # Pattern selection
        patterns = st.session_state.builder.list_patterns()
        selected_pattern = st.selectbox("Choose a pattern:", patterns)
        
        # Display pattern info
        pattern_info = st.session_state.builder.get_pattern_info(selected_pattern)
        st.info(pattern_info)
        
        # Number of layers
        n_layers = st.slider("Number of layers:", 1, 48, 12)
        
        # Base configuration overrides
        st.subheader("Base Configuration")
        base_attention = st.selectbox(
            "Base attention type:",
            [e.value for e in LayerType],
            index=0
        )
        base_mlp = st.selectbox(
            "Base MLP type:",
            [e.value for e in MLPType],
            index=0
        )
        
        # Generate layers
        if st.button("Generate Configuration"):
            try:
                st.session_state.layers = st.session_state.builder.from_pattern(
                    selected_pattern, 
                    n_layers,
                    base_attention=base_attention,
                    base_mlp=base_mlp
                )
                st.success(f"Generated {len(st.session_state.layers)} layers!")
            except Exception as e:
                st.error(f"Error generating configuration: {e}")
    
    with col2:
        st.subheader("Pattern Preview")
        if st.session_state.layers:
            create_layer_visualization(st.session_state.layers)


def manual_design_config():
    st.header("Manual Layer Design")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Layer Configuration")
        
        # Number of layers
        n_layers = st.slider("Number of layers:", 1, 48, 12)
        
        # Initialize layers if needed
        if len(st.session_state.get('manual_layers', [])) != n_layers:
            st.session_state.manual_layers = [
                {"attention": "self_sdpa", "mlp": "swiglu"} 
                for _ in range(n_layers)
            ]
        
        # Layer configuration
        st.subheader("Individual Layer Settings")
        
        # Bulk operations
        st.markdown("**Bulk Operations:**")
        bulk_col1, bulk_col2 = st.columns(2)
        
        with bulk_col1:
            bulk_attention = st.selectbox(
                "Set all attention to:",
                [e.value for e in LayerType],
                key="bulk_attention"
            )
            if st.button("Apply to All", key="bulk_attention_btn"):
                for layer in st.session_state.manual_layers:
                    layer["attention"] = bulk_attention
                st.rerun()
        
        with bulk_col2:
            bulk_mlp = st.selectbox(
                "Set all MLP to:",
                [e.value for e in MLPType],
                key="bulk_mlp"
            )
            if st.button("Apply to All", key="bulk_mlp_btn"):
                for layer in st.session_state.manual_layers:
                    layer["mlp"] = bulk_mlp
                st.rerun()
        
        # Individual layer configuration
        st.markdown("**Individual Layers:**")
        for i in range(min(10, n_layers)):  # Show first 10 layers in detail
            with st.expander(f"Layer {i}"):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.session_state.manual_layers[i]["attention"] = st.selectbox(
                        "Attention:",
                        [e.value for e in LayerType],
                        index=[e.value for e in LayerType].index(
                            st.session_state.manual_layers[i]["attention"]
                        ),
                        key=f"attention_{i}"
                    )
                with col_b:
                    st.session_state.manual_layers[i]["mlp"] = st.selectbox(
                        "MLP:",
                        [e.value for e in MLPType],
                        index=[e.value for e in MLPType].index(
                            st.session_state.manual_layers[i]["mlp"]
                        ),
                        key=f"mlp_{i}"
                    )
        
        if n_layers > 10:
            st.info(f"Showing first 10 layers. Layers 10-{n_layers-1} use the same configuration as layer 9.")
            # Apply layer 9 config to remaining layers
            for i in range(10, n_layers):
                st.session_state.manual_layers[i] = st.session_state.manual_layers[9].copy()
        
        # Generate configuration
        if st.button("Generate Configuration"):
            try:
                attention_types = [layer["attention"] for layer in st.session_state.manual_layers]
                mlp_types = [layer["mlp"] for layer in st.session_state.manual_layers]
                st.session_state.layers = st.session_state.builder.from_lists(
                    attention_types, mlp_types
                )
                st.success(f"Generated {len(st.session_state.layers)} layers!")
            except Exception as e:
                st.error(f"Error generating configuration: {e}")
    
    with col2:
        st.subheader("Configuration Preview")
        if st.session_state.get('manual_layers'):
            # Create preview layers for visualization
            preview_layers = [
                LayerSpec(
                    attention_type=layer["attention"],
                    mlp_type=layer["mlp"],
                    layer_idx=i
                )
                for i, layer in enumerate(st.session_state.manual_layers)
            ]
            create_layer_visualization(preview_layers)


def string_pattern_config():
    st.header("String Pattern Configuration")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Pattern String")
        
        # Help text
        st.markdown("""
        **Pattern Format:**
        - Use layer type names: `SDPA`, `SPECTRAL`, `TOKEN_PARAM`, `DIFF_ATTN`
        - Use repetition: `SDPA*3` for 3 SDPA layers
        - Separate with commas: `SPECTRAL,SDPA*2,TOKEN_PARAM`
        
        **Examples:**
        - `SDPA*12` - All SDPA layers
        - `SPECTRAL,SDPA*10,SPECTRAL` - Spectral sandwich
        - `SDPA*2,TOKEN_PARAM,SDPA*2,TOKEN_PARAM` - Alternating pattern
        """)
        
        # Pattern input
        pattern_string = st.text_area(
            "Enter pattern string:",
            value="SPECTRAL,SDPA*3,TOKEN_PARAM,SDPA*2,DIFF_ATTN,SDPA*4",
            height=100
        )
        
        # Number of layers
        n_layers = st.slider("Number of layers:", 1, 48, 12)
        
        # Generate configuration
        if st.button("Generate Configuration"):
            try:
                st.session_state.layers = st.session_state.builder.from_string_pattern(
                    pattern_string, n_layers
                )
                st.success(f"Generated {len(st.session_state.layers)} layers!")
            except Exception as e:
                st.error(f"Error generating configuration: {e}")
    
    with col2:
        st.subheader("Pattern Preview")
        if st.session_state.layers:
            create_layer_visualization(st.session_state.layers)


def file_upload_config():
    st.header("File Upload Configuration")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Upload Configuration File")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a configuration file",
            type=['json', 'yaml', 'yml']
        )
        
        if uploaded_file is not None:
            try:
                # Read file content
                content = uploaded_file.read().decode('utf-8')
                
                # Parse based on file type
                if uploaded_file.name.endswith('.json'):
                    config = json.loads(content)
                else:
                    config = yaml.safe_load(content)
                
                # Display configuration
                st.subheader("Configuration Preview")
                st.json(config)
                
                # Generate layers
                if st.button("Load Configuration"):
                    st.session_state.layers = st.session_state.builder._parse_config_dict(config)
                    st.success(f"Loaded {len(st.session_state.layers)} layers!")
                    
            except Exception as e:
                st.error(f"Error parsing file: {e}")
        
        # Example configurations
        st.subheader("Example Configurations")
        
        if st.button("Download JSON Example"):
            example_json = {
                "pattern": {
                    "name": "alternating",
                    "n_layers": 12,
                    "params": {
                        "base_attention": "self_sdpa",
                        "base_mlp": "swiglu"
                    }
                }
            }
            st.download_button(
                "Download",
                json.dumps(example_json, indent=2),
                "example_config.json",
                "application/json"
            )
        
        if st.button("Download YAML Example"):
            example_yaml = {
                "layers": [
                    {"attention_type": "spectral", "mlp_type": "swiglu"},
                    {"attention_type": "self_sdpa", "mlp_type": "swiglu"},
                    {"attention_type": "token_param", "mlp_type": "swiglu"},
                ]
            }
            st.download_button(
                "Download",
                yaml.dump(example_yaml, default_flow_style=False),
                "example_config.yaml",
                "application/x-yaml"
            )
    
    with col2:
        st.subheader("Configuration Preview")
        if st.session_state.layers:
            create_layer_visualization(st.session_state.layers)


def custom_pattern_config():
    st.header("Custom Pattern Creation")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Pattern Definition")
        
        # Pattern metadata
        pattern_name = st.text_input("Pattern name:", "my_custom_pattern")
        pattern_description = st.text_area("Description:", "My custom layer pattern")
        
        # Base configuration
        base_attention = st.selectbox(
            "Base attention type:",
            [e.value for e in LayerType],
            index=0
        )
        base_mlp = st.selectbox(
            "Base MLP type:",
            [e.value for e in MLPType],
            index=0
        )
        
        # Override rules
        st.subheader("Override Rules")
        
        if 'custom_overrides' not in st.session_state:
            st.session_state.custom_overrides = []
        
        # Add new override
        with st.expander("Add Override Rule"):
            condition_type = st.selectbox(
                "Condition type:",
                ["every", "position", "range", "fraction", "layer_idx"]
            )
            
            condition = {}
            if condition_type == "every":
                every = st.number_input("Every N layers:", min_value=1, value=2)
                offset = st.number_input("Offset:", min_value=0, value=0)
                condition = {"every": every, "offset": offset}
            elif condition_type == "position":
                position = st.selectbox("Position:", ["first", "last", "middle"])
                condition = {"position": position}
            elif condition_type == "range":
                start = st.number_input("Start layer:", min_value=0, value=0)
                end = st.number_input("End layer:", min_value=1, value=5)
                condition = {"range": [start, end]}
            elif condition_type == "fraction":
                start_frac = st.slider("Start fraction:", 0.0, 1.0, 0.0)
                end_frac = st.slider("End fraction:", 0.0, 1.0, 0.3)
                condition = {"fraction": [start_frac, end_frac]}
            elif condition_type == "layer_idx":
                indices = st.text_input("Layer indices (comma-separated):", "0,5,10")
                try:
                    idx_list = [int(x.strip()) for x in indices.split(',')]
                    condition = {"layer_idx": idx_list}
                except:
                    condition = {"layer_idx": [0]}
            
            override_attention = st.selectbox(
                "Override attention type:",
                [e.value for e in LayerType],
                index=1
            )
            override_mlp = st.selectbox(
                "Override MLP type:",
                [e.value for e in MLPType],
                index=0
            )
            
            if st.button("Add Override"):
                override = {
                    "condition": condition,
                    "attention_type": override_attention,
                    "mlp_type": override_mlp
                }
                st.session_state.custom_overrides.append(override)
                st.rerun()
        
        # Display current overrides
        if st.session_state.custom_overrides:
            st.subheader("Current Overrides")
            for i, override in enumerate(st.session_state.custom_overrides):
                with st.expander(f"Override {i+1}"):
                    st.json(override)
                    if st.button(f"Remove Override {i+1}", key=f"remove_{i}"):
                        st.session_state.custom_overrides.pop(i)
                        st.rerun()
        
        # Test pattern
        n_layers = st.slider("Test with N layers:", 1, 24, 12)
        
        if st.button("Test Pattern"):
            try:
                # Create pattern
                pattern = LayerPattern(
                    name=pattern_name,
                    description=pattern_description,
                    base_attention=base_attention,
                    base_mlp=base_mlp,
                    overrides=st.session_state.custom_overrides
                )
                
                # Generate layers
                st.session_state.layers = pattern.generate_layers(n_layers)
                st.success(f"Generated {len(st.session_state.layers)} layers!")
                
                # Add to builder
                st.session_state.builder.add_custom_pattern(pattern)
                
            except Exception as e:
                st.error(f"Error creating pattern: {e}")
    
    with col2:
        st.subheader("Pattern Preview")
        if st.session_state.layers:
            create_layer_visualization(st.session_state.layers)


def create_layer_visualization(layers: List[LayerSpec]):
    """Create a visualization of the layer configuration."""
    
    # Prepare data for visualization
    layer_data = []
    for layer in layers:
        layer_data.append({
            'Layer': layer.layer_idx,
            'Attention': layer.attention_type,
            'MLP': layer.mlp_type
        })
    
    df = pd.DataFrame(layer_data)
    
    # Color mapping for attention types
    attention_colors = {
        'self_sdpa': '#1f77b4',
        'diff_attn': '#ff7f0e',
        'token_param': '#2ca02c',
        'spectral': '#d62728',
        'self_flex': '#9467bd'
    }
    
    # Create visualization
    fig = go.Figure()
    
    # Add attention type bars
    for att_type in df['Attention'].unique():
        mask = df['Attention'] == att_type
        fig.add_trace(go.Bar(
            x=df[mask]['Layer'],
            y=[1] * sum(mask),
            name=att_type,
            marker_color=attention_colors.get(att_type, '#gray'),
            text=df[mask]['Attention'],
            textposition='inside',
            hovertemplate=f'<b>Layer %{{x}}</b><br>Attention: {att_type}<br>MLP: %{{customdata}}<extra></extra>',
            customdata=df[mask]['MLP']
        ))
    
    fig.update_layout(
        title="Layer Configuration Visualization",
        xaxis_title="Layer Index",
        yaxis_title="",
        barmode='stack',
        height=400,
        showlegend=True,
        yaxis=dict(showticklabels=False)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Layers", len(layers))
    
    with col2:
        attention_counts = df['Attention'].value_counts()
        most_common = attention_counts.index[0]
        st.metric("Most Common Attention", most_common, f"{attention_counts[most_common]} layers")
    
    with col3:
        unique_attention = len(df['Attention'].unique())
        st.metric("Attention Types Used", unique_attention)


def display_results():
    """Display the final configuration results."""
    
    st.header("Configuration Results")
    
    # Layer summary table
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Layer Configuration Table")
        
        # Create DataFrame for display
        display_data = []
        for layer in st.session_state.layers:
            display_data.append({
                'Layer': layer.layer_idx,
                'Attention Type': layer.attention_type,
                'MLP Type': layer.mlp_type,
                'Custom Params': str(layer.custom_params) if layer.custom_params else 'None'
            })
        
        df = pd.DataFrame(display_data)
        st.dataframe(df, use_container_width=True)
    
    with col2:
        st.subheader("Export Configuration")
        
        # Generate configuration for export
        config_dict = {
            "layers": [
                {
                    "attention_type": layer.attention_type,
                    "mlp_type": layer.mlp_type,
                    "custom_params": layer.custom_params
                }
                for layer in st.session_state.layers
            ]
        }
        
        # JSON export
        json_str = json.dumps(config_dict, indent=2)
        st.download_button(
            "Download JSON",
            json_str,
            "layer_config.json",
            "application/json"
        )
        
        # YAML export
        yaml_str = yaml.dump(config_dict, default_flow_style=False)
        st.download_button(
            "Download YAML",
            yaml_str,
            "layer_config.yaml",
            "application/x-yaml"
        )
        
        # Python code export
        python_code = generate_python_code(st.session_state.layers)
        st.download_button(
            "Download Python Code",
            python_code,
            "layer_config.py",
            "text/plain"
        )


def generate_python_code(layers: List[LayerSpec]) -> str:
    """Generate Python code for the configuration."""
    
    attention_types = [layer.attention_type for layer in layers]
    mlp_types = [layer.mlp_type for layer in layers]
    
    code = f"""# Generated layer configuration
from models.transformer.modeling_transformer import TransformerConfig

# Layer configuration
attention_types = {attention_types}
mlp_types = {mlp_types}

# Create config
config = TransformerConfig(
    hidden_size=512,
    n_heads=8,
    n_layers={len(layers)},
    expansion_ratio=8/3,
    dropout=0.1,
    rotary=True,
    causal=False,
    attention_types=attention_types,
    mlp_types=mlp_types,
    vocab_size=32000,
    max_length=2048,
    lm_type="mlm",
    tie_embeddings=True
)
"""
    
    return code


if __name__ == "__main__":
    main() 