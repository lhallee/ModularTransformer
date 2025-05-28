import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Any, List, Union
from transformers import PreTrainedModel, PretrainedConfig, PreTrainedTokenizer
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass

from ..attention.self_attention import MultiHeadAttention
from ..attention.diff_attention import DiffAttention
from ..attention.token_param_attention import TokenParamAttention
from ..attention.spectral_attention import MultiHeadSpectralAttention
from ..feedforward.MLP import swiglu_ffn, relu2_ffn
from ..generate_mixin import GenerateMixin

# Import the new layer configuration system
try:
    from .layer_config import LayerConfigBuilder, LayerSpec
    LAYER_CONFIG_AVAILABLE = True
except ImportError:
    LAYER_CONFIG_AVAILABLE = False


class TransformerConfig(PretrainedConfig):
    model_type = "transformer"
    def __init__(
        self,
        ### Used in base transformer
        hidden_size: int = 512,
        n_heads: int =  8,
        n_layers: int = 12,
        expansion_ratio: float = 8 / 3,
        dropout: float = 0.1,
        rotary: bool = True,
        causal: bool = False,
        # Options: ["self_sdpa", "self_flex", "diff_attn", "token_param", "spectral"]
        attention_types: list[str] = ["self_sdpa"],
        # Options: ["swiglu", "relu2"]
        mlp_types: list[str] = ["swiglu"],
        ### For LM
        vocab_size: int = 32000,
        max_length: int = 2048,
        # Options: ["mlm", "ar", "diffusion"]
        lm_type: str = "mlm",
        tie_embeddings: bool = True,
        ### New flexible layer configuration options
        layer_config: Optional[Union[str, dict, List[LayerSpec]]] = None,
        layer_pattern: Optional[str] = None,
        layer_config_file: Optional[str] = None,
        **kwargs: Any
    ):
        super().__init__(**kwargs)
        
        # Store basic configuration
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.expansion_ratio = expansion_ratio
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.dropout = dropout
        self.rotary = rotary
        self.causal = causal
        self.lm_type = lm_type
        self.tie_embeddings = tie_embeddings
        
        # Handle layer configuration
        self.layer_specs = self._resolve_layer_configuration(
            attention_types, mlp_types, layer_config, layer_pattern, layer_config_file
        )
        
        # Extract attention_types and mlp_types for backward compatibility
        self.attention_types = [spec.attention_type for spec in self.layer_specs]
        self.mlp_types = [spec.mlp_type for spec in self.layer_specs]
        
        # Validate configuration
        assert len(self.attention_types) == len(self.mlp_types), 'attention_types and mlp_types must be the same length'
        
        # Update n_layers to match actual layer count
        if len(self.layer_specs) != n_layers:
            self.n_layers = len(self.layer_specs)
    
    def _resolve_layer_configuration(
        self, 
        attention_types: List[str], 
        mlp_types: List[str],
        layer_config: Optional[Union[str, dict, List[LayerSpec]]],
        layer_pattern: Optional[str],
        layer_config_file: Optional[str]
    ) -> List[LayerSpec]:
        """Resolve layer configuration from various input formats."""
        
        if not LAYER_CONFIG_AVAILABLE:
            # Fallback to original behavior if layer_config module not available
            if len(attention_types) != len(mlp_types):
                raise ValueError("attention_types and mlp_types must have the same length")
            return [
                LayerSpec(attention_type=att, mlp_type=mlp, layer_idx=i, custom_params={})
                for i, (att, mlp) in enumerate(zip(attention_types, mlp_types))
            ]
        
        builder = LayerConfigBuilder()
        
        # Priority order: layer_config > layer_config_file > layer_pattern > original lists
        
        if layer_config is not None:
            if isinstance(layer_config, list):
                # Direct list of LayerSpec objects
                return layer_config
            elif isinstance(layer_config, str):
                # String pattern
                return builder.from_string_pattern(layer_config, self.n_layers)
            elif isinstance(layer_config, dict):
                # Dictionary configuration
                return builder._parse_config_dict(layer_config)
        
        elif layer_config_file is not None:
            # Load from file
            if layer_config_file.endswith('.json'):
                return builder.from_json(layer_config_file)
            elif layer_config_file.endswith(('.yaml', '.yml')):
                return builder.from_yaml(layer_config_file)
            else:
                raise ValueError(f"Unsupported file format: {layer_config_file}")
        
        elif layer_pattern is not None:
            # Use predefined pattern
            return builder.from_pattern(layer_pattern, self.n_layers)
        
        else:
            # Use original attention_types and mlp_types
            return builder.from_lists(attention_types, mlp_types)
    
    @classmethod
    def from_pattern(
        cls,
        pattern_name: str,
        n_layers: int,
        hidden_size: int = 512,
        n_heads: int = 8,
        **kwargs
    ):
        """Create configuration from a predefined pattern."""
        return cls(
            hidden_size=hidden_size,
            n_heads=n_heads,
            n_layers=n_layers,
            layer_pattern=pattern_name,
            **kwargs
        )
    
    @classmethod
    def from_string_pattern(
        cls,
        pattern: str,
        n_layers: int,
        hidden_size: int = 512,
        n_heads: int = 8,
        **kwargs
    ):
        """Create configuration from a string pattern."""
        return cls(
            hidden_size=hidden_size,
            n_heads=n_heads,
            n_layers=n_layers,
            layer_config=pattern,
            **kwargs
        )
    
    @classmethod
    def from_config_file(
        cls,
        config_file: str,
        hidden_size: int = 512,
        n_heads: int = 8,
        **kwargs
    ):
        """Create configuration from a config file."""
        return cls(
            hidden_size=hidden_size,
            n_heads=n_heads,
            layer_config_file=config_file,
            **kwargs
        )
    
    def get_layer_info(self) -> str:
        """Get a summary of the layer configuration."""
        info = f"Transformer with {self.n_layers} layers:\n"
        
        # Count layer types
        attention_counts = {}
        mlp_counts = {}
        
        for spec in self.layer_specs:
            attention_counts[spec.attention_type] = attention_counts.get(spec.attention_type, 0) + 1
            mlp_counts[spec.mlp_type] = mlp_counts.get(spec.mlp_type, 0) + 1
        
        info += "Attention types:\n"
        for att_type, count in attention_counts.items():
            info += f"  - {att_type}: {count} layers\n"
        
        info += "MLP types:\n"
        for mlp_type, count in mlp_counts.items():
            info += f"  - {mlp_type}: {count} layers\n"
        
        return info


# Create a simple LayerSpec class for when layer_config module is not available
if not LAYER_CONFIG_AVAILABLE:
    @dataclass
    class LayerSpec:
        attention_type: str
        mlp_type: str
        layer_idx: int
        custom_params: dict = None
        
        def __post_init__(self):
            if self.custom_params is None:
                self.custom_params = {}


@dataclass
class TransformerOutput(ModelOutput):
    """Output type for ESM++ models."""
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    last_hidden_state: Optional[torch.Tensor] = None


class TiedLinear(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight
    
    def forward(self, x):
        return F.linear(x, self.weight)


class BasicTransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, n_heads: int, expansion_ratio: float, dropout: float, rotary: bool, causal: bool):
        super().__init__()
        self.attn = MultiHeadAttention(hidden_size, n_heads, rotary, causal)
        self.ffn = swiglu_ffn(hidden_size, expansion_ratio, dropout)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.attn(x, attention_mask) + x
        x = self.ffn(x) + x
        return x


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig, layer_spec: LayerSpec):
        super().__init__()
        self.layer_idx = layer_spec.layer_idx
        self.attn_layers = nn.ModuleList([])
        self.ffn_layers = nn.ModuleList([])
        
        # Get custom parameters for this layer
        custom_params = layer_spec.custom_params or {}
        
        # Create attention layer
        attn_type = layer_spec.attention_type
        if attn_type == "self_sdpa":
            self.attn_layers.append(MultiHeadAttention(
                hidden_size=config.hidden_size,
                n_heads=config.n_heads,
                rotary=config.rotary,
                causal=config.causal,
                **custom_params.get('attention', {})
            ))
        elif attn_type == "self_flex":
            raise NotImplementedError("Self Flex Attention is not implemented")
        elif attn_type == "diff_attn":
            self.attn_layers.append(DiffAttention(
                hidden_size=config.hidden_size,
                n_heads=config.n_heads,
                rotary=config.rotary,
                causal=config.causal,
                layer_idx=self.layer_idx,
                **custom_params.get('attention', {})
            ))
        elif attn_type == "token_param":
            self.attn_layers.append(TokenParamAttention(
                hidden_size=config.hidden_size,
                n_heads=config.n_heads,
                rotary=config.rotary,
                causal=config.causal,
                **custom_params.get('attention', {})
            ))
        elif attn_type == "spectral":
            self.attn_layers.append(MultiHeadSpectralAttention(
                hidden_size=config.hidden_size,
                n_heads=config.n_heads,
                max_length=config.max_length,
                **custom_params.get('attention', {})
            ))
        else:
            raise ValueError(f"Invalid attention type: {attn_type}")
        
        # Create MLP layer
        mlp_type = layer_spec.mlp_type
        if mlp_type == "swiglu":
            self.ffn_layers.append(swiglu_ffn(
                hidden_size=config.hidden_size,
                expansion_ratio=config.expansion_ratio,
                dropout=config.dropout,
                **custom_params.get('mlp', {})
            ))
        elif mlp_type == "relu2":
            self.ffn_layers.append(relu2_ffn(
                hidden_size=config.hidden_size,
                expansion_ratio=config.expansion_ratio,
                dropout=config.dropout,
                **custom_params.get('mlp', {})
            ))
        else:
            raise ValueError(f"Invalid mlp type: {mlp_type}")

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for attn, ffn in zip(self.attn_layers, self.ffn_layers):
            x = attn(x, attention_mask) + x
            x = ffn(x) + x
        return x
    

class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(config, layer_spec) 
            for layer_spec in config.layer_specs
        ])

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        if attention_mask is not None and attention_mask.dim() == 2:
            attention_mask = attention_mask[:, None, None, :].expand(batch_size, 1, seq_len, seq_len).bool()
        for layer in self.layers:
            x = layer(x, attention_mask)
        return x


class TransformerForLM(PreTrainedModel, GenerateMixin):
    config_class = TransformerConfig
    
    def __init__(self, config: TransformerConfig, tokenizer: PreTrainedTokenizer):
        PreTrainedModel.__init__(self, config)
        GenerateMixin.__init__(self, tokenizer)
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.transformer = Transformer(config)
        
        self.lm_head = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            (nn.Linear(config.hidden_size, config.vocab_size) 
             if not config.tie_embeddings 
             else TiedLinear(self.embeddings.weight))
        )

        self.ce_loss = nn.CrossEntropyLoss()
        self.vocab_size = config.vocab_size
        self.lm_type = config.lm_type
        self.tokenizer = tokenizer
        self.special_tokens = {
            "pad_token_id": self.tokenizer.pad_token_id,
            "cls_token_id": self.tokenizer.cls_token_id,
            "sep_token_id": self.tokenizer.sep_token_id,
            "mask_token_id": self.tokenizer.mask_token_id,
        }
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                module.bias.data.zero_()
            if module.weight is not None:
                module.weight.data.fill_(1.0)
            
    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.LongTensor:
        pass

    def ar_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        original_tokens: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> torch.Tensor:
        x = self.embeddings(input_ids)
        x = self.transformer(x, attention_mask)
        logits = self.lm_head(x)
        loss = None
        
        if original_tokens is not None:
            # set special tokens to -100 for original_tokens
            for _, idx in self.special_tokens.items():
                original_tokens[original_tokens == idx] = -100
            # shift logits and labels by one
            logits = logits[:, :-1, :]
            original_tokens = original_tokens[:, 1:]
            loss = self.ce_loss(logits.reshape(-1, self.vocab_size), original_tokens.reshape(-1))
        return logits, loss, x

    def mlm_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> torch.Tensor:
        x = self.embeddings(input_ids)
        x = self.transformer(x, attention_mask)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = self.ce_loss(logits.view(-1, self.vocab_size), labels.view(-1))
        return logits, loss, x

    def diffusion_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> torch.Tensor:
        pass

    def get_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> torch.Tensor:
        if self.lm_type == "ar":
            return self.ar_forward(input_ids, attention_mask, **kwargs)[0]
        elif self.lm_type == "mlm":
            return self.mlm_forward(input_ids, attention_mask, **kwargs)[0]
        elif self.lm_type == "diffusion":
            return self.diffusion_forward(input_ids, attention_mask, **kwargs)[0]
        else:
            raise ValueError(f"Invalid LM type: {self.lm_type}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None, # original tokens with -100 for unmasked tokens
        original_tokens: Optional[torch.Tensor] = None, # original tokens
        mask_ratio: Optional[float] = None, # mask ratio
        return_preds: bool = True,
    ) -> TransformerOutput:

        if self.lm_type == "ar":
            logits, loss, last_hidden_state = self.ar_forward(input_ids, attention_mask, original_tokens)
        elif self.lm_type == "mlm":
            logits, loss, last_hidden_state = self.mlm_forward(input_ids, attention_mask, labels)
        elif self.lm_type == "diffusion":
            logits, loss, last_hidden_state = self.diffusion_forward(input_ids, attention_mask, mask_ratio)
        else:
            raise ValueError(f"Invalid LM type: {self.lm_type}")
        
        return TransformerOutput(
            loss=loss,
            logits=logits.argmax(dim=-1) if return_preds else logits,
            last_hidden_state=last_hidden_state,
        )


if __name__ == "__main__":
    # py -m models.transformer.modeling_transformer
    import torch
    from transformers import AutoTokenizer
    from functools import partial
    
    # Helper function to print test results
    def print_test_header(test_name):
        print("\n" + "="*80)
        print(f" {test_name} ".center(80, "="))
        print("="*80)
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    preview_decode = lambda ids: tokenizer.decode(ids, skip_special_tokens=False).replace(' ', '')
    # Sample text
    texts = [
        "MEVEAG",
        "MEEEWAGVVPLE"
    ]
    
    # Tokenize inputs
    encoded = tokenizer(texts, padding=True, return_tensors="pt")
    # preview the input_ids
    for i, text in enumerate(texts):
        print(f"Text {i+1}: {text}")
        print(f"Input IDs: {encoded['input_ids'][i]}")
        print(f"Attention Mask: {encoded['attention_mask'][i]}")
        print(f"Decoded: {preview_decode(encoded['input_ids'][i])}")
        print("\n")
    
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    
    # Test configurations for each model type
    configs = {
        "ar": TransformerConfig(
            hidden_size=128,
            n_heads=4,
            n_layers=2,
            expansion_ratio=2.0,
            vocab_size=tokenizer.vocab_size,
            max_length=512,
            dropout=0.1,
            rotary=True,
            causal=True,
            lm_type="ar",
            attention_types=["self_sdpa"],
            mlp_types=["swiglu"],
            tie_embeddings=True
        ),
        "mlm": TransformerConfig(
            hidden_size=128,
            n_heads=4,
            n_layers=2,
            expansion_ratio=2.0,
            vocab_size=tokenizer.vocab_size,
            max_length=512,
            dropout=0.1,
            rotary=True,
            causal=False,
            lm_type="mlm",
            attention_types=["self_sdpa"],
            mlp_types=["swiglu"],
            tie_embeddings=True
        ),
        "diffusion": TransformerConfig(
            hidden_size=128,
            n_heads=4,
            n_layers=2,
            expansion_ratio=2.0,
            vocab_size=tokenizer.vocab_size,
            max_length=512,
            dropout=0.1,
            rotary=True,
            causal=False,
            lm_type="diffusion",
            attention_types=["self_sdpa"],
            mlp_types=["swiglu"],
            tie_embeddings=True
        )
    }
    
    # Test each model type
    for model_type, config in configs.items():
        print_test_header(f"Testing {model_type.upper()} Model")
        
        # Initialize model
        model = TransformerForLM(config, tokenizer).to(device)
        print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Test forward pass
        print("\n1. Testing forward pass...")
        
        if model_type == "ar":
            # For AR, we need original tokens for loss calculation
            original_tokens = input_ids.clone()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                original_tokens=original_tokens
            )
            
        elif model_type == "mlm":
            # For MLM, create masked input and labels
            masked_input = input_ids.clone()
            labels = input_ids.clone()
            
            # Create random mask (15% of tokens)
            mask_prob = torch.full(input_ids.shape, 0.15)
            masked_indices = torch.bernoulli(mask_prob).bool().to(device)
            
            # Don't mask special tokens
            for special_token_id in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]:
                if special_token_id is not None:
                    masked_indices = masked_indices & (input_ids != special_token_id)
            
            # Set labels to -100 for unmasked tokens (will be ignored in loss)
            labels[~masked_indices] = -100
            
            # Replace masked tokens with [MASK]
            masked_input[masked_indices] = tokenizer.mask_token_id
            
            outputs = model(
                input_ids=masked_input,
                attention_mask=attention_mask,
                labels=labels
            )
            
        elif model_type == "diffusion":
            # For diffusion, use mask_ratio
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                mask_ratio=0.3
            )
        
        print(f"Forward pass successful!")
        print(f"Loss: {outputs.loss.item() if outputs.loss is not None else 'N/A'}")
        print(f"Logits shape: {outputs.logits.shape}")
        print(f"Last hidden state shape: {outputs.last_hidden_state.shape}")
        
        # Test generation
        print("\n2. Testing generation...")
        
        if model_type == "ar":
            # Test autoregressive generation
            generated = model.generate(
                input_ids=input_ids[:1, :3],  # Use shorter prompt for generation
                attention_mask=attention_mask[:1, :3],
                max_length=20,
                temperature=0.8,
                top_k=3,
                top_p=0.95,
                do_sample=True,
                preview=True
            )
            
            print(f"Generated sequence shape: {generated.shape}")
            print("Generated text:")
            print(preview_decode(generated[0]))
            
        elif model_type == "mlm":
            # Create input with masks for MLM generation
            masked_input = input_ids.clone()
            
            # Create random mask (30% of tokens)
            mask_prob = torch.full(input_ids.shape, 0.3)
            masked_indices = torch.bernoulli(mask_prob).bool().to(device)
            
            # Don't mask special tokens
            for special_token_id in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]:
                if special_token_id is not None:
                    masked_indices = masked_indices & (input_ids != special_token_id)
            
            # Replace masked tokens with [MASK]
            masked_input[masked_indices] = tokenizer.mask_token_id
            
            generated = model.generate(
                input_ids=masked_input[:1],
                attention_mask=attention_mask[:1],
                masked_indices=masked_indices[:1],
                num_iterations=3,
                confidence_strategy="min",
                preview=True
            )
            
            print(f"Generated sequence shape: {generated.shape}")
            print("Original text:")
            print(preview_decode(input_ids[0]))
            print("Masked input:")
            print(preview_decode(masked_input[0]))
            print("Generated text:")
            print(preview_decode(generated[0]))
            
        elif model_type == "diffusion":
            # Test diffusion generation
            prompt_ids = input_ids[:1, :3]  # Use shorter prompt
            prompt_mask = attention_mask[:1, :3]
            
            generated = model.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                gen_length=32,
                num_steps=32,
                timestep_spacing="uniform",
                confidence_strategy="min",
                preview=True
            )
            
            print(f"Generated sequence shape: {generated.shape}")
            print("Prompt:")
            print(preview_decode(prompt_ids[0]))
            print("Generated text:")
            print(preview_decode(generated[0]))
        
        print("\nTest completed successfully!")
