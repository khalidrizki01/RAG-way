from transformers import Qwen3Config, Qwen3ForCausalLM
import torch.nn as nn
import torch
import re

class XQwen3Config(Qwen3Config):
    def __init__(
        self,
        projector_type = 'mlp2x_gelu',
        retriever_hidden_size = 384,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.projector_type = projector_type
        self.retriever_hidden_size = retriever_hidden_size

class Projector(nn.Module):
    def __init__(self,config):
        super().__init__()
        projector_type = config.projector_type
        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
        if mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.group(1))
            modules = [nn.Linear(config.retriever_hidden_size, config.hidden_size)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(config.hidden_size, config.hidden_size))
            self.projector = nn.Sequential(*modules)
    
    def forward(self,context_embedding):
        return self.projector(context_embedding)
    
class XQwen3ForCausalLM(Qwen3ForCausalLM):
    def __init__(self,config):
        super().__init__(config)
        if hasattr(config,"retriever_hidden_size") and config.retriever_hidden_size > 0: 
            self.projector = Projector(config)
            self.retriever_hidden_size = config.retriever_hidden_size
        self.post_init()

    def set_xrag_token_id(self,token_id):
        self.xrag_token_id = token_id
    
    def prepare_inputs_embeds(self, input_ids, retrieval_embeds):
        inputs_embeds = self.model.embed_tokens(input_ids)
        retrieval_embeds = retrieval_embeds.view(-1, self.retriever_hidden_size)

        # # Temukan posisi semua xrag token
        xrag_positions = torch.nonzero(input_ids == self.xrag_token_id).squeeze(-1)
        num_xrag_tokens = len(xrag_positions)
        
        num_retrieval_embeds = retrieval_embeds.shape[0] 
        assert num_xrag_tokens == num_retrieval_embeds, f"num_xrag_tokens = {num_xrag_tokens}, num_retrieval_embeds = {num_retrieval_embeds}"
        retrieval_projected = self.projector(retrieval_embeds.to(inputs_embeds.dtype))  # diragukan apakah dia mengolah masing-masing retrieval_embeds secara individual atau secara berjamaah(?)

        # # Catat nilai xrag token sebelum disematkan
        xrag_before = inputs_embeds[xrag_positions[:, 0], xrag_positions[:, 1]]  #.to(torch.float32).clone().detach().cpu().numpy()  # Sebelum disematkan

        # Periksa apakah nilai antar xRAG token sebelum disematkan sama
        for i in range(1, len(xrag_before)):
            if (xrag_before[i] != xrag_before[i-1]).all():
                print(f"WARNING: Token xRAG sebelum disematkan memiliki nilai yang berbeda antara posisi {i-1} dan {i}")

        for i, position in enumerate(xrag_positions):
            inputs_embeds[position[0], position[1]] = retrieval_projected[i]

        # Pastikan nilai xrag token setelah disematkan berbeda antar posisi
            if i > 0:
                prev_position = xrag_positions[i-1]
                prev_value = inputs_embeds[prev_position[0], prev_position[1]]
                curr_value = inputs_embeds[position[0], position[1]]
                
                if (prev_value == curr_value).all():
                    print(f"WARNING: Token xRAG pada posisi {i-1} dan {i} memiliki nilai yang sama setelah disematkan.")

        return inputs_embeds
    
    def forward(
        self,
        input_ids=None,
        retrieval_embeds=None,
        attention_mask=None,
        **kwargs,
    ):

        inputs_embeds = kwargs.pop("inputs_embeds", None)
        at_the_beginning_of_generation = False

        if inputs_embeds is not None:
            assert not self.training
            assert retrieval_embeds is None
            at_the_beginning_of_generation = True

        if not at_the_beginning_of_generation:
            if retrieval_embeds is not None:
                inputs_embeds = self.prepare_inputs_embeds(input_ids, retrieval_embeds)
                input_ids = None
                if attention_mask is not None:
                    assert inputs_embeds.shape[1] == attention_mask.shape[1], f"inputs embeds dan attention mask tidak memiliki dimensi yang sama: {(inputs_embeds.shape, attention_mask.shape)}"

        return super().forward(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs,
        )
    
    @torch.no_grad()
    def generate(
        self,
        input_ids=None,
        retrieval_embeds=None,
        **kwargs,
    ):

        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported for generate")
        
        inputs_embeds = None
        if retrieval_embeds is not None:
            inputs_embeds = self.prepare_inputs_embeds(input_ids, retrieval_embeds)
            input_ids = None
            if attention_mask is not None:
                assert inputs_embeds.shape[1] == attention_mask.shape[1], (inputs_embeds.shape, attention_mask.shape)
            return super().generate(
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                **kwargs
            )
        
        else:
            return super().generate(
                attention_mask=attention_mask,
                input_ids=input_ids,
                **kwargs
            )