from transformers import AutoModel
from torch import Tensor
import torch.nn.functional as F


class E5Retriever:
    def __init__(self, model_name: str):
        """
        Inisialisasi model E5 sebagai retriever.
        Menggunakan AutoModel dan AutoTokenizer untuk pemrosesan teks.
        """
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name)  #.to("cuda:0")
        self.embed_dim = self.model.config.hidden_size  

    def get_embed_dim(self):
        """Mengembalikan dimensi embedding dari model."""
        return self.embed_dim
    
    def get_embed_length(self):
        """Panjang embedding tetap 1 karena representasi diringkas."""
        return 1
    
    def get_embedding(self, input_ids, attention_mask):
        """
        Menghasilkan embedding dengan average pooling.
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
            """Melakukan average pooling pada representasi token, mengabaikan padding."""
            last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
            return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

        embeddings = average_pool(outputs.last_hidden_state, attention_mask)
        return F.normalize(embeddings, p=2, dim=1)  # Normalisasi L2
    
    def get_doc_embedding(self, input_ids, attention_mask):
        """Menghasilkan embedding untuk dokumen."""
        return self.get_embedding(input_ids, attention_mask)
    
    def get_query_embedding(self, input_ids, attention_mask):
        """Menghasilkan embedding untuk query."""
        return self.get_embedding(input_ids, attention_mask)
    
    def to(self, device):
        """Memindahkan model ke device tertentu."""
        self.model = self.model.to(device) 
        return self
    
    def eval (self):
        self.model.eval()
        return self