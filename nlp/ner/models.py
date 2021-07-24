from torch import nn
import transformers


class EntityModel(nn.Module):
    def __init__(self, model_dir, num_poses, num_tags):
        super().__init__()
        self.num_poses = num_poses
        self.num_tags = num_tags
        self.model = transformers.BertModel.from_pretrained(model_dir)
        emb_dim = self.model.embeddings.word_embeddings.embedding_dim
        self.dropout = nn.Dropout(0.3)
        self.out_pos = nn.Linear(emb_dim, self.num_poses)
        self.out_tag = nn.Linear(emb_dim, self.num_tags)

    def forward(self, ids, mask, token_type_ids):
        outputs = self.model(ids, attention_mask=mask, token_type_ids=token_type_ids)
        embs = outputs.last_hidden_state

        logits_pos = self.out_pos(self.dropout(embs))
        logits_tag = self.out_tag(self.dropout(embs))
        return logits_pos, logits_tag
