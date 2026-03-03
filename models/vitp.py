import timm, torch
from kdyck.utils import *

class VitProcedural(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = timm.create_model(
            args.model,  
            pretrained=False, 
            no_embed_class=True, 
            num_classes=args.k*2,   
        )

        self.args = args

        embeddings = torch.load(args.embeddings_path)
        self.embedding = torch.nn.Embedding.from_pretrained(embeddings, freeze=args.freeze_patch_embeddings)
        self.model.pos_embed.requires_grad = not args.freeze_pos_embeddings
        print("Model initialized with patch embedding shape:", self.embedding.weight.shape)
        print("State of pos_embed requires_grad:", self.model.pos_embed.requires_grad)
        print("State of embedding weight requires_grad:", self.embedding.weight.requires_grad)

    def forward(self, x):
        try:
            model = self.model.module
        except:
            model = self.model

        x_embed = self.embedding(x)
        if self.args.skip_pos_embeddings:
            x = x_embed
        else:
            if self.args.procedural_order == "standard":
                temp_pos = torch.arange(x.shape[1]).unsqueeze(0)
            elif self.args.procedural_order == "vertical":
                temp_pos = vertical_unravel(torch.arange(x.shape[1]).unsqueeze(0))
            elif self.args.procedural_order == "row_alternate":
                temp_pos = row_alternate(torch.arange(x.shape[1]).unsqueeze(0))
            else:
                raise NotImplementedError(f"Procedural order {self.args.procedural_order} not implemented")

            if self.args.shuffle == "pos":
                pos_embed = model.pos_embed[:, temp_pos[0], :]
                x = x_embed + pos_embed
            if self.args.shuffle == "both":
                x = x_embed + model.pos_embed[:, :, :]
                x = x[:, temp_pos[0], :]
            else:
                x = x_embed + model.pos_embed[:, :, :]
        x = model.pos_drop(x)
        x = model.norm_pre(x)
        for blk in model.blocks:
            x = blk(x)
        x = model.norm(x)
        x = model.fc_norm(x)
        x = model.head(x)
        return x

    def set_train(self):
        try:
            self.model.train()
            self.model.pos_embed.requires_grad = not self.args.freeze_pos_embeddings
        except:
            self.model.module.train()
            self.model.module.pos_embed.requires_grad = not self.args.freeze_pos_embeddings

    def set_eval(self):
        try:
            self.model.eval()
        except:
            self.model.module.eval()
        