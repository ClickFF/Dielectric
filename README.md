# Dielectric
## Purpose
The dielectric constant, specifically the static relative permittivity, is a fundamental physicochemical property that governs the electrical behavior of organic compounds, aiding in assessing solvation free energy and bioactivity. Since the dielectric constant is cumbersome to experimentally measure, it is important to be able to predict the dielectric constant to design compounds for desired electrochemical properties. Thus, these models are trained to predict the dielectric constant from structure of compound, from SMILES, and temperature alone.
## Files:
The models are separated by the model types: variational autoencoder(VAE), XGBoost regression, and large language model(LLM). For each model type, there are 4 models, based on the prediction goals: dielectric constant(reg), log2 of dielectric constant(log), reciprocal of the dielectric constant(recp), and the kirkwood function(kw). For the LLM models, all models used the same tokenizer, as provided in the folder. For the VAE models, there are two files for each model: one from the VAE itself(vae) and one for the prediction head(pred). All the models output the normalized prediction tasks based on the training dataset distribution.
## Specifications:
RDKit: version 2025.03.6 

Scikit-learn: version 1.7.2 

PyTorch: version 2.9.1 

PyTorch Geometric: version 2.7.0 

ChemBERTa: ChemBERTa-77M-MTR 

## Usage:
### XGBoost Regression:
To initialize the XGBoost specifications: 
```python
regr = GradientBoostingRegressor(n_estimators=4000, random_state=64, max_depth=4, learning_rate=0.03)
```
To load the trained models:
```python
regr= joblib.load('regr_reg.joblib')
```
To make predictions with the model:
```python
predlbl = regr.predict(X_test)
```
### Variational AutoEncoder:
The VAE is modified from AttentiveFP models:
```python
class GATEConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__(aggr='add', node_dim=0)

        self.dropout = dropout

        self.att_l = Parameter(torch.empty(1, out_channels))
        self.att_r = Parameter(torch.empty(1, in_channels))

        self.lin1 = Linear(in_channels + edge_dim, out_channels, False)
        self.lin2 = Linear(out_channels, out_channels, False)

        self.bias = Parameter(torch.empty(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att_l)
        glorot(self.att_r)
        glorot(self.lin1.weight)
        glorot(self.lin2.weight)
        zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor) -> Tensor:
        # edge_updater_type: (x: Tensor, edge_attr: Tensor)
        alpha = self.edge_updater(edge_index, x=x, edge_attr=edge_attr)

        # propagate_type: (x: Tensor, alpha: Tensor)
        out = self.propagate(edge_index, x=x, alpha=alpha)
        out = out + self.bias
        return out

    def edge_update(self, x_j: Tensor, x_i: Tensor, edge_attr: Tensor,
                    index: Tensor, ptr: OptTensor,
                    size_i: Optional[int]) -> Tensor:
        x_j = F.leaky_relu_(self.lin1(torch.cat([x_j, edge_attr], dim=-1)))
        alpha_j = (x_j @ self.att_l.t()).squeeze(-1)
        alpha_i = (x_i @ self.att_r.t()).squeeze(-1)
        alpha = alpha_j + alpha_i
        alpha = F.leaky_relu_(alpha)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return self.lin2(x_j) * alpha.unsqueeze(-1)

class AFPEncoder(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        edge_dim: int,
        num_layers: int,
        num_timesteps: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.num_timesteps = num_timesteps
        self.dropout = dropout

        self.lin1 = Linear(in_channels, hidden_channels)

        self.gate_conv = GATEConv(hidden_channels, hidden_channels, edge_dim,
                                  dropout)
        self.gru = GRUCell(hidden_channels, hidden_channels)

        self.atom_convs = torch.nn.ModuleList()
        self.atom_grus = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            conv = GATConv(hidden_channels, hidden_channels, dropout=dropout,
                           add_self_loops=False, negative_slope=0.01)
            self.atom_convs.append(conv)
            self.atom_grus.append(GRUCell(hidden_channels, hidden_channels))

        self.conv_mu = GATv2Conv(in_channels=hidden_channels, out_channels=out_channels, edge_dim=edge_dim, heads=8, concat=False)
        self.conv_logstd = GATv2Conv(in_channels=hidden_channels, out_channels=out_channels, edge_dim=edge_dim, heads=8, concat=False)


    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor,
                batch: Tensor, return_attn=False) -> Tensor:
        """"""  # noqa: D419
        # Atom Embedding:
        x = F.leaky_relu_(self.lin1(x))

        h = F.elu_(self.gate_conv(x, edge_index, edge_attr))
        h = F.dropout(h, p=self.dropout, training=self.training)
        x = self.gru(h, x).relu_()

        for conv, gru in zip(self.atom_convs, self.atom_grus):
            h = conv(x, edge_index)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = gru(h, x).relu()
        if not return_attn:
            return self.conv_mu(x, edge_index, edge_attr, batch), self.conv_logstd(x, edge_index, edge_attr, batch)
        else:
            mu, (edge_index, attn_weights) = self.conv_mu(x, edge_index, edge_attr, return_attention_weights=True)
            return mu, attn_weights.mean(dim=1)
        
class Decoder(torch.nn.Module):
    def __init__(self, latent_dim, hidden_dim, out_dim):
        super(Decoder, self).__init__()
        self.edge_decoder = InnerProductDecoder()
        self.atom_predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim) 
        )
    
    def forward(self, z, edge_index, sigmoid):
        return self.edge_decoder(z, edge_index, sigmoid)
    
    def predict(self, z, masked_index):
        return self.atom_predictor(z[masked_index])

def mask_attributes(data, mask_prob=0.15):
    masked_data = data.clone()
    
    num_atoms = data.x.size(0)
    indices = torch.randperm(num_atoms)[:int(num_atoms * mask_prob)]

    targets = data.x[indices].clone()
    masked_data.x[indices] = 0.0 
    
    return masked_data, indices, targets

class AFPPredictor(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        edge_dim: int,
        num_layers: int,
        num_timesteps: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.num_timesteps = num_timesteps
        self.dropout = dropout

        self.mol_ini = GATv2Conv(in_channels=in_channels, out_channels=hidden_channels, dropout=dropout, edge_dim=edge_dim, heads=8, concat=False)
        self.mol_conv = GATConv(hidden_channels, hidden_channels,
                                dropout=dropout, add_self_loops=False,
                                negative_slope=0.01)
        self.mol_conv.explain = False 
        self.mol_gru = GRUCell(hidden_channels, hidden_channels)

        gate_nn = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.Sigmoid()
        )
        
        self.pool = AttentionalAggregation(gate_nn=gate_nn)

        self.lin2 = Linear(hidden_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor,
                batch: Tensor) -> Tensor:
        """"""  # noqa: D419

        row = torch.arange(batch.size(0), device=batch.device)
        global_edge_index = torch.stack([row, batch], dim=0)

        x = F.leaky_relu(self.mol_ini(x, edge_index, edge_attr, batch))

        out = global_add_pool(x, batch).relu()
        for _ in range(self.num_timesteps):
            h = F.elu_(self.mol_conv((x, out), global_edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
            out = self.mol_gru(h, out).relu_()

        out = F.dropout(out, p=self.dropout, training=self.training)
        return self.lin2(out)

encoder = AFPEncoder(in_channels=12, hidden_channels=512, out_channels=128, edge_dim=4, num_layers=5, num_timesteps=4, dropout=0.25)
decoder = Decoder(128, 64, 12)
predictor = AFPPredictor(in_channels=128, hidden_channels=512, out_channels=1, edge_dim=4, num_layers=5, num_timesteps=4, dropout=0.25)
model = VGAE(encoder, decoder)
```
To load the saved models:
```python
model.load_state_dict(torch.load(“vae_reg.pth”))
predictor.load_state_dict(torch.load(“pred_reg.pth”))
```
To make predictions:
```python
SMILE_CHARSET = '["C", "B", "F", "I", "H", "O", "N", "S", "P", "Cl", "Br"]'
bond_mapping = {"SINGLE": 0, "DOUBLE": 1, "TRIPLE": 2, "AROMATIC": 3}
bond_mapping.update(
    {0: BondType.SINGLE, 1: BondType.DOUBLE, 2: BondType.TRIPLE, 3: BondType.AROMATIC}
)
SMILE_CHARSET = ast.literal_eval(SMILE_CHARSET)

SMILE_to_index = dict((c, i) for i, c in enumerate(SMILE_CHARSET))
index_to_SMILE = dict((i, c) for i, c in enumerate(SMILE_CHARSET))
atom_mapping = dict(SMILE_to_index)
atom_mapping.update(index_to_SMILE)

batch_size = 32
ATOM_DIM = len(SMILE_CHARSET)

def smiles_to_graph(smiles, temp):
    # Converts SMILES to molecule object
    molecule = Chem.MolFromSmiles(smiles)
    molecule = Chem.AddHs(molecule)

    features = np.zeros((len(molecule.GetAtoms()), ATOM_DIM), "float32")

    # loop over each atom in molecule
    for atom in molecule.GetAtoms():
        i = atom.GetIdx()
        atom_type = atom_mapping[atom.GetSymbol()]
        features[i] = np.eye(ATOM_DIM)[atom_type]

    tempvect = np.full((len(molecule.GetAtoms()), 1), temp)
    features = np.concat((features, tempvect), axis=1)

    features[np.where(np.sum(features, axis=1) == 0)[0], -1] = 1

    return features.tolist()

def mol_to_graph(smiles, temperature):
    molecule = Chem.MolFromSmiles(smiles)
    molecule = Chem.AddHs(molecule)
    adjacency_matrix = rdmolops.GetAdjacencyMatrix(molecule)
    try:
        atom_features = smiles_to_graph(smiles, temperature)
    except:
        return False
    edge_features = []
    bond_properties = ['GetBondTypeAsDouble', 'GetIsAromatic', 'GetIsConjugated']
    for bond in molecule.GetBonds():
        temp = []
        for prop in bond_properties:
            try:
                value = getattr(bond, prop)()
                temp.append(float(value))
            except Exception as e:
                print(f"{prop}: Error - {e}")
        temp = temp + [float(bond.IsInRing())]
        edge_features.append(temp)

    edge_feat = []
    edge_src = []
    edge_dst = []
    adj = []
    for row in adjacency_matrix:
        lirow = list(row)
        adj.append(lirow)

    distinct = []
    for i in range(0, len(adj)):
        pos = len(edge_src)
        dpos = len(distinct)
        for l in range(1, len(adj[i]) + 1):
            j = len(adj[i]) - l
            bond = adj[i][j]
            if bond == 1:
                if j > i:
                    distinct.insert(dpos, [i, j])
                else:
                    ri = distinct.index([j, i])
                edge_src.insert(pos, i)
                edge_dst.insert(pos, j)
    for ind in range(0, len(edge_src)):
        start = edge_src[ind]
        end = edge_dst[ind]
        if end > start:
            bond = distinct.index([start, end])
            edge_feat.append(edge_features[bond])
        else:
            bond = distinct.index([end, start])
            edge_feat.append(edge_features[bond])
    
    sparse = [edge_src, edge_dst]

    return [atom_features, sparse, edge_feat]

graph = mol_to_graph(smiles, temp) # Temperature has to be normalized

driver = []
    if graph != False:
        driver.append(Data(
            x = torch.tensor(np.array(graph[0]), dtype=torch.float),
            edge_index = torch.tensor(np.array(graph[1]), dtype=torch.float).long(),
            edge_attr = torch.tensor(np.array(graph[2]), dtype=torch.float),
            y = torch.tensor(np.array(34), dtype=torch.float),
            ))
    driv = DataLoader(driver, batch_size=1, shuffle=False, collate_fn=collate_fn)


mu, _ = model(mol.x, mol.edge_index, mol.edge_attr, mol.batch)
predlbl = predictor(mu, mol.edge_index, mol.edge_attr, mol.batch)
```
### Large Language Model:
The model was a modified version of ChemBERTa’s model:
```python
class CustomChemBertaForRegression(RobertaPreTrainedModel):
    def __init__(self, config, extra_feature_dim=1):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.extra_feature_dim = extra_feature_dim
        
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        modified_hidden_size = config.hidden_size + self.extra_feature_dim
        
        config.hidden_size = modified_hidden_size
        self.classifier = RobertaClassificationHead(config)
        config.hidden_size = modified_hidden_size - self.extra_feature_dim
        
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, temps=None, constants=None, **kwargs):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        
        sequence_output = outputs[0]
        cls_embedding = sequence_output[:, 0, :] # Shape: [batch_size, hidden_size]
        
        if temps.dim() == 1:
            temps = temps.unsqueeze(-1)
            
        modified_embedding = torch.cat((cls_embedding, temps), dim=-1)
        logits = self.classifier(modified_embedding.unsqueeze(1))
        
        loss = None
        if constants is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits.squeeze(), constants.squeeze())
            
        return {"loss": loss, "logits": logits, "attentions": outputs["attentions"]} if loss is not None else {"logits": logits}

To load the saved model:
tokenizer = AutoTokenizer.from_pretrained("./llm_tokenizer")
config = AutoConfig.from_pretrained("./llm_reg")

model = CustomChemBertaForRegression.from_pretrained(
    "./llm_reg", 
    config=config,
    extra_feature_dim=1 
)
```
To make a prediction:
```python
smile = smiles
temp = temp
constant = constant # true dielectric constant, used to calculate loss

inputs = tokenizer(smile, padding=True, truncation=True, return_tensors="pt")
outputs = model(
    input_ids=inputs["input_ids"], 
    attention_mask=inputs["attention_mask"], 
    temps=torch.tensor(temp, dtype=torch.float),
    constants=torch.tensor([constant, constant], dtype=torch.float)
)

predlbl = float(outputs["logits"][0])
```

For more information, please look at the associated paper.
