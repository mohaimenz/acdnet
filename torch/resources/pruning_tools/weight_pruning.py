import torch
def copy_model_weights(model, W_flat, W_shapes, param_name=['weight']):
    offset = 0
    if isinstance(W_shapes, list):
        W_shapes = iter(W_shapes)
    for name, W in model.named_parameters():
        if name.strip().split(".")[-1] in param_name:
            name_, shape = next(W_shapes)
            if shape is None:
                continue
            assert name_ == name
            numel = W.numel()
            W.data.copy_(W_flat[offset: offset + numel].view(shape))
            offset += numel

def layers_n(model, normalized=True, param_name=['weight']):
    res = {}
    count_res = {}
    for name, W in model.named_parameters():
        if name.strip().split(".")[-1] in param_name and name.strip().split(".")[-2][:2] != "bn" and W.dim() > 1:
            layer_name = name
            W_n = W.data.view(-1)
            if W_n.dim() > 0:
                if not normalized:
                    res[layer_name] = W_n.shape[0]
                else:
                    res[layer_name] = float(W_n.shape[0]) / torch.numel(W)
                count_res[layer_name] = W_n.shape[0]
            else:
                res[layer_name] = 0
                count_res[layer_name] = 0

    return res, count_res

def l0norm(model, k, param_name=['weight']):
    # get all the weights
    W_shapes = []
    res = []
    # print(param_name);
    # print(model.named_parameters());
    for name, W in model.named_parameters():
        # print(name);
        # print(W.data);
        if name.strip().split(".")[-1] in param_name:
            # print(W.dim());
            if W.dim() == 1:
                W_shapes.append((name, None))
            else:
                W_shapes.append((name, W.data.shape))
                res.append(W.data.view(-1))
    # print(res);
    res = torch.cat(res, dim=0)
    # if normalized:
    #     assert 0.0 <= k <= 1.0
    #     nnz = round(res.shape[0] * k)
    # else:
    assert k >= 1 and round(k) == k
    nnz = k
    if nnz == res.shape[0]:
        z_idx = []
    else:
        _, z_idx = torch.topk(torch.abs(res), int(res.shape[0] - nnz), largest=False, sorted=False)
        res[z_idx] = 0.0
        copy_model_weights(model, res, W_shapes, param_name)
    # print('Prunned');
    return z_idx, W_shapes
