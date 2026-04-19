import torch
import torch.nn.functional as F
import torch.optim as optim
from typing import Optional, Dict, Union, Callable
from tqdm.notebook import tqdm, trange

from utils.utils import make_averager, permute_pixels

def test_model(
    test_dl: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    perm: Optional[torch.Tensor] = None,
    device: str = "cuda",
) -> Dict[str, Union[float, Callable[[Optional[float]], float]]]:

    model.eval()
    test_loss_averager = make_averager()

    correct = 0
    for data, target in test_dl:
        data, target = data.to(device), target.to(device)

        if perm is not None:
            data = permute_pixels(data, perm)

        output = model(data)
        test_loss_averager(F.cross_entropy(output, target).item())
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).cpu().sum().item()

    return {
        "accuracy": 100.0 * correct / len(test_dl.dataset),
        "loss_averager": test_loss_averager,
        "correct": correct,
    }


def fit(
    epochs: int,
    train_dl: torch.utils.data.DataLoader,
    test_dl: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    tag: str,
    models_accuracy: dict,
    perm: Optional[torch.Tensor] = None,
    device: str = "cuda",
) -> float:

    for epoch in trange(epochs, desc="train epoch"):
        model.train()
        train_loss_averager = make_averager()

        tqdm_iterator = tqdm(
            enumerate(train_dl),
            total=len(train_dl),
            desc="batch [loss: None]",
            leave=False,
        )

        for batch_idx, (data, target) in tqdm_iterator:
            data, target = data.to(device), target.to(device)

            if perm is not None:
                data = permute_pixels(data, perm)

            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            opt.step()
            opt.zero_grad()

            train_loss_averager(loss.item())
            tqdm_iterator.set_description(
                f"train batch [avg loss: {train_loss_averager(None):.3f}]"
            )
            tqdm_iterator.refresh()

        test_out = test_model(test_dl, model, perm, device)

        print(
            f"Epoch: {epoch}\n"
            f"Train set: Average loss: {train_loss_averager(None):.4f}\n"
            f"Test set: Average loss: {test_out['loss_averager'](None):.4f}, "
            f"Accuracy: {test_out['correct']}/{len(test_dl.dataset)} "
            f"({test_out['accuracy']:.0f}%)\n"
        )

    models_accuracy[tag] = test_out["accuracy"]
    return test_out["accuracy"]


def get_model_optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    return optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
