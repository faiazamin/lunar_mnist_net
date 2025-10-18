import torch, mlflow, yaml
from torch import nn
from dataset import get_loaders
from model import MNISTNet
from utils import seed_everything, accuracy

def train(config_path="src/config.yaml"):
    # load config
    cfg = yaml.safe_load(open(config_path))
    seed_everything(cfg["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dl, val_dl = get_loaders(cfg["batch_size"])
    model = MNISTNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.CrossEntropyLoss()

    mlflow.set_experiment("mnist_repro_pipeline")
    with mlflow.start_run():
        mlflow.log_params(cfg)
        for epoch in range(cfg["epochs"]):
            model.train()
            total_loss = 0
            for xb, yb in train_dl:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                preds = model(xb)
                loss = loss_fn(preds, yb)
                loss.backward()
                opt.step()
                total_loss += loss.item()
            train_loss = total_loss / len(train_dl)

            # validation
            model.eval()
            val_loss, acc = 0, 0
            with torch.no_grad():
                for xb, yb in val_dl:
                    xb, yb = xb.to(device), yb.to(device)
                    preds = model(xb)
                    val_loss += loss_fn(preds, yb).item()
                    acc += accuracy(preds, yb)
            val_loss /= len(val_dl)
            acc /= len(val_dl)

            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss, "val_acc": acc}, step=epoch)
            print(f"Epoch {epoch+1}/{cfg['epochs']} | train:{train_loss:.4f} | val:{val_loss:.4f} | acc:{acc:.4f}")

        torch.save(model.state_dict(), "model.pt")
        mlflow.pytorch.log_model(model, "model")

if __name__ == "__main__":
    train()