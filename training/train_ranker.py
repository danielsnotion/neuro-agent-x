import torch
from models.dataset import get_dataset
from engine.neural_scorer import encode_program
from models.ranker import ProgramRanker

def train():
    data = get_dataset()

    X = []
    y = []

    for item in data:
        X.append(encode_program(item["program"]))
        y.append(item["score"])

    X = torch.tensor(X)
    y = torch.tensor(y).unsqueeze(1)

    model = ProgramRanker(input_dim=X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(100):
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}: Loss {loss.item()}")




if __name__ == "__main__":
    train()