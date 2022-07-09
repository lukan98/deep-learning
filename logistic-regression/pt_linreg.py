import torch
import torch.optim as optim

a = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

X = torch.tensor([1, 2, 5])
Y = torch.tensor([3, 5, 11])

optimizer = optim.SGD([a, b], lr=0.01)

for i in range(1000):
    # afin regresijski model
    Y_ = a * X + b

    diff = (Y - Y_)

    # kvadratni gubitak -- postavljeno računanje srednje vrijednosti da gubitak ne bi bio
    # prevelik za velike skupine podataka
    loss = torch.mean(diff ** 2)

    # ručno izračunati gradijenti
    dL_da = -2 / len(X) * torch.dot(diff, torch.tensor(X, dtype=torch.float32))
    dL_db = -2 / len(X) * torch.sum(diff)

    # računanje gradijenata
    loss.backward()

    # korak optimizacije
    optimizer.step()

    print(f'step: {i}, loss:{loss}, Y_:{Y_}, a:{a}, b {b}')
    # dodan ispis gradijenata u svakom koraku
    print(f'\tgrad_a:{a.grad}, grad_b:{b.grad}')
    # dodan ispis ručno izračunatog gradijenata u svakom koraku
    print(f'\tmanual grad_a:{dL_da}, manual grad_b:{dL_db}')

    # Postavljanje gradijenata na nulu
    optimizer.zero_grad()
