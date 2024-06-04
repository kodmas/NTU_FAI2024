import torch
from tqdm.auto import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
"""
Implementation of Autoencoder
"""
class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, encoding_dim: int) -> None:
        """
        Modify the model architecture here for comparison
        """
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.Linear(encoding_dim, encoding_dim//2),
            # nn.Linear(encoding_dim//2, encoding_dim//3),
            # nn.Linear(encoding_dim//3, encoding_dim//4),
            # nn.Linear(encoding_dim//4, encoding_dim//4),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            # nn.Linear(encoding_dim//4, encoding_dim//3),
            # nn.Linear(encoding_dim//3, encoding_dim//2),
            nn.Linear(encoding_dim//2, encoding_dim),
            nn.Linear(encoding_dim, input_dim),
        )
    
    def forward(self, x):
        #TODO: 5%
        # Hint: a forward pass includes one pass of encoder and decoder
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def fit(self, X, epochs=10, batch_size=32):
        #TODO: 5%
        # Hint: a regular pytorch training includes:
        # 1. define optimizer
        # 2. define loss function
        # 3. define number of epochs
        # 4. define batch size
        # 5. define data loader
        # 6. define training loop
        # 7. record loss history 
        # Note that you can use `self(X)` to make forward pass.
        
        optimizer = optim.Adam(self.parameters(),lr = 1e-3)
        

        # compute error between g(x), x 
        loss_function = nn.MSELoss()

        data_loader = DataLoader(TensorDataset(torch.tensor(X,dtype=torch.float)),batch_size = batch_size,shuffle = True)

        loss_history = []
        for epoch in tqdm(range(epochs)):
            epoch_loss = 0
            for batch in data_loader:
                optimizer.zero_grad()
                x = torch.cat(batch)
                x_hat = self.forward(x)
                loss = loss_function(x_hat,x)
                loss.backward()
                optimizer.step()
                loss_history.append(loss.item())
                epoch_loss += loss.item()
        #     if epoch%10 == 0:
        #         print("Epoch: ",epoch," Loss: ",epoch_loss/len(data_loader))
        # print("Training Loss: ",loss_history[-1])
        # plt.plot(loss_history)
        # plt.xlabel('Epoch')
        # plt.ylabel('Loss')
        # plt.title('Training Loss')
        
        # plt.savefig('autoencoder_loss.png')
        # plt.close()
        
    
    def transform(self, X):
        #TODO: 2%
        #Hint: Use the encoder to transform X
        self.eval()
        return self.encoder(torch.tensor(X).float()).detach().numpy()
    
    def reconstruct(self, X):
        #TODO: 2%
        #Hint: Use the decoder to reconstruct transformed X
        self.eval()
        X_transformed = self.transform(torch.tensor(X).float())

        return self.decoder(torch.tensor(X_transformed).float()).detach().numpy()


"""
Implementation of DenoisingAutoencoder
"""
class DenoisingAutoencoder(Autoencoder):
    def __init__(self, input_dim, encoding_dim, noise_factor=0.2):
        super(DenoisingAutoencoder, self).__init__(input_dim,encoding_dim)
        self.noise_factor = noise_factor
    
    def add_noise(self, x):
        #TODO: 3%
        #Hint: Generate Gaussian noise with noise_factor
        mean = torch.zeros(x.shape)
        std = torch.ones(x.shape) * self.noise_factor
        return x + torch.normal(mean,std)
    
    def fit(self, X, epochs=10, batch_size=32):
        #TODO: 4%
        #Hint: Follow the same procedure above but remember to add_noise before training.
        optimizer = optim.Adam(self.parameters(),lr = 1e-3)
        # optimizer = optim.RMSprop(self.parameters(),lr = 1e-3)
        # optimizer = optim.SGD(self.parameters(),lr = 1e-3)
        # compute error between g(x), x
        loss_function = nn.MSELoss()

        data_loader = DataLoader(TensorDataset(torch.tensor(X,dtype=torch.float)),batch_size = batch_size,shuffle = True)

        loss_history = []
        for epoch in tqdm(range(epochs)):
            epoch_loss = 0
            for batch in data_loader:
                optimizer.zero_grad()
                x = torch.cat(batch)
                x = self.add_noise(x)
                x_hat = self.forward(x)
                loss = loss_function(x_hat,x)
                loss.backward()
                optimizer.step()
                loss_history.append(loss.item())
                epoch_loss += loss.item()
        #     if epoch%10 == 0:
        #         print("Epoch: ",epoch," Loss: ",epoch_loss/len(data_loader))

        # print("Training Loss: ",loss_history[-1])
        # plt.plot(loss_history)
        # plt.xlabel('Epoch')
        # plt.ylabel('Loss')
        # plt.title('Training Loss')

        # plt.savefig('denoising_autoencoder_loss.png')
