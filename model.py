import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        features = self.bn(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, dropout=0.25):
        super(DecoderRNN, self).__init__()
        # class variables
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        ## Model layers
        # Embedding layer to turn words into vectors of specified sizes
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        # LSTM layer to take embedded word vectors as inputs and outputs hidden states of hidden sizes
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, dropout, batch_first=True)
        
        # set dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # final fully connected linear layer to map hidden state to the vocabulary output size
         
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        # initialize the weights
        self.init_weights()
        
    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.xavier_uniform_(self.word_embeddings.weight)
    
    def forward(self, features, captions):
        # create emdedded word vectors for each caption omitting the start and end tag
        embeddings = self.word_embeddings(captions[:,:-1])
        # 
        embeddings = torch.cat((features.unsqueeze(dim=1), embeddings), dim=1)
        
        # pass the lstm over word embeddings to get the output and hidden state
        lstm_out, _ = self.lstm(embeddings)
        
        # pass output through dropout
        lstm_out = self.dropout(lstm_out)
                                
        # Fully connected linear layer as output for vocabulary caption
        output = self.fc(lstm_out)
        
        return output

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        captions = []
        
        for i in range(max_len):
            output, states = self.lstm(inputs, states)
            output = self.fc(output.squeeze(1))
            
            # predict the next word
            pred_word = output.argmax(dim=1)
            captions.append(pred_word.item())
            
            inputs = self.word_embeddings(pred_word)
            inputs = inputs.unsqueeze(1)
        
        return captions