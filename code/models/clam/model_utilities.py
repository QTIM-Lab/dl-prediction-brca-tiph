# PyTorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F



# Function: Initialize NN weights
def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()
        
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)



# Class: Attention Network (CLAM)
class AttentionNet(nn.Module):

    """
    Attention Network without Gating (2 fc layers)
    Args:
        L: input feature dimension
        D: hidden layer dimension
        dropout: whether to use dropout
        dropout_prob: the probability of dropout (default = 0.25)
        n_classes: number of classes 
    """


    # Method: __init__
    def __init__(self, L=1024, D=256, dropout=False, dropout_prob=0.25, n_classes=1):
        super(AttentionNet, self).__init__()
        
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()
        ]

        if dropout:
            self.module.append(nn.Dropout(p=dropout_prob))

        self.module.append(nn.Linear(D, n_classes))
        
        self.module = nn.Sequential(*self.module)

        return
    

    # Method: forwards
    def forward(self, x):

        # The shape of the returned tensor is: [b, N, n_classes]

        return self.module(x), x



# Class: Attention Network with Sigmoid Gating (CLAM)
class AttentionNetGated(nn.Module):

    """
    Attention Network with Sigmoid Gating (3 fc layers)
    Args:
        L: input feature dimension
        D: hidden layer dimension
        dropout: whether to use dropout
        dropout_prob: the probability of dropout (default = 0.25)
        n_classes: number of classes 
    """


    # Method: __init__
    def __init__(self, L=1024, D=256, dropout=False, dropout_prob=0.25, n_classes=1):
        super(AttentionNetGated, self).__init__()
        
        # Attention a
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()
        ]
        
        # Attention b
        self.attention_b = [
            nn.Linear(L, D),
            nn.Sigmoid()
        ]

        if dropout:
            self.attention_a.append(nn.Dropout(p=dropout_prob))
            self.attention_b.append(nn.Dropout(p=dropout_prob))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

        return


    # Method: forward
    def forward(self, x):

        # The shape of tensor A should be [b, N, n_classes]
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)

        return A, x



# Class: AM_SB
class AM_SB(nn.Module):

    """
    Args:
        gate: whether to use gated attention network
        size_arg: config for network size
        dropout: whether to use dropout
        dropout_prob: the probability of dropout (default = 0.25)
        n_classes: number of classes 
    """


    # Method: __init__
    def __init__(self, gate=True, size_arg="small", dropout=False, dropout_prob=0.25, n_classes=2, encoding_size=1024):
        super(AM_SB, self).__init__()

        # Build lists of sizes for the layers, following the rationale of the first version of CLAM
        small = [int(encoding_size * f) for f in (1, 0.5, 0.25)]
        big = [int(encoding_size * f) for f in (1, 0.5, 0.75)]
        self.size_dict = {
            "small":small,
            "big":big
        }
        # self.size_dict = {
        #     "small": [1024, 512, 256], 
        #     "big": [1024, 512, 384]
        # }
        size = self.size_dict[size_arg]
        

        # Build feature extractor
        fc = [
            nn.Linear(size[0], size[1]), 
            nn.ReLU()
        ]
        

        # Dropout with probability `dropout_prob`
        if dropout:
            fc.append(nn.Dropout(p=dropout_prob))


        # Attention Net
        if gate:
            attention_net = AttentionNetGated(
                L=size[1], 
                D=size[2],
                dropout=dropout,
                dropout_prob=dropout_prob,
                n_classes=1
            )
        else:
            attention_net = AttentionNet(
                L = size[1],
                D = size[2], 
                dropout = dropout, 
                dropout_prob=dropout_prob,
                n_classes = 1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        
        
        # Classifiers
        self.classifier = nn.Linear(size[1], n_classes)


        
        # Assign class variables
        self.n_classes = n_classes

        # Initialize model's weights
        initialize_weights(self)

        return


    # Method: forward
    def forward(self, h):

        # Shape: b X N X K (or NxK)
        A, h = self.attention_net(h)
        
        # Shape: b X K X N (or KxN)
        A = torch.transpose(A, 2, 1)
 
        # Apply Softmax over dim=2 (i.e., N) or last dimension
        A_act = F.softmax(A, dim=2)  # softmax over N

        # To use this torch.mm function, the Tensors shouldn't have a batch dimension
        # M = torch.mm(A, h)
        # So, we squeeze them for the right dimensional shapes
        A_act_ = torch.squeeze(A_act, dim=1)
        h_ = torch.squeeze(h, dim=0)

        # Get features after attention
        M = torch.mm(A_act_, h_)

        # Compute the logits
        logits = self.classifier(M)

        # Apply Softmax to dim=1 to get a probability vector with shape [b, n_classes]
        y_proba = F.softmax(logits, dim=1)
        y_pred = torch.argmax(y_proba, dim=1)

        # Create a dictionary for the model outputs
        ouput_dict = {
            "logits":logits,
            "y_proba":y_proba,
            "y_pred":y_pred,
            "A_raw":A,
            "features": M
        }

        return ouput_dict



# Class: AM_SB_Regression
class AM_SB_Regression(nn.Module):

    """
    Args:
        gate: whether to use gated attention network
        size_arg: config for network size
        dropout: whether to use dropout
        dropout_prob: the probability of dropout (default = 0.25)
    """


    # Method: __init__
    def __init__(self, gate=True, size_arg="small", dropout=False, dropout_prob=0.25, encoding_size=1024):
        super(AM_SB_Regression, self).__init__()

        # Build lists of sizes for the layers, following the rationale of the first version of CLAM
        small = [int(encoding_size * f) for f in (1, 0.5, 0.25)]
        big = [int(encoding_size * f) for f in (1, 0.5, 0.75)]
        self.size_dict = {
            "small":small,
            "big":big
        }
        # self.size_dict = {
        #     "small": [1024, 512, 256], 
        #     "big": [1024, 512, 384]
        # }
        size = self.size_dict[size_arg]
        

        # Build feature extractor
        fc = [
            nn.Linear(size[0], size[1]), 
            nn.ReLU()
        ]
        

        # Dropout with probability `dropout_prob`
        if dropout:
            fc.append(nn.Dropout(p=dropout_prob))


        # Attention Net
        if gate:
            attention_net = AttentionNetGated(
                L=size[1], 
                D=size[2],
                dropout=dropout,
                dropout_prob=dropout_prob,
                n_classes=1
            )
        else:
            attention_net = AttentionNet(
                L = size[1],
                D = size[2], 
                dropout = dropout, 
                dropout_prob=dropout_prob,
                n_classes = 1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        
        
        # Classifiers
        self.classifier = nn.Linear(size[1], 1)


        # Initialize model's weights
        initialize_weights(self)

        return


    # Method: forward
    def forward(self, h):

        # Shape: b X N X K (or NxK)
        A, h = self.attention_net(h)
        
        # Shape: b X K X N (or KxN)
        A = torch.transpose(A, 2, 1)
 
        # Apply Softmax over dim=2 (i.e., N) or last dimension
        A_act = F.softmax(A, dim=2)  # softmax over N

        # To use this torch.mm function, the Tensors shouldn't have a batch dimension
        # M = torch.mm(A, h)
        # So, we squeeze them for the right dimensional shapes
        A_act_ = torch.squeeze(A_act, dim=1)
        h_ = torch.squeeze(h, dim=0)

        # Get features after attention
        M = torch.mm(A_act_, h_)

        # Compute the logits
        logits = self.classifier(M)

        # Create a dictionary for the model outputs
        ouput_dict = {
            "logits":logits,
            "A_raw":A,
            "features": M
        }

        return ouput_dict



# Class: AM_MB
class AM_MB(AM_SB):
    def __init__(self, gate=True, size_arg="small", dropout=False, dropout_prob=0.25, n_classes=2, encoding_size=1024):
        nn.Module.__init__(self)

        # Build lists of sizes for the layers, following the rationale of the first version of CLAM
        small = [int(encoding_size * f) for f in (1, 0.5, 0.25)]
        big = [int(encoding_size * f) for f in (1, 0.5, 0.75)]
        self.size_dict = {
            "small":small,
            "big":big
        }
        # self.size_dict = {
        #     "small": [1024, 512, 256],
        #     "big": [1024, 512, 384]
        # }
        size = self.size_dict[size_arg]


        # Build feature extractor
        fc = [
            nn.Linear(size[0], size[1]), 
            nn.ReLU()
        ]
        
        # Dropout with probability `dropout_prob`
        if dropout:
            fc.append(nn.Dropout(0.25))

        # Attention Net
        if gate:
            attention_net = AttentionNetGated(
                L=size[1], 
                D=size[2], 
                dropout=dropout, 
                dropout_prob=dropout_prob,
                n_classes=n_classes
            )
        else:
            attention_net = AttentionNet(
                L=size[1], 
                D=size[2], 
                dropout=dropout, 
                dropout_prob=dropout_prob,
                n_classes=n_classes
            )
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        
        # Build a classifier for each class
        bag_classifiers = [nn.Linear(size[1], 1) for i in range(n_classes)]
        self.classifiers = nn.ModuleList(bag_classifiers)
        
        
        # Class variables
        self.n_classes = n_classes

        # Initalize model's weights
        initialize_weights(self)

        return


    # Method: forward
    def forward(self, h):

        # Shape: b X N X K (or NxK)
        A, h = self.attention_net(h)
        
        # Shape: b X K X N (or KxN)
        A = torch.transpose(A, 2, 1)
        
        
        # Apply Softmax over dim=2 (i.e., N) or last dimension
        A_act = F.softmax(A, dim=2)  # softmax over N

        # To use this torch.mm function, the Tensors shouldn't have a batch dimension
        # M = torch.mm(A, h)
        # So, we squeeze them for the right dimensional shapes
        A_act_ = torch.squeeze(A_act, dim=0)
        h_ = torch.squeeze(h, dim=0)

        # Get features after attention
        M = torch.mm(A_act_, h_)

        # Everything seems fine
        logits = torch.empty(1, self.n_classes).float().to(M.device)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](M[c])

        # y prediction
        y_pred = torch.topk(logits, 1, dim = 1)[1]
        
        # y probability
        y_proba = F.softmax(logits, dim = 1)

        # Create a dictionary for the model outputs
        ouput_dict = {
            "logits":logits,
            "y_proba":y_proba,
            "y_pred":y_pred,
            "A_raw":A,
            "features": M
        }

        return ouput_dict
