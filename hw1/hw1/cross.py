"""
   cross.py
   COMP9444, CSE, UNSW
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class Full3Net(torch.nn.Module):
    def __init__(self, hid):
        super(Full3Net, self).__init__()
        self.in_to_hid1 = torch.nn.Linear(2, hid)
        self.hid1_to_hid2 = torch.nn.Linear(hid, hid)
        self.hid2_to_out = torch.nn.Linear(hid, 1)

    def forward(self, input):
        hid1_sum = self.in_to_hid1(input)
        self.hid1 = torch.tanh(hid1_sum)
        hid2_sum = self.hid1_to_hid2(self.hid1)
        self.hid2 = torch.tanh(hid2_sum)
        output_sum = self.hid2_to_out(self.hid2)
        output = torch.sigmoid(output_sum)
        return output

class Full4Net(torch.nn.Module):
    def __init__(self, hid):
        super(Full4Net, self).__init__()
        self.in_to_hid1 = torch.nn.Linear(2, hid)
        self.hid1_to_hid2 = torch.nn.Linear(hid, hid)
        self.hid1_to_hid3 = torch.nn.Linear(hid, hid)
        self.hid2_to_out = torch.nn.Linear(hid, 1)
        self.hid1 = None
        self.hid2 = None
        self.hid3 = None

    def forward(self, input):
        hid1_sum = self.in_to_hid1(input)
        self.hid1 = torch.tanh(hid1_sum)
        hid2_sum = self.hid1_to_hid2(self.hid1)
        self.hid2 = torch.tanh(hid2_sum)
        hid3_sum = self.hid1_to_hid3(self.hid2)
        self.hid3 = torch.tanh(hid3_sum)
        output_sum = self.hid2_to_out(self.hid3)
        output = torch.sigmoid(output_sum)
        return output

class DenseNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(DenseNet, self).__init__()
        self.in_to_hid1 = torch.nn.Linear(2, num_hid)
        self.in_to_hid2 = torch.nn.Linear(2, num_hid)
        self.in_to_out = torch.nn.Linear(2, 1)
        self.hid1_to_hid2 = torch.nn.Linear(num_hid, num_hid)
        self.hid1_to_out = torch.nn.Linear(num_hid, 1)
        self.hid2_to_out = torch.nn.Linear(num_hid, 1)
        self.hid1 = None
        self.hid2 = None

    def forward(self, input):
        in_hid1_sum = self.in_to_hid1(input)
        self.hid1 = torch.tanh(in_hid1_sum)
        in_hid2_sum = self.in_to_hid2(input)
        hid1_hid2_sum = self.hid1_to_hid2(self.hid1)
        hid2_sum = in_hid2_sum + hid1_hid2_sum
        self.hid2 = torch.tanh(hid2_sum)
        in_out_sum = self.in_to_out(input)
        hid1_out_sum = self.hid1_to_out(self.hid1)
        hid2_out_sum = self.hid2_to_out(self.hid2)
        output_sum = in_out_sum + hid1_out_sum + hid2_out_sum
        output = torch.sigmoid(output_sum)
        return output
