import numpy as np
import torch

# read in M, m, and r
file = open('word_word_correlation','r')
m, r = map(int, file.readline().split())

M = []
for i in range(m):
  M.append(list(map(float, file.readline().split())))

# SVD
U, S, Vh = np.linalg.svd(M)
s = np.sqrt(np.diag(S[:r]))
A = np.matmul(U[:,:r], s)
W = np.matmul(s, Vh.T[:r,:])

# set negatives to 0
A[A < 0] = 0
W[W < 0] = 0

# Turn everything into Tensors
M = torch.tensor(M, dtype=torch.float32)
A = torch.tensor(A, requires_grad=True, dtype=torch.float32)
W = torch.tensor(W, requires_grad=True, dtype=torch.float32)

# Machine Learn
A_optimizer = torch.optim.Adam([A],lr=.0088)
W_optimizer = torch.optim.Adam([W],lr=.0088)
optimizers = [A_optimizer, W_optimizer] # order seems to matter for lr

for e in range(1000*2):
  # Approximation of M
  AW = torch.matmul(A,W)

  # Loss
  L = torch.sum(torch.pow(M-AW,2))
  
  # Alternating Optimization
  optimizer = optimizers[e%2]

  # optimization
  optimizer.zero_grad()
  L.backward()
  optimizer.step()

  # nonnegative
  with torch.no_grad():
    if e%2 == 0:
      A[A < 0] = 0
    else:
      W[W < 0] = 0

  if e%2 == 0:
    print(f"epoch {e//2}, loss: {L}")

# output A and W
output_file = open('nmf_ans','w')
output = ""

A = A.detach().numpy()
W = W.detach().numpy()
for N in [A,W]:
  for row in N:
    for j, x in enumerate(row):
      if j != len(row) - 1:
        output += str(x) + " "
      else:
        output += str(x) + "\n"
output_file.write(output[:-1])