import os

print('Running k-Huber with p-SR...')
os.system('python run_synthetic.py k_huber Rademacher')

print('Running k-Huber with p-SG...')
os.system('python run_synthetic.py k_huber Gaussian')

print('Running e-SVR  with p-SR...')
os.system('python run_synthetic.py e_svr Rademacher')

print('Running e-SVR with p-SG...')
os.system('python run_synthetic.py e_svr Gaussian')