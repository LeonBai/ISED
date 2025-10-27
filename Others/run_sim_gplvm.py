import pickle
import numpy as np
import GPy
from GPy.mappings import MLP

with open('../SeED/dataset/Xs.pkl','rb') as file:
    
    raw_data = pickle.load(file)
    
from sklearn import preprocessing

gplvm_embeddings = []

for i in range(3):
    X_normalized = preprocessing.MinMaxScaler().fit_transform(raw_data[i])

    train_data = X_normalized[:500]
    test_data = X_normalized[500:500+500]

    zs = []
    for run_idx in range(10):
        print(f'=== Run {run_idx+1:02d}/10 ===')
    
        latent_dim = 2
        kernel = GPy.kern.RBF(input_dim=latent_dim)

        # 4. Build a back-constrained GPLVM
        mapping = MLP(input_dim=train_data.shape[1],    # here 2
                      output_dim=latent_dim,          # here 2
                      hidden_dim=32,                  # you can tune this
                      name='back_mlp')

        model = GPy.models.BCGPLVM(
            Y=train_data,
            input_dim=latent_dim,
            kernel=kernel,
            mapping=mapping
        )

        # 5) Optimize both the GPLVM and the MLP-encoder jointly
        model.optimize(messages=True, max_iters=5000)


        gplvm_embedding = preprocessing.MinMaxScaler().fit_transform(model.mapping.f(test_data))
        zs.append(gplvm_embedding)
    zs = np.stack(zs, axis=0)
    gplvm_embeddings.append(zs)
gplvm_embeddings = np.stack(gplvm_embeddings)

np.save('sim_gplvm_embeddings_mlp_rbf_kernerl_d=2.npy',gplvm_embeddings)

print ('Finished!')