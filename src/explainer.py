import shap
import torch
import numpy as np
import matplotlib.pyplot as plt

class LSTMExplainer:
    def __init__(self, model, background_data):
        self.model = model
        self.model.eval()
        
        self.seq_length = background_data.shape[1]
        self.n_features = background_data.shape[2]
        
        def model_wrapper(x):
            if isinstance(x, np.ndarray):
                x = torch.FloatTensor(x)
            if len(x.shape) == 2: 
                x = x.reshape(-1, self.seq_length, self.n_features)
            
            with torch.no_grad():
                output = self.model(x)
                return output[:, 0].numpy()
        
        background_2d = background_data.reshape(len(background_data), -1).numpy()
        
        n_summary = min(50, len(background_2d))
        background_summary = shap.kmeans(background_2d, n_summary)
        
        print(f"Number of background samples: {n_summary}")
        
        self.explainer = shap.KernelExplainer(
            model_wrapper,
            background_summary,
            link="identity"
        )

#one single api
    def explain_prediction(self, sequence, feature_names=None):
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(self.n_features)]
            
        sequence_2d = sequence.reshape(sequence.shape[0], -1)
        
        shap_values = self.explainer.shap_values(sequence_2d.numpy())
        
        shap_3d = np.array(shap_values).reshape(-1, self.seq_length, self.n_features)
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_3d.reshape(-1, self.n_features),
            feature_names=feature_names,
            show=False
        )
        plt.tight_layout()
        plt.savefig("shap_summary.png", bbox_inches='tight', dpi=300)
        plt.close()
        

        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_3d.reshape(-1, self.n_features),
            feature_names=feature_names,
            plot_type="bar",
            show=False
        )
        plt.tight_layout()
        plt.savefig("shap_summary_bar.png", bbox_inches='tight', dpi=300)
        plt.close()
        
        return shap_values 