import torch

from gradientflow import riesz_kernel
from networks import (
    DEQ,
    MaxClassifier,
    TorchDEQModel,
    FFN,
    MultiLayerNormalizingFlow,
    CrossAttentionEncoder,
)

from utils import target_identity, target_completion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_hyperparams_classifier_modelnet(args):
    target_fn = target_identity

    init_kwargs = {"len_z": 10, "adaptive_gaussian": False}

    mmd_kwargs = {
        "beta": 0,
        "gamma_beta": 0,
        "optimizer": "sgd",
        "optim_kwargs": {"lr": 5},
        "scheduler_kwargs": {"gamma": 1.0},
        "stop_type": "mmd",
        "mask": 10,
        "rescale_losses": False,
        "autodiff": True,
        "rescale_autodiff": True,
        "reg": 0.0,
        "kernel": riesz_kernel,
    }

    tol = 1e-4

    scheduler_kwargs = {"step_size": 8, "gamma": 0.1}

    config = {
        "mmd_kwargs": mmd_kwargs,
        "model": DEQ.__name__,
        "tol": tol,
        "f_max_iter": 200,
        "solver": "mmd",
        "mask": False,
        "batch_size": 64,
        "classifier": "max",
        "init_kwargs": init_kwargs,
        "track_accuracy": True,
        "x_encoder": True,
        "model_size": "p128",
        "scheduler_kwargs": scheduler_kwargs,
        "adaptive_max_iter": True,
        "dataset": "modelnet",
    }

    d_z = 128

    classifier = MaxClassifier
    classifier_kwargs = {}
    loss_fn = "cross_entropy"

    deq_kwargs = {
        "f_solver": "mmd",
        "f_max_iter": 200,
        "f_stop_mode": args.stop_mode,
        "f_tol": tol,
    }

    d_encoder = d_z

    deq_model_kwargs = {
        "hidden_dim_equiv_1": 16,
        "hidden_dim_equiv_2": 128,
        "hidden_dim_equiv_3": 16,
        "d_encoder": d_encoder,
        "num_heads": 32,
        "num_layers": 3,
        "num_layers_self_encoders": 1,
        "dim_feedforward": 4 * d_z,
        "bilinear": True,
        "FFNNetwork": FFN,
        "ffn_hidden": 4 * d_z,
        "d_z": d_z,
        "d_x": 3,
        "encoder": CrossAttentionEncoder,
        "z_encoder": True,
        "bilinear_pushforward": True,
    }
    
    model = TorchDEQModel(
        deq_model_class=DEQ,
        deq_model_kwargs=deq_model_kwargs,
        deq_kwargs=deq_kwargs,
        d_z=d_z,
        d_x=3,
        n_classes=40,
        init_kwargs=init_kwargs,
        solver_kwargs=mmd_kwargs,
        mask=False,
        classifier=classifier,
        classifier_kwargs=classifier_kwargs,
    )

    print(model)
    config["classifier_num_params"] = model.classifier.num_params
    config["deq_model_num_params"] = model.deq_model.num_params
    config["deq_model_kwargs"] = deq_model_kwargs
    config["d_z"] = d_z

    model.to(device)
    model.deq_model.to(device)
    model.classifier.to(device)
    model.eval()

    return model, target_fn, loss_fn, config


def get_hyperparams_classifier_mnist(args):
    target_fn = target_identity

    init_kwargs = {"len_z": 10, "adaptive_gaussian": False}

    scheduler_kwargs = {"step_size": 2, "gamma": 0.2}
    mmd_kwargs = {
        "beta": 0,
        "gamma_beta": 0,
        "optimizer": "sgd",
        "optim_kwargs": {"lr": 5},
        "scheduler_kwargs": {"gamma": 1.0},
        "stop_type": "mmd",
        "mask": False,
        "rescale_losses": False,
        "autodiff": True,
        "rescale_autodiff": True,
        "reg": 0.0,
        "kernel": riesz_kernel,
    }
    tol = 1e-4

    config = {
        "mmd_kwargs": mmd_kwargs,
        "model": DEQ.__name__,
        "tol": tol,
        "f_max_iter": 200,
        "solver": "mmd",
        "mask": False,
        "batch_size": 64,
        "classifier": "max",
        "init_kwargs": init_kwargs,
        "track_accuracy": True,
        "x_encoder": True,
        "model_size": "p128",
        "scheduler_kwargs": scheduler_kwargs,
        "adaptive_max_iter": True,
        "dataset": "modelnet",
    }

    d_z = 128

    classifier = MaxClassifier
    classifier_kwargs = {}
    loss_fn = "cross_entropy"

    deq_kwargs = {
        "f_solver": "mmd",
        "f_max_iter": 200,
        "f_stop_mode": args.stop_mode,
        "f_tol": tol,
    }

    d_encoder = d_z

    deq_model_kwargs = {
        "hidden_dim_equiv_1": 16,
        "hidden_dim_equiv_2": 128,
        "hidden_dim_equiv_3": 16,
        "d_encoder": d_encoder,
        "num_heads": 32,
        "num_layers": 1,
        "num_layers_self_encoders": 1,
        "dim_feedforward": 4 * d_z,
        "bilinear": True,
        "FFNNetwork": FFN,
        "ffn_hidden": 4 * d_z,
        "d_z": d_z,
        "encoder": CrossAttentionEncoder,
        "z_encoder": True,
        "bilinear_pushforward": True,
    }
    

    model = TorchDEQModel(
        deq_model_class=DEQ,
        deq_model_kwargs=deq_model_kwargs,
        deq_kwargs=deq_kwargs,
        d_z=d_z,
        init_kwargs=init_kwargs,
        solver_kwargs=mmd_kwargs,
        mask=False,
        classifier=classifier,
        classifier_kwargs=classifier_kwargs,
    )
    

    print(model)
    config["classifier_num_params"] = model.classifier.num_params
    config["deq_model_num_params"] = model.deq_model.num_params
    config["deq_model_kwargs"] = deq_model_kwargs
    config["d_z"] = d_z

    model.to(device)
    model.deq_model.to(device)
    model.classifier.to(device)
    model.eval()

    return model, target_fn, loss_fn, config


def get_hyperparams_completion_mnist(args):
    target_fn = target_completion
    loss_fn = "mmd"

    d_z = 128

    dim_sequence = [2, d_z]
    embedding_kwargs = {"dim_sequence": dim_sequence}
    fix_embedding_out = False

    scheduler_kwargs = {"step_size": 2, "gamma": 0.2}

    init_kwargs = {
        "mode": "completion",
        "len_z": "max",
        "adaptive_gaussian": True,
        "scale": 1.275,
        "p": 2,
    }

    mmd_kwargs = {
        "beta": 0,
        "gamma_beta": 0,
        "optimizer": "sgd",
        "optim_kwargs": {"lr": 5},
        "scheduler_kwargs": {"gamma": 1.0},
        "stop_type": "mmd",
        "mask": True,
        "rescale_losses": False,
        "autodiff": True,
        "rescale_autodiff": True,
        "reg": 0.0,
        "kernel": riesz_kernel
    }

    tol = 1e-4

    config = {
        "mmd_kwargs": mmd_kwargs,
        "model": DEQ.__name__,
        "tol": tol,
        "f_max_iter": 200,
        "solver": "mmd",
        "mask": True,
        "batch_size": 64,
        "classifier": None,
        "init_kwargs": init_kwargs,
        "track_accuracy": False,
        "x_encoder": True,
        "model_size": "p128",
        "target_kwargs": {"mode": "cluster", "n_clusters": 2, "d_clusters": 0.6},
        "fix_all": False,
        "fix_encoders": True,
        "scheduler_kwargs": scheduler_kwargs,
        "set_limits_points": True,
        "adaptive_max_iter": True,
        "dataset": "mnist",
        "regularizing_fn": "entropy_kde_gaussian",
        "regularizing_const": 1e-6,
    }

    deq_kwargs = {
        "f_solver": "mmd",
        "f_max_iter": 200,
        "f_stop_mode": args.stop_mode,
        "f_tol": tol,
    }

    d_encoder = d_z
    ffn_hidden = 4 * d_encoder

    deq_model_kwargs = {
        "hidden_dim_equiv_1": 16,
        "hidden_dim_equiv_2": 128,
        "hidden_dim_equiv_3": 16,
        "d_encoder": d_encoder,
        "num_heads": 32,
        "num_layers": 3,
        "num_layers_self_encoders": 1,
        "dim_feedforward": 4 * d_encoder,
        "bilinear": True,
        "FFNNetwork": FFN,
        "ffn_hidden": ffn_hidden,
        "d_z": d_encoder,
        "encoder": CrossAttentionEncoder,
        "z_encoder": True,
        "bilinear_pushforward": True,
    }
    

    model = TorchDEQModel(
        deq_model_class=DEQ,
        deq_model_kwargs=deq_model_kwargs,
        deq_kwargs=deq_kwargs,
        d_z=d_z,
        embedding_model=MultiLayerNormalizingFlow,
        embedding_kwargs=embedding_kwargs,
        fix_embedding_out=fix_embedding_out,
        init_kwargs=init_kwargs,
        solver_kwargs=mmd_kwargs,
        mask=True,
    )
    

    print(model)
    config["deq_model_num_params"] = model.deq_model.num_params
    config["deq_model_kwargs"] = deq_model_kwargs
    config["embedding"] = MultiLayerNormalizingFlow.__name__
    config["embedding_num_params"] = model.embedding_model.num_params
    config["embedding_dims"] = embedding_kwargs["dim_sequence"]
    config["d_z"] = d_z

    model.to(device)
    model.deq_model.to(device)
    model.eval()

    return model, target_fn, loss_fn, config
